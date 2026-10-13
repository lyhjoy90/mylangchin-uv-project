"""  이영후_관련/01_urban_rag/app.py
===================================================================
[프로젝트 1] 도시정비 민원 및 규정 QA 웹 앱 (01_urban_rag/app.py)
===================================================================
- 역할: Gradio UI 기반 PDF 문서 로드 및 대화형 법령 QA 웹 인터페이스
===================================================================
"""

import os
from dotenv import load_dotenv

# .env 로드
load_dotenv()

import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

# RAG 백엔드 세션 클래스
class UrbanRAGWebEngine:
    def __init__(self):
        self.embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        self.llm = ChatUpstage(model="solar-pro3", temperature=0.0)

    def find_default_pdf(self):
        """data 폴더 내의 기본 PDF 파일 경로 탐색"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith(".pdf"):
                    return os.path.join(data_dir, file)
        return None

    def build_retriever_from_pdf(self, pdf_path: str):
        if not pdf_path or not os.path.exists(pdf_path):
            return None
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(splits, self.embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 4})

    def generate_answer(self, retriever, question: str):
        template = """[시스템 역할]
당신은 도시정비사업 관련 법령 및 가이드라인 전문 AI 어시스턴트입니다.
아래 제공된 [참고 규정]만을 기반으로 질문에 답변하세요.

[답변 작성 수칙]:
1. 해당 조항의 명칭과 시행일자/부칙 내용을 반드시 언급하여 검증하세요.
2. 참고 규정에 없는 내용은 추측하여 답변하지 마세요.

[참고 규정]:
{context}

[질문]:
{question}

[답변]:"""
        prompt = ChatPromptTemplate.from_template(template)
        format_docs = lambda docs: "\n\n".join(d.page_content for d in docs)
        
        answer_chain = (
            RunnableLambda(lambda x: {"context": format_docs(x["docs"]), "question": x["question"]})
            | prompt
            | self.llm
            | StrOutputParser()
        )
        chain = RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(answer=answer_chain)
        result = chain.invoke(question)
        
        # 출처 페이지 매핑
        sources = set()
        for doc in result["docs"]:
            page = doc.metadata.get("page")
            if isinstance(page, int):
                sources.add(f"Page {page + 1}")
        
        source_str = f"\n\n📌 **참고 출처**: {', '.join(sorted(sources))}" if sources else ""
        return f"{result['answer']}{source_str}"

engine = UrbanRAGWebEngine()

# Gradio 이벤트 핸들러
def respond(message, chat_history, pdf_file, session_retriever):
    if not message or not message.strip():
        return chat_history, "", session_retriever
    
    # 세션에 DB가 없는 경우 (첫 질문 시)
    if session_retriever is None:
        target_pdf = None
        if pdf_file is not None:
            target_pdf = pdf_file.name if hasattr(pdf_file, 'name') else pdf_file
        else:
            target_pdf = engine.find_default_pdf()

        if target_pdf and os.path.exists(target_pdf):
            session_retriever = engine.build_retriever_from_pdf(target_pdf)
        else:
            bot_message = "⚠️ 좌측에서 도시정비 관련 법령/지침 PDF 파일을 먼저 업로드해주세요."
            return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": bot_message}], "", session_retriever
        
    try:
        bot_message = engine.generate_answer(session_retriever, message)
    except Exception as e:
        bot_message = f"⚠️ 오류 발생: {e}"

    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": bot_message}
    ]
    return chat_history, "", session_retriever

# UI 생성
def create_ui():
    with gr.Blocks(title="도시정비 민원 및 규정 QA 시스템") as demo:
        gr.Markdown("# 🏢 도시정비 민원 및 규정 QA 웹 어시스턴트")
        gr.Markdown("도시정비 관련 법령/지침 PDF를 업로드(또는 data 폴더 문서 사용)하고 질문하면 부칙 및 조항 근거와 함께 답변합니다.")
        
        session_retriever = gr.State(None)
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="법령/지침 PDF 업로드", file_types=[".pdf"])
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="💬 대화창", height=500)
                msg = gr.Textbox(label="질문 입력", placeholder="예: 정비구역 지정 기본요건과 부칙 시행일자를 알려주세요.", lines=2)
                with gr.Row():
                    submit_btn = gr.Button("📤 질문하기", variant="primary")
                    clear_btn = gr.Button("🗑️ 대화 초기화")
        
        inputs = [msg, chatbot, pdf_input, session_retriever]
        outputs = [chatbot, msg, session_retriever]
        
        submit_btn.click(respond, inputs, outputs)
        msg.submit(respond, inputs, outputs)
        clear_btn.click(lambda: ([], "", None), None, outputs)
        
        # 새로운 PDF 파일이 업로드되면 세션 초기화하여 재구축되도록 설정
        pdf_input.change(lambda: None, None, session_retriever)
        
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1")