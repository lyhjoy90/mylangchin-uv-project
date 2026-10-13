"""
===================================================================
[프로젝트 2] IC/교차로 설계 지침 웹 어시스턴트 (02_ic_design_assistant/app.py)
===================================================================
- 역할: Gradio UI 기반 설계 지침/편람 PDF QA, 동적 RAG 파라미터 조절
- 실행 명령어: python 이영후_관련/02_ic_design_assistant/app.py
===================================================================
"""
## 특징: gr.Slider와 gr.Accordion을 활용한 청크 크기/검색 개수(k) 동적 조절 및 교차로·IC 설계 지침 검색기
import os
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

class ICDesignEngine:
    def __init__(self):
        self.embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        self.llm = ChatUpstage(model="solar-pro3", temperature=0.0)

    def find_default_pdf(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith(".pdf"):
                    return os.path.join(data_dir, file)
        return None

    def build_retriever(self, pdf_path: str, chunk_size: int, search_k: int):
        if not pdf_path or not os.path.exists(pdf_path):
            return None
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        splits = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(splits, self.embeddings)
        return vector_store.as_retriever(search_kwargs={"k": search_k})

    def generate_answer(self, retriever, question: str):
        template = """[시스템 역할]
당신은 도로 및 교차로/IC 설계 지침 전문 AI 어시스턴트입니다.
아래 제공된 [설치/설계 지침] 문서 내용만을 근거로 질문에 답변하세요.

[답변 수칙]:
1. 테이퍼 길이, 수직여유고, 곡선반경 등 수식 및 규격 관련 기준이 있을 경우 정밀하게 기재하세요.
2. 문서에 없는 기준은 절대 추측하여 답변하지 마세요.

[설계 지침]:
{context}

[질문]:
{question}

[답변]:"""
        prompt = ChatPromptTemplate.from_template(template)
        format_docs = lambda docs: "\n\n".join(d.page_content for d in docs)
        
        answer_chain = (
            RunnableLambda(lambda x: {"context": format_docs(x["docs"]), "question": x["question"]})
            | prompt | self.llm | StrOutputParser()
        )
        chain = RunnableParallel(docs=retriever, question=RunnablePassthrough()).assign(answer=answer_chain)
        result = chain.invoke(question)
        
        sources = set()
        for doc in result["docs"]:
            page = doc.metadata.get("page")
            if isinstance(page, int):
                sources.add(f"Page {page + 1}")
        
        source_str = f"\n\n📌 **참고 출처**: {', '.join(sorted(sources))}" if sources else ""
        return f"{result['answer']}{source_str}"

engine = ICDesignEngine()

def respond(message, chat_history, pdf_file, chunk_size, search_k, session_retriever):
    if not message or not message.strip():
        return chat_history, "", session_retriever
    
    if session_retriever is None:
        target_pdf = pdf_file.name if pdf_file and hasattr(pdf_file, 'name') else engine.find_default_pdf()
        if target_pdf and os.path.exists(target_pdf):
            session_retriever = engine.build_retriever(target_pdf, int(chunk_size), int(search_k))
        else:
            bot_message = "⚠️ 좌측에서 교차로/IC 설계 지침 PDF 문서를 선택해주세요."
            return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": bot_message}], "", session_retriever

    try:
        bot_message = engine.generate_answer(session_retriever, message)
    except Exception as e:
        bot_message = f"⚠️ 오류 발생: {e}"

    chat_history = chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": bot_message}]
    return chat_history, "", session_retriever

def create_ui():
    with gr.Blocks(title="교차로/IC 설계 지침 모니터링 어시스턴트") as demo:
        gr.Markdown("# 🛣️ IC 및 교차로 설계 지침 RAG 어시스턴트")
        gr.Markdown("설계 지침 PDF 문서를 기반으로 테이퍼 길이, 수직 여유고 등 수식 규격을 조회합니다.")
        
        session_retriever = gr.State(None)
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_input = gr.File(label="설계 편람 PDF 업로드", file_types=[".pdf"])
                
                with gr.Accordion("⚙️ RAG 파라미터 설정", open=True):
                    chunk_size = gr.Slider(minimum=300, maximum=1500, value=800, step=100, label="청크 크기 (Chunk Size)")
                    search_k = gr.Slider(minimum=1, maximum=10, value=4, step=1, label="검색 문서 수 (Search K)")
            
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="💬 대화창", height=500)
                msg = gr.Textbox(label="질문 입력", placeholder="예: 단일로 변속차로의 테이퍼 최소 길이 및 수직여유고 기준을 알려주세요.", lines=2)
                with gr.Row():
                    submit_btn = gr.Button("📤 질문하기", variant="primary")
                    clear_btn = gr.Button("🗑️ 대화 초기화")

        inputs = [msg, chatbot, pdf_input, chunk_size, search_k, session_retriever]
        outputs = [chatbot, msg, session_retriever]
        
        submit_btn.click(respond, inputs, outputs)
        msg.submit(respond, inputs, outputs)
        clear_btn.click(lambda: ([], "", None), None, outputs)
        
        # 설정 변경 시 세션 초기화
        pdf_input.change(lambda: None, None, session_retriever)
        chunk_size.change(lambda: None, None, session_retriever)
        search_k.change(lambda: None, None, session_retriever)
        
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7862)