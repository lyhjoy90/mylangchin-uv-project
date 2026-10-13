"""
===================================================================
[프로젝트 6] 조합 정관 및 감정평가 분쟁 검토 웹 (06_urban_redev_advisor/app.py)
===================================================================
- 역할: Gradio UI 기반 조합 정관 PDF/TXT 업로드 및 분쟁 검토 QA (Port 7866)
- 실행 명령어: python 이영후_관련/06_urban_redev_advisor/app.py
===================================================================
"""

import os
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

class UrbanRedevAppEngine:
    def __init__(self):
        self.embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        self.llm = ChatUpstage(model="solar-pro3", temperature=0.0)

    def find_default_docs(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith(".pdf") or file.endswith(".txt") or file.endswith(".md"):
                    return os.path.join(data_dir, file)
        return None

    def build_retriever(self, file_path: str):
        if not file_path or not os.path.exists(file_path):
            return None
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")
            
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        splits = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(splits, self.embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 3})

    def generate_answer(self, retriever, question: str):
        template = """[시스템 역할]
당신은 도시정비사업 조합 정관 및 감정평가 분쟁 전문 법률 어시스턴트입니다.
아래 제공된 [조합 정관 규정]만을 기반으로 질문에 답변하세요.

[답변 작성 수칙]:
1. 조항 번호와 정족수, 감정평가 기준을 정밀하게 언급하세요.
2. 규정에 없는 사항은 추측하지 마세요.

[조합 정관 규정]:
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

engine = UrbanRedevAppEngine()

def respond(message, chat_history, file_input, session_retriever):
    if not message or not message.strip():
        return chat_history, "", session_retriever
    
    if session_retriever is None:
        target_file = None
        if file_input is not None:
            target_file = file_input.name if hasattr(file_input, 'name') else file_input
        else:
            target_file = engine.find_default_docs()

        if target_file and os.path.exists(target_file):
            session_retriever = engine.build_retriever(target_file)
        else:
            bot_message = "⚠️ 좌측에서 조합 정관 문서(PDF/TXT)를 업로드해주세요."
            return chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": bot_message}], "", session_retriever
        
    try:
        bot_message = engine.generate_answer(session_retriever, message)
    except Exception as e:
        bot_message = f"⚠️ 오류 발생: {e}"

    chat_history = chat_history + [{"role": "user", "content": message}, {"role": "assistant", "content": bot_message}]
    return chat_history, "", session_retriever

def create_ui():
    with gr.Blocks(title="조합 정관 및 분쟁 검토 AI 어시스턴트") as demo:
        gr.Markdown("# ⚖️ 도시정비 조합 정관 & 감정평가 분쟁 검토 AI")
        gr.Markdown("재개발·재건축 조합 정관 문서를 업로드하고 의결 정족수, 감정평가 산정 방식, 위반 이자율 등을 질의하세요.")
        
        session_retriever = gr.State(None)
        
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="조합 정관 문서 업로드 (.pdf, .txt)", file_types=[".pdf", ".txt", ".md"])
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="💬 대화창", height=500)
                msg = gr.Textbox(label="질문 입력", placeholder="예: 사업시행계획서 변경 시 필요한 조합원 동의 정족수와 감정평가액 산정 기준을 알려주세요.", lines=2)
                with gr.Row():
                    submit_btn = gr.Button("📤 질문하기", variant="primary")
                    clear_btn = gr.Button("🗑️ 대화 초기화")
        
        inputs = [msg, chatbot, file_input, session_retriever]
        outputs = [chatbot, msg, session_retriever]
        
        submit_btn.click(respond, inputs, outputs)
        msg.submit(respond, inputs, outputs)
        clear_btn.click(lambda: ([], "", None), None, outputs)
        
        file_input.change(lambda: None, None, session_retriever)
        
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7866)