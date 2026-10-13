"""
===================================================================
[프로젝트 3] 교통 문제 해결 대책 DB 추천 웹 (03_traffic_solution_db/app.py)
===================================================================
- 역할: 과거 교통 정체/사고 개선 사례 DB 검색 및 탭 분리 기반 AI 추천
- 실행 명령어: python 이영후_관련/03_traffic_solution_db/app.py
===================================================================
"""
## 특징: gr.Tabs를 적용하여 1) AI 대책 추천 답변과 2) 검색된 우수사례 원문 DB를 탭으로 분리 제공
import os
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class TrafficSolutionEngine:
    def __init__(self):
        self.embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        self.llm = ChatUpstage(model="solar-pro3", temperature=0.2)
        self.vector_store = None
        self.retriever = None
        self.build_default_db()

    def build_default_db(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        docs = []
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file)
                if file.endswith(".txt") or file.endswith(".md"):
                    docs.extend(TextLoader(file_path, encoding="utf-8").load())
                elif file.endswith(".pdf"):
                    docs.extend(PyPDFLoader(file_path).load())
        
        if not docs:
            from langchain_core.documents import Document
            docs = [Document(page_content="[사례1] 교차로 꼬리물기: 감응신호 및 Yellow 앞당김 적용으로 대기행렬 25% 감소.")]

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(docs)
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 6})

    def search_and_recommend(self, problem_type: str, detail_text: str):
        query = f"[{problem_type}] {detail_text}"
        retrieved_docs = self.retriever.invoke(query)
        
        # 1. 원문 DB 정리
        raw_db_text = ""
        for i, doc in enumerate(retrieved_docs, 1):
            raw_db_text += f"### 📄 사례 {i}\n{doc.page_content}\n\n---\n"

        # 2. AI 추천서 생성
        template = """당신은 교통공학 전문 컨설턴트입니다.
아래 [과거 개선 우수사례 DB]를 분석하여 [현장 교통 문제]에 대한 실무 맞춤형 해결 대책 보고서를 작성하세요.

[과거 개선 사례 DB]:
{context}

[현장 문제]:
{query}

[교통 개선 대책 추천서]:"""
        prompt = ChatPromptTemplate.from_template(template)
        context_str = "\n\n".join(d.page_content for d in retrieved_docs)
        chain = prompt | self.llm | StrOutputParser()
        
        ai_recommendation = chain.invoke({"context": context_str, "query": query})
        return ai_recommendation, raw_db_text

engine = TrafficSolutionEngine()

def create_ui():
    with gr.Blocks(title="교통 문제 해결 대책 DB 검색 시스템") as demo:
        gr.Markdown("# 🚦 교통 문제 해결 우수사례 DB & AI 대책 추천")
        gr.Markdown("교통 정체/사고 유형을 선택하고 현장 상황을 입력하면 유사 우수사례 DB 기반 개선안을 제안합니다.")
        
        with gr.Row():
            with gr.Column(scale=1):
                problem_type = gr.Dropdown(
                    choices=["신호 교차로 정체", "병목 구간/합류부 정체", "보행자 사고 다발", "무신호 교차로 안전"],
                    value="신호 교차로 정체",
                    label="교통 문제 유형 선택"
                )
                detail_text = gr.Textbox(
                    label="현장 상태 및 정체 내용 입력",
                    placeholder="예: 출퇴근 시간대 좌회전 대기행렬이 주간선 도로까지 침범하여 꼬리물기가 발생함.",
                    lines=4
                )
                search_btn = gr.Button("🔍 대책 검색 및 AI 분석", variant="primary")

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("🤖 AI 종합 추천서"):
                        recommend_output = gr.Markdown("좌측에 문제 상황을 입력한 후 검색 버튼을 눌러주세요.")
                    with gr.TabItem("📑 검색된 우수사례 원문 DB (k=6)"):
                        raw_db_output = gr.Markdown("유사 우수사례 검색 결과가 이곳에 표시됩니다.")

        search_btn.click(
            engine.search_and_recommend,
            inputs=[problem_type, detail_text],
            outputs=[recommend_output, raw_db_output]
        )

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7863)