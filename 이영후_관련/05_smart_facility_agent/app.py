"""
===================================================================
[프로젝트 5] 스마트 시설물 입지/규격 평가 대시보드 (05_smart_facility_agent/app.py)
===================================================================
- 역할: 후보지 현황 입력 시 정량 평가표 및 기술 규격 검토 보고서 생성
- 실행 명령어: python 이영후_관련/05_smart_facility_agent/app.py
===================================================================
"""

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

class SmartFacilityEngine:
    def __init__(self):
        self.embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        self.llm = ChatUpstage(model="solar-pro3", temperature=0.0)
        self.retriever = None
        self.build_db()

    def build_db(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        docs = []
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file)
                if file.endswith(".md") or file.endswith(".txt"):
                    docs.extend(TextLoader(file_path, encoding="utf-8").load())
                elif file.endswith(".pdf"):
                    docs.extend(PyPDFLoader(file_path).load())
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        splits = splitter.split_documents(docs)
        vector_store = FAISS.from_documents(splits, self.embeddings)
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    def evaluate(self, candidate_info: str):
        if not candidate_info or not candidate_info.strip():
            return "후보지 현황 데이터를 입력해주세요."

        retrieved_docs = self.retriever.invoke(candidate_info)
        context_str = "\n\n".join(d.page_content for d in retrieved_docs)

        template = """[시스템 역할]
당신은 스마트도시 시설물 입지 선정 평가위원입니다.
제공된 [설치 가이드라인 규정]과 [후보지 현황 데이터]를 비교 분석하여 입지 적합성을 평가하고 정량적 점수 표 및 종합 의견 보고서를 작성하세요.

[설치 가이드라인]:
{context}

[후보지 데이터]:
{candidate_info}

[입지/규격 평가 보고서]:"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context_str, "candidate_info": candidate_info})

engine = SmartFacilityEngine()

def create_ui():
    with gr.Blocks(title="스마트 시설물 입지 및 규격 평가 시스템") as demo:
        gr.Markdown("# 🏢 스마트 시설물 입지 배점 및 규격 진단 대시보드")
        gr.Markdown("스마트 주차안내, VDS, CEMS 시설물의 현장 후보지 조건을 입력하면 입지 적합성 평가 보고서를 자동 작성합니다.")

        with gr.Row():
            with gr.Column(scale=1):
                candidate_input = gr.Textbox(
                    label="후보지 현황 데이터 입력",
                    placeholder="예: 후보지 A: 일 평균 교통량 15,000대, 공영주차장 200m 이내 위치, 상업지역 중심가 위치.",
                    lines=6
                )
                eval_btn = gr.Button("📋 입지/규격 평가 보고서 생성", variant="primary")
            
            with gr.Column(scale=2):
                report_output = gr.Markdown("좌측에 후보지 현황을 입력한 후 평가 버튼을 눌러주세요.")

        eval_btn.click(engine.evaluate, inputs=candidate_input, outputs=report_output)

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7865)