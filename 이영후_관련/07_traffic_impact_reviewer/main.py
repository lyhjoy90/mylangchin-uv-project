"""
===================================================================
[프로젝트 7] 도로점용 공사 교통소통대책서 검토 에이전트 (07_traffic_impact_reviewer/main.py)
===================================================================
- 역할: 공사 소통대책서 분석, 지침 대조 미비점 판별 및 개조식 검토의견서 생성
- 실행 명령어: python 이영후_관련/07_traffic_impact_reviewer/main.py
===================================================================
"""

import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class TrafficImpactEngine:
    def __init__(self):
        self.embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        self.llm = ChatUpstage(model="solar-pro3", temperature=0.0)
        self.vector_store = None
        self.retriever = None

    def build_db(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        docs = []

        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file)
                if file.endswith(".pdf"):
                    docs.extend(PyPDFLoader(file_path).load())
                elif file.endswith(".txt") or file.endswith(".md"):
                    docs.extend(TextLoader(file_path, encoding="utf-8").load())

        if not docs:
            from langchain_core.documents import Document
            docs = [
                Document(page_content="[도로점용 공사 소통대책 지침] 1. 차로 차단 시 공사 안내 표지판은 최소 50m 전방에 설치해야 함. 2. 야간 공사 시 신호수 배치 및 LED 점멸식 가방형 안내판 필수. 3. 보행자 우회도로 확보 시 최소 폭원 1.5m 이상을 확보하고 신호수를 배치해야 함.")
            ]

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(docs)
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

    def review_plan(self, plan_text: str):
        if not plan_text or not plan_text.strip():
            return "공사 소통대책서 내용을 입력해 주세요."

        retrieved_docs = self.retriever.invoke(plan_text)
        context_str = "\n\n".join(d.page_content for d in retrieved_docs)

        template = """[시스템 역할]
당신은 지자체 교통행정과 도로점용 공사 교통소통대책 검토관입니다.
제공된 [도로점용 공사 소통대책 지침]을 바탕으로 신청된 [공사 소통대책서 계획]의 미비점을 검토하고, 공문서 결재용 개조식 검토의견서를 작성하세요.

[작성 양식]:
1. [사업 개요 및 검토 목적]
2. [지침 대비 주요 미비 및 보완 요청 사항] (개조식 - )
3. [종합 검토 의견 (적합/조건부 승인/재검토)]

[소통대책 지침]:
{context}

[공사 소통대책서 내용]:
{plan_text}

[행정 개조식 검토의견서]:"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context_str, "plan_text": plan_text})

engine = TrafficImpactEngine()

if __name__ == "__main__":
    engine.build_db()
    sample_plan = "공사 구간: 옥녀교차로 일원, 차로차단: 1차로 차단, 안전시설: 20m 전방 안내판 설치, 보행자 우회 폭: 1.0m 확보 예정."
    print("=== [도로점용 공사 소통대책서 검토의견서] ===")
    print(engine.review_plan(sample_plan))