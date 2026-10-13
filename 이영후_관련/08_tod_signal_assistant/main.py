# 📌 2. 프로젝트 8: `08_tod_signal_assistant`

#**지자체 교통신호 민원 분석 & TOD 조정 권고 AI** (`Port 7868`)

"""
===================================================================
[프로젝트 8] 교통신호 민원 분석 & TOD 조정 권고 AI (08_tod_signal_assistant/main.py)
===================================================================
- 역할: 민원 분석, 신호운영 편람 기반 TOD 신호값 조정 제안 및 대민 답변서 생성
- 실행 명령어: python 이영후_관련/08_tod_signal_assistant/main.py
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

class TODSignalEngine:
    def __init__(self):
        self.embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        self.llm = ChatUpstage(model="solar-pro3", temperature=0.1)
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
                Document(page_content="[경찰청 교통신호운영 편람] 1. 교차로 좌회전 대기행렬이 간선도로를 침범할 경우 주시간대(TOD) 좌회전 녹색시간을 3~5초 연장 검토. 2. 꼬리물기 발생 시 교차로 간 연동값(Offset) 재조정 및 All-Red(전적신호) 1~2초 연장. 3. 보행자 보행시간 부족 민원 시 보행속도 0.9m/s 기준 최소 보행시간 재산정.")
            ]

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = splitter.split_documents(docs)
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

    def analyze_and_recommend(self, complaint_text: str):
        if not complaint_text or not complaint_text.strip():
            return "신호 관련 민원 내용을 입력해 주세요."

        retrieved_docs = self.retriever.invoke(complaint_text)
        context_str = "\n\n".join(d.page_content for d in retrieved_docs)

        template = """[시스템 역할]
당신은 하남시/지자체 교통정보센터 신호운영 담당 AI 공학 컨설턴트입니다.
시민의 [교통신호 민원]을 분석하여 1) 원인 분석, 2) TOD 신호 주기 조정 권고안, 3) 공식 대민 답변글을 작성하세요.

[신호 운영 지침]:
{context}

[시민 민원 내용]:
{complaint_text}

[신호 조정 권고 및 대민 답변서]:
1. 📍 [민원 요약 및 정체 원인 분석]
- 대상 교차로 및 발생 시간대:
- 정체/부작용 원인:

2. 🚦 [TOD 신호 주기 조정 권고안] (Offset, 녹색시간, All-Red 등)
- 신호 조정안:

3. ✉️ [공식 대민 답변글 (행정 보고서/답변체)]
- """
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"context": context_str, "complaint_text": complaint_text})

engine = TODSignalEngine()

if __name__ == "__main__":
    engine.build_db()
    sample_complaint = "출근시간마다 진등교차로 좌회전 신호가 너무 짧아서 차가 주간선도로까지 길게 늘어서고 꼬리물기가 심해요! 당장 신호 늘려주세요!"
    print("=== [교통신호 민원 분석 & TOD 조정 권고] ===")
    print(engine.analyze_and_recommend(sample_complaint))