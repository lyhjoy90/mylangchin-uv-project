"""
===================================================================
[프로젝트 3] 교통 문제 해결 대책 DB 구축 및 추천 시스템 (03_traffic_solution_db/main.py)
===================================================================
- 목표: 과거 교통 정체/사고 우수 개선 사례 DB를 기반으로 현장 맞춤형 대책안 추천
- 특징: 위치 유형(IC, 교차로 등) 및 문제 유형(병목, 딜레마존 등) 메타데이터 검색 및 문서 자동 로드 지원
===================================================================
"""

import os
from dotenv import load_dotenv

# .env 파일에서 UPSTAGE_API_KEY 로드
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class TrafficSolutionDB:
    def __init__(self):
        self.embeddings = UpstageEmbeddings(model="embedding-query")
        self.llm = ChatUpstage(model="solar-pro")
        self.vector_store = None
        self.retriever = None

    def load_documents_and_build_db(self, data_dir: str = "data"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_data_dir = os.path.join(base_dir, data_dir)

        docs = []
        print(f"[*] 데이터 폴더 탐색 중: {target_data_dir}")

        if os.path.exists(target_data_dir):
            for file in os.listdir(target_data_dir):
                file_path = os.path.join(target_data_dir, file)
                if file.endswith(".pdf"):
                    print(f"[*] PDF 개선사례 읽는 중: {file}")
                    loader = PyPDFLoader(file_path)
                    docs.extend(loader.load())
                elif file.endswith(".txt") or file.endswith(".md"):
                    print(f"[*] 텍스트 개선사례 읽는 중: {file}")
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs.extend(loader.load())

        # data 폴더에 파일이 없을 경우 사용되는 예시 데이터
        if not docs:
            print("[!] data 폴더에 파일이 없어 기본 내장 사례 데이터를 사용합니다.")
            docs = [
                Document(
                    page_content="[사례 A] 고속도로 IC 진출입부 병목 구간: 분리형 유도선 표시 및 노면 색상유도선 설치로 합류 사고 35% 감소.",
                    metadata={"location_type": "IC램프", "issue": "병목"}
                ),
                Document(
                    page_content="[사례 B] 도심 4차로 교차로 딜레마존 정체: 제1주신호등과 제2주신호등의 앙각을 조정하고 올레드(All-Red) 시간을 1초 연장함.",
                    metadata={"location_type": "교차로", "issue": "딜레마존"}
                )
            ]

        print(f"[*] 총 {len(docs)}개 문서 로드 완료. FAISS 임베딩 진행 중...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        splits = splitter.split_documents(docs)

        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})
        print("[+] Vector DB 구축 완료!\n")

    def get_chain(self):
        template = """[시스템 역할]
당신은 교통 안전 및 정체 개선대책 기술 컨설턴트입니다.
과거 우수 개선 사례를 바탕으로 지점 문제 해결을 위한 정형화된 대책안을 추천하세요.

[과거 사례 DB]:
{context}

[현장 문제]:
{question}

[추천 개선대책]:"""
        prompt = ChatPromptTemplate.from_template(template)
        return (
            {"context": self.retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

if __name__ == "__main__":
    db = TrafficSolutionDB()
    db.load_documents_and_build_db()
    chain = db.get_chain()

    query = "고속도로 IC 램프 합류 구간에서 정체 및 병목 현상이 발생할 때 적용 가능한 대책을 알려주세요."
    print(f"[질문]: {query}\n")
    print("[답변]:")
    print(chain.invoke(query))