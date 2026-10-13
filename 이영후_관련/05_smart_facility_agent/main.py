"""
===================================================================
[프로젝트 5] 스마트 시설물 입지 평가 보조 에이전트 (05_smart_facility_agent/main.py)
===================================================================
- 목표: 스마트 주차 정보 수집 시스템 등 도시/교통 시설물 설치 가이드라인 규정과
        후보지 현황 데이터를 비교 분석하여 입지 적합성 평가 보고서 작성
- 특징: RAG 기반 입지 규정 검색 + 후보지 조건 평가 체인 결합
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

class SmartFacilityAgent:
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
                    print(f"[*] PDF 입지 지침 읽는 중: {file}")
                    loader = PyPDFLoader(file_path)
                    docs.extend(loader.load())
                elif file.endswith(".txt") or file.endswith(".md"):
                    print(f"[*] 텍스트 가이드라인 읽는 중: {file}")
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs.extend(loader.load())

        # data 폴더에 파일이 없는 경우 기본 예시 가이드라인 사용
        if not docs:
            print("[!] data 폴더에 지침 문서가 없어 기본 입지 기준 샘플을 사용합니다.")
            docs = [
                Document(
                    page_content="스마트 주차 정보 수집 시스템 입지 평가 기준:\n"
                                 "1. 일 평균 유동교통량 1만 대 이상 (배점 40점)\n"
                                 "2. 인근 공영주차장 300m 이내 연계 가능성 높음 (배점 30점)\n"
                                 "3. 상업지역 및 혼잡지역 인접성 (배점 30점)",
                    metadata={"category": "주차정보시스템", "type": "입지기준"}
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
당신은 스마트도시 시설물 입지 선정 평가위원입니다.
제공된 [설치 가이드라인 규정]과 [후보지 현황 데이터]를 객관적으로 비교 분석하여 입지 적합성을 평가하고 정량적 점수 및 종합 의견을 제시하세요.

[설치 가이드라인]:
{context}

[후보지 데이터]:
{candidate_info}

[입지 평가 보고서]:"""
        prompt = ChatPromptTemplate.from_template(template)
        return (
            {"context": self.retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "candidate_info": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

if __name__ == "__main__":
    agent = SmartFacilityAgent()
    agent.load_documents_and_build_db()
    chain = agent.get_chain()

    # 테스트 1: 스마트 주차 시스템 입지 정량 평가 테스트
    candidate_data_1 = "후보지 A: 일 평균 교통량 15,000대, 공영주차장 200m 이내 위치, 상업지역 중심가 위치."
    print("==================================================")
    print(f"[후보지 1 데이터]: {candidate_data_1}\n")
    print("[입지 평가 보고서]:")
    print(chain.invoke(candidate_data_1))
    print("==================================================\n")

    # 테스트 2: VDS/ITS 규격 검토 테스트
    candidate_data_2 = "교차로 단락부 구간 VDS 설치 검토: 진출입 램프 합류 지점으로부터 200m 이격, 수직 여유고 5.5m 확보 예정."
    print(f"[후보지 2 데이터]: {candidate_data_2}\n")
    print("[입지/규격 검토 보고서]:")
    print(chain.invoke(candidate_data_2))