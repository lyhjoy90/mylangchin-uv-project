""" ##python 이영후_관련/01_urban_rag/main.py
===================================================================
[프로젝트 1] 도시정비 민원 및 규정 QA 시스템 (01_urban_rag/main.py)
===================================================================
- 목표: 정비사업 가이드라인 및 관련 법령/편람을 벡터화하여 정밀한 규정 조회 및 답변 자동화
- 특징: 부칙/시행일자 및 법령 조항을 반드시 검증하도록 프롬프트를 보완한 RAG 시스템
- 주요 기능:
  1. PDF/텍스트 문서 자동 로드 및 FAISS Vector DB 구축
  2. 법령 시행일자 및 부칙 검증 강화 프롬프트 적용
===================================================================
"""

import os
from dotenv import load_dotenv

# .env 파일에서 UPSTAGE_API_KEY 불러오기
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class UrbanRAGSystem:
    def __init__(self):
        self.embeddings = UpstageEmbeddings(model="embedding-query")
        self.llm = ChatUpstage(model="solar-pro")
        self.vector_store = None
        self.retriever = None

    def load_documents_and_build_db(self, data_dir: str = "data"):
        # main.py 파일 위치 기준으로 data 폴더의 절대 경로를 자동 계산합니다.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_data_dir = os.path.join(base_dir, data_dir)

        docs = []
        print(f"[*] 데이터 폴더 탐색 중: {target_data_dir}")

        if os.path.exists(target_data_dir):
            for file in os.listdir(target_data_dir):
                if file.endswith(".pdf"):
                    pdf_path = os.path.join(target_data_dir, file)
                    print(f"[*] PDF 문서 읽는 중: {file}")
                    loader = PyPDFLoader(pdf_path)
                    docs.extend(loader.load())

        if not docs:
            print("[!] PDF 문서를 찾지 못하여 기본 예시 문서를 사용합니다.")
            from langchain_core.documents import Document
            docs = [
                Document(page_content="[도시정비조례 제12조] 정비구역 지정 기본요건: 노후·불량건축물의 수가 대상구역 안의 건축물 총수의 3분의 2 이상이어야 한다. (시행일: 2025.01.01)"),
                Document(page_content="[도시정비조례 부칙] 이 조례는 2025년 1월 1일부터 시행한다.")
            ]

        # 문서 분할 및 벡터 DB 구축
        print(f"[*] 총 {len(docs)}개 페이지 로드 완료. 텍스트 분할 및 임베딩 진행 중...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = splitter.split_documents(docs)

        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        print("[+] FAISS Vector DB 구축 완료!\n")

    def get_chain(self):
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
        return (
            {"context": self.retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

if __name__ == "__main__":
    system = UrbanRAGSystem()
    system.load_documents_and_build_db()
    chain = system.get_chain()

    # 질문 실행
    query = "도시 및 주거환경정비법에서 정비구역 지정 요건과 시행일자 또는 관련 조항에 대해 설명해줘."
    print(f"[질문]: {query}\n")
    print("[답변]:")
    print(chain.invoke(query))