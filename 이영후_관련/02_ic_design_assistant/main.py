"""
===================================================================
[프로젝트 2] IC / 교차로 설계 지침 모니터링 어시스턴트
===================================================================
- 목표: 고속도로 IC 연결로, 변속차로(Taper) 및 신호교차로 산정공식 지침을 학습한 기술 검색 보조도구 구축
- 특징: 수식, 앙각, Taper 길이 등 치수 및 설계 기준표 자동 로드 및 파싱 지원
===================================================================
"""

import os
from dotenv import load_dotenv

# .env 파일에서 UPSTAGE_API_KEY 로드
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class ICDesignAssistant:
    def __init__(self):
        self.embeddings = UpstageEmbeddings(model="embedding-query")
        self.llm = ChatUpstage(model="solar-pro")
        self.vector_store = None
        self.retriever = None

    def load_documents_and_build_db(self, data_dir: str = "data"):
        # main.py 위치 기준 data 폴더 절대 경로 설정
        base_dir = os.path.dirname(os.path.abspath(__file__))
        target_data_dir = os.path.join(base_dir, data_dir)

        docs = []
        print(f"[*] 데이터 폴더 탐색 중: {target_data_dir}")

        if os.path.exists(target_data_dir):
            for file in os.listdir(target_data_dir):
                if file.endswith(".pdf"):
                    pdf_path = os.path.join(target_data_dir, file)
                    print(f"[*] PDF 지침 문서 읽는 중: {file}")
                    loader = PyPDFLoader(pdf_path)
                    docs.extend(loader.load())

        # PDF 파일이 없을 경우 예시 기준표 사용
        if not docs:
            print("[!] PDF 문서를 찾지 못하여 샘플 기준 데이터를 사용합니다.")
            docs = [
                Document(
                    page_content="| 설계속도(km/h) | 감속차로 Taper 비율 | 최소 길이(m) |\n| 100 | 1:20 | 60 |\n| 80 | 1:15 | 50 |",
                    metadata={"category": "Taper길이"}
                ),
                Document(
                    page_content="황색신호시간(Yellow Time) 산정식: Y = t + V / (2a + 2gG)\n(t: 인지반응시간 1초, V: 접근속도, a: 감속도, G: 경사도)",
                    metadata={"category": "신호타이밍"}
                )
            ]

        print(f"[*] 총 {len(docs)}개 페이지 로드 완료. 텍스트 분할 및 임베딩 진행 중...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        splits = splitter.split_documents(docs)

        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        print("[+] FAISS Vector DB 구축 완료!\n")

    def get_chain(self):
        template = """[시스템 역할]
당신은 도로 및 교통공학 설계 지침 기술 어시스턴트입니다.
제공된 [설계 기준표 및 수식]을 바탕으로 질문에 수치와 기준을 정확히 명시하여 답변하세요.

[설계 기준]:
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
    assistant = ICDesignAssistant()
    assistant.load_documents_and_build_db()
    chain = assistant.get_chain()

    # 테스트 질문 (넣으신 지침 문서 내용에 맞춰 질문을 변경해보세요)
    query = "테이퍼(Taper) 설치 기준이나 황색신호시간 산정 공식에 대해 설명해줘."
    print(f"[질문]: {query}\n")
    print("[답변]:")
    print(chain.invoke(query))