"""
===================================================================
[프로젝트 6] 조합 정관 및 감정평가 분쟁 검토 AI (06_urban_redev_advisor/main.py)
===================================================================
- 역할: 재개발·재건축 조합 정관 PDF/TXT 문서를 분석하여 정족수 및 불공정 조항 검토
- 실행 명령어: python 이영후_관련/06_urban_redev_advisor/main.py
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
from langchain_core.runnables import RunnablePassthrough

class UrbanRedevAdvisorEngine:
    def __init__(self):
        self.embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
        self.llm = ChatUpstage(model="solar-pro3", temperature=0.0)
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
                    print(f"[*] PDF 로드 중: {file}")
                    loader = PyPDFLoader(file_path)
                    docs.extend(loader.load())
                elif file.endswith(".txt") or file.endswith(".md"):
                    print(f"[*] 텍스트/마크다운 로드 중: {file}")
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs.extend(loader.load())

        if not docs:
            print("[!] 기본 정관 문서를 찾지 못해 표준 조합 정관 예시 데이터를 생성합니다.")
            from langchain_core.documents import Document
            docs = [
                Document(page_content="[조합정관 제22조(총회의 의결방법)] 총회의 의결은 조합원 과반수 출석과 출석 조합원 과반수의 찬성으로 의결한다. 단, 사업시행계획서 작성 및 변경, 조합 정관의 변경은 조합원 3분의 2 이상의 찬성으로 의결한다."),
                Document(page_content="[조합정관 제35조(감정평가 및 분담금)] 조합원의 종전자산 감정평가는 지자체장이 추천한 2인 이상의 감정평가법인이 평가한 금액을 산술평균하여 산정한다. 분담금 납부 기한을 위반할 경우 연 12%의 지연이자율을 적용한다.")
            ]

        print(f"[*] 총 {len(docs)}개 문서 로드 완료. FAISS DB 구축 중...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
        splits = splitter.split_documents(docs)

        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        print("[+] Vector DB 구축 완료!\n")

    def get_chain(self):
        template = """[시스템 역할]
당신은 도시정비사업 조합 정관 및 감정평가 분쟁 전문 법률 어시스턴트입니다.
제공된 [조합 정관 규정]만을 기반으로 의결 정족수, 감정평가 절차, 위반 시 자격 요건을 분석하여 답변하세요.

[답변 작성 수칙]:
1. 해당 조항 번호(예: 제22조)와 명시된 의결 비율(예: 과반수, 2/3 이상)을 명확하게 인용하세요.
2. 참고 규정에 명시되지 않은 분쟁 사항은 추측하지 마세요.

[조합 정관 규정]:
{context}

[질문]:
{question}

[조합 정관 검토 의견]:"""
        prompt = ChatPromptTemplate.from_template(template)
        format_docs = lambda docs: "\n\n".join(d.page_content for d in docs)
        
        return (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

if __name__ == "__main__":
    advisor = UrbanRedevAdvisorEngine()
    advisor.load_documents_and_build_db()
    chain = advisor.get_chain()

    query = "조합 정관 변경이나 사업시행계획서 작성 시 필요한 총회 의결 정족수 조건은 어떻게 되나요?"
    print(f"[질문]: {query}\n")
    print("[답변]:")
    print(chain.invoke(query))