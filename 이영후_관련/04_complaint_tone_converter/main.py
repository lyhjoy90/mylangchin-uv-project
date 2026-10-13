"""
===================================================================
[프로젝트 4] 도시교통 및 민원 대응 업무 말투 변환기 (04_complaint_tone_converter/main.py)
===================================================================
- 목표: 거친 시민 민원, 일상적인 현장 메모, 비격식적 보고 내용을 
        공공기관/지자체 제출용 정형화된 공문서 및 보고서체로 자동 변환
- 특징: LangChain PromptTemplate + ChatUpstage(solar-pro) 조합으로 
        격식성, 명확성, 행정적 어조 유지
===================================================================
"""

import os
from dotenv import load_dotenv

# .env 파일에서 UPSTAGE_API_KEY 로드
load_dotenv()

from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ComplaintToneConverter:
    def __init__(self):
        self.llm = ChatUpstage(model="solar-pro")

    def get_chain(self):
        template = """[시스템 역할]
당신은 도시교통 및 행정 공문서 전문 작성 어시스턴트입니다.
입력된 거친 민원글, 현장 메모, 또는 일상적인 문장을 지자체 및 공공기관에 제출 가능한 **정형화된 행정 보고서체(개조식 문장)**로 다듬어 변환하세요.

[변환 원칙]:
1. 감정적인 표현이나 주관적 문장을 배제하고, 객관적 사실 중심의 감결한 문장(~함, ~검토 필요, ~ 조치 예정)으로 작성하세요.
2. 현장 상황(위치, 문제점, 요구사항)을 항목별로 명확히 분류하세요.
3. 전문적인 교통 및 행정 용어(예: 상충, 병목, 정체, 검토, 조치 등)를 사용하세요.

[원본 입력 문장]:
{raw_text}

[행정 보고서체 변환 결과]:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        return prompt | self.llm | StrOutputParser()

if __name__ == "__main__":
    converter = ComplaintToneConverter()
    chain = converter.get_chain()

    # 테스트 민원 메모 입력
    sample_input = """
    거기 신호등이 너무 짧아서 차들이 맨날 밀리고 난리예요. 
    좌회전 받아야 되는데 3번이나 신호 기다려야 겨우 지나갑니다. 
    사고 날 뻔한 적도 한 두번이 아닌데 빨리 신호 시간 늘려주던가 
    차로를 새로 하나 더 만들어주세요!
    """

    print("==================================================")
    print("[원본 입력 문장]:")
    print(sample_input.strip())
    print("==================================================\n")
    print("[행정 보고서체 변환 결과]:")
    print(chain.invoke({"raw_text": sample_input}))