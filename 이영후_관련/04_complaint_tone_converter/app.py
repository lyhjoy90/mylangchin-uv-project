"""
===================================================================
[프로젝트 4] 교통 민원 자동 변환 및 피드백 챗봇 (04_complaint_tone_converter/app.py)
===================================================================
- 역할: 거친 민원 메모를 개조식 행정 보고서체로 변환 + 피드백 수집
- 실행 명령어: python 이영후_관련/04_complaint_tone_converter/app.py
===================================================================
"""
##특징: gr.ChatInterface 및 flagging_mode="manual"을 적용한 민원 변환 및 품질 피드백(Like/Dislike) 수집 챗봇
import os
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ComplaintConverterEngine:
    def __init__(self):
        self.llm = ChatUpstage(model="solar-pro3", temperature=0.1)
        self.template = """당신은 지자체 교통행정과 전문 보고서 작성관입니다.
시민의 거친 민원 메모나 전화 녹취록을 공식 행정 개조식 보고서로 즉시 변환하세요.

[작성 수칙]:
1. 감정적/감성적 표현을 완벽히 제거하고 사실 및 요청 내용 위주로 정리하세요.
2. [민원 요약], [주요 요구사항], [현장 검토 필요 항목] 3가지 구문으로 구분하여 개조식(- )으로 작성하세요.

[시민 민원 내용]:
{complaint}

[행정 보고서체 변환]:"""
        self.prompt = ChatPromptTemplate.from_template(self.template)
        self.chain = self.prompt | self.llm | StrOutputParser()

    def convert(self, message, history):
        if not message or not message.strip():
            return "민원 내용을 입력해주세요."
        return self.chain.invoke({"complaint": message})

engine = ComplaintConverterEngine()

def create_ui():
    demo = gr.ChatInterface(
        fn=engine.convert,
        title="💬 교통 민원 행정 보고서체(개조식) 자동 변환기",
        description="거친 민원 메모나 구두 접수 내용을 입력하면 공문서용 개조식 보고서체로 깔끔하게 변환합니다.",
        examples=[
            "출근길마다 신호등이 왜 이리 길어! 옥녀교차로 좌회전 신호 짧아서 차 다 막히고 난리도 아님 당장 고쳐요!",
            "내흥초 앞 횡단보도 바닥 신호등 고장 난 것 같음. 야간에 애들 건너기 위험하니까 점검 좀 해봐요."
        ],
        flagging_mode="manual",
        flagging_options=["Like", "Dislike", "표현 미흡"],
        flagging_dir="flagged_complaints"
    )
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7864)