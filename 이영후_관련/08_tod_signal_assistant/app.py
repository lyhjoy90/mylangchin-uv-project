"""
===================================================================
[프로젝트 8] 교통신호 민원 분석 & TOD 조정 권고 웹 (08_tod_signal_assistant/app.py)
===================================================================
- 역할: 민원 입력 ➔ 원인 분석 ➔ TOD 신호 조정안 ➔ 대민 답변글 생성 UI (Port 7868)
- 실행 명령어: python 이영후_관련/08_tod_signal_assistant/app.py
===================================================================
"""

import os
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from main import engine

engine.build_db()

def create_ui():
    with gr.Blocks(title="지자체 교통신호 민원 분석 & TOD 조정 권고 AI") as demo:
        gr.Markdown("# 🚦 교통신호 민원 분석 & TOD 신호 조정 권고 AI")
        gr.Markdown("시민의 교차로 정체 민원을 분석하여 신호운영 편람 기반의 TOD(Time of Day) 조정안 및 대민 답변글을 자동 생성합니다.")

        with gr.Row():
            with gr.Column(scale=1):
                complaint_input = gr.Textbox(
                    label="교통신호 시민 민원 내용 입력",
                    placeholder="예: 퇴근시간대 아이데코 사거리 직진 신호가 짧아서 꼬리물기가 발생하고 보행자 신호도 너무 짧습니다.",
                    lines=6
                )
                analyze_btn = gr.Button("⚙️ 민원 분석 & TOD 조정안 생성", variant="primary")

            with gr.Column(scale=2):
                result_output = gr.Markdown("좌측에 민원 내용을 입력한 후 분석 버튼을 눌러주세요.")

        analyze_btn.click(engine.analyze_and_recommend, inputs=complaint_input, outputs=result_output)

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7868)