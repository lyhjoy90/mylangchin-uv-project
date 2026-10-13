"""
===================================================================
[프로젝트 7] 도로점용 공사 교통소통대책서 검토 웹 (07_traffic_impact_reviewer/app.py)
===================================================================
- 역할: 소통대책서 입력/업로드 시 개조식 검토의견서 자동 생성 UI (Port 7867)
- 실행 명령어: python 이영후_관련/07_traffic_impact_reviewer/app.py
===================================================================
"""

import os
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from main import engine

engine.build_db()

def create_ui():
    with gr.Blocks(title="도로점용 공사 교통소통대책서 검토 에이전트") as demo:
        gr.Markdown("# 🚧 도로점용 공사 교통소통대책서 AI 검토 에이전트")
        gr.Markdown("도로점용 공사 소통대책서 개요 및 차단 계획을 입력하면 국토부/지자체 가이드라인 대비 미비점을 검토하여 개조식 검토의견서를 작성합니다.")

        with gr.Row():
            with gr.Column(scale=1):
                plan_input = gr.Textbox(
                    label="공사 교통소통대책서 주요 내용 입력",
                    placeholder="예: 공사 구간: 내흥초 사거리, 2차로 차단, 안내표지판 30m 전방 설치, 야간공사 진행, 보행자 통로 1.2m 확보.",
                    lines=8
                )
                review_btn = gr.Button("📋 검토의견서 자동 생성", variant="primary")

            with gr.Column(scale=2):
                report_output = gr.Markdown("좌측에 소통대책서 계획을 입력 후 검토 버튼을 눌러주세요.")

        review_btn.click(engine.review_plan, inputs=plan_input, outputs=report_output)

    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="127.0.0.1", server_port=7867)