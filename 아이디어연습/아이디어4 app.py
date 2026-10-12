import streamlit as st
import requests

st.set_page_config(
    page_title="C-ITS 스마트 교통 관제 대시보드",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 C-ITS 스마트 교통 제어 및 AI 관제 대시보드")
st.markdown("VDS 센서, CCTV 관제 로그, 시민 민원을 융합하여 실시간 LOS를 산출하고 VMS 안내 문구 및 대응 가이드를 자동 생성합니다.")

# 사이드바 입력 패널
st.sidebar.header("📡 실시간 센서 및 현장 입력")
section_id = st.sidebar.text_input("도로 구간 ID", value="경부선_서울방향_104K")
speed = st.sidebar.slider("실시간 속도 (km/h)", min_value=0.0, max_value=120.0, value=18.5)
volume = st.sidebar.number_input("교통량 (대/h)", value=2450)
occupancy = st.sidebar.slider("점유율 (%)", min_value=0.0, max_value=100.0, value=75.2)

cctv_log = st.sidebar.text_area("CCTV 관제 로그", value="2차로 상에 승용차 단독 추돌 사고 발생 후 후속 차량들이 급감속하며 정체 꼬리가 빠르게 길어지고 있음.")
complaint_text = st.sidebar.text_area("시민 교통 민원", value="사고 지점 후방 1km 구간에서 차량들이 엉켜서 옴짝달싹 못 하고 있습니다.")

if st.sidebar.button("🚀 AI 관제 분석 실행", type="primary"):
    # API 페이로드 구성
    payload = {
        "section_id": section_id,
        "speed": speed,
        "volume": volume,
        "occupancy": occupancy,
        "cctv_log": cctv_log,
        "complaint_text": complaint_text
    }

    with st.spinner("AI 관제 에이전트가 데이터를 분석하고 있습니다..."):
        try:
            # FastAPI 서버로 요청 (로컬 테스트 기준)
            response = requests.post("http://127.0.0.1:8000/api/c-its/analyze", json=payload)
            if response.status_code == 200:
                data = response.json()
                
                # 메인 화면 레이아웃 구성
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 VDS 센서 전처리 및 LOS 판정")
                    vds = data["processed_vds"]
                    st.metric(label="구간 명칭", value=vds["section_id"])
                    st.metric(label="실시간 속도", value=f"{vds['speed']} km/h")
                    st.metric(label="도로 서비스 수준 (LOS)", value=vds["los_grade"], delta="정체/위험" if "F" in vds["los_grade"] else "정상", delta_color="inverse")
                    st.json(vds)

                    if data.get("complaint_result"):
                        st.subheader("📝 시민 교통 민원 자동 분류")
                        comp = data["complaint_result"]
                        st.info(f"**분류:** {comp['category']} | **긴급도:** {comp['urgency_level']}점")
                        for item in comp["action_plan"]:
                            st.write(f"- {item}")

                with col2:
                    st.subheader("🚨 C-ITS AI 돌발 대응 리포트")
                    report = data["incident_report"]
                    st.warning(f"**돌발 유형:** {report['incident_type']} (심각도: {report['severity']})")
                    st.write(f"⏱️ **예상 정상화 소요 시간:** 약 {report['estimated_recovery_min']}분")
                    
                    # VMS 전광판 시각화 UI
                    st.markdown("### 📺 VMS (도로전광표지) 표출 시뮬레이션")
                    vms_container = st.container()
                    with vms_container:
                        st.markdown(
                            f"""
                            <div style="background-color: #111; color: #ff5555; padding: 15px; border-radius: 8px; font-family: monospace; font-size: 20px; text-align: center; border: 2px solid #444;">
                                💡 [VMS 전광판]<br>
                                {'<br>'.join(report['vms_message'])}
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

                    st.subheader("💡 관제 센터 조치 가이드")
                    for guide in report["response_guide"]:
                        st.success(f"✔️ {guide}")

            else:
                st.error(f"서버 에러 발생: {response.text}")
        except Exception as e:
            st.error(f"백엔드 서버와 통신할 수 없습니다. FastAPI 서버가 실행 중인지 확인하세요. (에러: {e})")