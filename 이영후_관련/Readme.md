통합 디렉토리 구조
mylangchin-uv-project/                    <-- 최상위 프로젝트 루트
├── .venv/                                <-- 공유 가상환경
├── .env                                  # API 키 통합 관리 (UPSTAGE_API_KEY, OPENAI_API_KEY 등)
└── 이영후_관련/                          <-- 도시정비 & 교통 AI 프로젝트 메인 작업 폴더
    │
    ├── 01_urban_rag/                     <-- [기존 완료] 1. 도시정비 민원 및 규정 QA RAG
    │   ├── data/                         # 도시정비 관련 법령/지침 PDF 문서
    │   ├── main.py                       # CLI 백엔드 검증 스크립트
    │   ├── app.py                        # Gradio 기반 웹 UI 스크립트 (Port 7861)
    │   └── README.md
    │
    ├── 02_ic_design_assistant/           <-- [기존 완료] 2. IC/교차로 설계 지침 모니터링
    │   ├── data/                         # 도로용량편람, 설계지침 PDF 문서
    │   ├── main.py                       # CLI 백엔드 검증 스크립트
    │   ├── app.py                        # Gradio 기반 웹 UI 스크립트 (Port 7862)
    │   └── README.md
    │
    ├── 03_traffic_solution_db/           <-- [기존 완료] 3. 교통 문제 해결 대책 DB 구축 및 추천
    │   ├── data/                         # 교통 개선 우수사례 PDF/TXT 문서
    │   ├── main.py                       # CLI 백엔드 검증 스크립트
    │   ├── app.py                        # Gradio 기반 웹 UI 스크립트 (Port 7863)
    │   └── README.md
    │
    ├── 04_complaint_tone_converter/      <-- [기존 완료] 4. 교통 민원 자동 변환기 및 피드백 챗봇
    │   ├── main.py                       # CLI 백엔드 검증 스크립트
    │   ├── app.py                        # Gradio 기반 웹 UI 스크립트 (Port 7864)
    │   └── README.md
    │
    ├── 05_smart_facility_agent/          <-- [기존 완료] 5. 스마트 시설물 입지/규격 진단 대시보드
    │   ├── data/                         # 스마트 주차/VDS/ITS 설치 가이드라인 (.md, .pdf)
    │   ├── main.py                       # CLI 백엔드 검증 스크립트
    │   ├── app.py                        # Gradio 기반 웹 UI 스크립트 (Port 7865)
    │   └── README.md
    │
    │  ─── [신규 확장 단계: 수업 응용 아이디어 3선] ───────────────────────
    │
    ├── 06_urban_redev_advisor/           <-- [신규 06] 조합 정관 및 감정평가 분쟁 검토 AI
    │   ├── data/                         # 조합 정관, 사업시행계획서 샘플
    │   ├── main.py                       # 조항/정족수 파싱 백엔드
    │   ├── app.py                        # Gradio 대화형 검토 웹 UI (Port 7866)
    │   └── README.md
    │
    ├── 07_traffic_impact_reviewer/       <-- [신규 07] 도로점용 공사 교통소통대책서 검토 에이전트
    │   ├── data/                         # 도로점용 소통대책 가이드라인
    │   ├── main.py                       # 안전시설/우회도로 미비점 판별 백엔드
    │   ├── app.py                        # Gradio 검토의견서 생성 웹 UI (Port 7867)
    │   └── README.md
    │
    └── 08_tod_signal_assistant/          <-- [신규 08] 지자체 교통신호 민원 분석 & TOD 조정 권고 AI
        ├── data/                         # 신호운영 편람, TOD 주기 배분 기준
        ├── main.py                       # 민원 텍스트 분석 및 Offset/녹색시간 산정 백엔드
        ├── app.py                        # Gradio 신호조정 및 답변서 자동생성 UI (Port 7868)
        └── README.md