from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = FastAPI(
    title="C-ITS 스마트 교통 관제 AI API",
    version="1.0.0",
    description="VDS/CCTV/민원 데이터를 연계한 실시간 혼잡도 분석 및 돌발 대응 AI 시스템"
)

# LLM 및 파서 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Pydantic 모델 정의
class TrafficAnalysisRequest(BaseModel):
    section_id: str = Field(description="도로 구간 ID", example="경부선_서울방향_104K")
    speed: float = Field(description="실시간 속도 (km/h)", example=18.5)
    volume: int = Field(description="교통량 (대/h)", example=2450)
    occupancy: float = Field(description="점유율 (%)", example=75.2)
    cctv_log: str = Field(description="CCTV 현장 관제 로그", example="2차로 상 승용차 추돌 사고 발생")
    complaint_text: Optional[str] = Field(description="시민 교통 민원 내용", example="사고 지점 후방 극심한 정체 발생")

class ComplaintAnalysis(BaseModel):
    category: str = Field(description="민원 분류")
    urgency_level: int = Field(description="긴급도 (1~5)")
    action_plan: List[str] = Field(description="현장 조치 사항")

class IncidentResponseReport(BaseModel):
    incident_type: str = Field(description="돌발 상황 유형")
    severity: str = Field(description="심각도 수준")
    vms_message: List[str] = Field(description="VMS 3행 이내 문구 리스트")
    response_guide: List[str] = Field(description="관제 센터 대응 가이드")
    estimated_recovery_min: int = Field(description="예상 정상화 소요 시간 (분)")

class IntegratedTrafficResponse(BaseModel):
    processed_vds: dict
    complaint_result: Optional[ComplaintAnalysis] = None
    incident_report: IncidentResponseReport


@app.post("/api/c-its/analyze", response_model=IntegratedTrafficResponse, summary="실시간 교통 관제 및 AI 분석 수행")
def analyze_traffic_data(req: TrafficAnalysisRequest):
    try:
        # 1. VDS 전처리 및 LOS 판정
        speed = req.speed
        occupancy = req.occupancy
        status_flag = "SENSOR_ANOMALY_OR_GRIDLOCK" if (speed <= 0 or occupancy > 80) else "NORMAL_OPERATION"
        
        if speed >= 80: los = "LOS A"
        elif speed >= 70: los = "LOS B"
        elif speed >= 60: los = "LOS C"
        elif speed >= 50: los = "LOS D"
        elif speed >= 30: los = "LOS E (서행)"
        else: los = "LOS F (정체/위험)"

        processed_vds = {
            "section_id": req.section_id,
            "speed": speed,
            "volume": req.volume,
            "occupancy": occupancy,
            "status_flag": status_flag,
            "los_grade": los
        }

        # 2. 민원 분석 (입력된 경우에만)
        complaint_result = None
        if req.complaint_text:
            complaint_parser = PydanticOutputParser(pydantic_object=ComplaintAnalysis)
            complaint_prompt = PromptTemplate(
                template="다음 현장 교통 민원을 분석하여 분류하고 조치안을 도출하세요.\n민원내용: {text}\n{format_instructions}",
                input_variables=["text"],
                partial_variables={"format_instructions": complaint_parser.get_format_instructions()}
            )
            complaint_chain = complaint_prompt | llm | complaint_parser
            complaint_result = complaint_chain.invoke({"text": req.complaint_text})

        # 3. 돌발 대응 리포트 생성
        incident_parser = PydanticOutputParser(pydantic_object=IncidentResponseReport)
        incident_prompt = PromptTemplate(
            template="""당신은 C-ITS 스마트 교통 관제 AI 마스터 에이전트입니다.
아래의 실시간 VDS 센서 분석 결과와 CCTV 관제 로그를 바탕으로 종합 대응 리포트와 VMS 안내 문구를 생성하세요.

[VDS 센서 데이터 분석 결과]
- 구간 ID: {section_id}
- 실시간 속도: {speed} km/h
- 교통량: {volume} 대/h
- 서비스 수준(LOS): {los_grade}
- 센서 상태 플래그: {status_flag}

[CCTV 현장 관제 로그]
- {cctv_log}

{format_instructions}""",
            input_variables=["section_id", "speed", "volume", "los_grade", "status_flag", "cctv_log"],
            partial_variables={"format_instructions": incident_parser.get_format_instructions()}
        )
        incident_chain = incident_prompt | llm | incident_parser
        incident_report = incident_chain.invoke({**processed_vds, "cctv_log": req.cctv_log})

        return IntegratedTrafficResponse(
            processed_vds=processed_vds,
            complaint_result=complaint_result,
            incident_report=incident_report
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))