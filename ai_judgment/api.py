# coding: utf-8
"""
FastAPI 服务 (Module 1: POST /api/analyze)
==========================================

启动时单例加载模型；POST /api/analyze 接收 news_id，跑推理 -> 量化指标 ->
LLM 报告，统一返回 JSON（量化指标 + 文本报告两部分，契约见 API_CONTRACT.md）。

本地启动：
    uvicorn ai_judgment.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import config
from .inference import get_engine
from .llm_report import generate_llm_report

app = FastAPI(title="TGRFN AI 智能研判分析系统", version="1.0.0")

# 允许前端跨域联调（生产应收敛 allow_origins）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------- 数据契约 -----------------------------
class AnalyzeRequest(BaseModel):
    news_id: str = Field(..., description="待研判新闻的 ID")
    with_report: bool = Field(True, description="是否调用 LLM 生成自然语言研判报告")


class AnalyzeResponse(BaseModel):
    code: int = 0
    message: str = "ok"
    news_id: str
    elapsed_ms: int
    metrics: dict          # 量化指标（internals.JudgmentMetrics）
    report: dict | None    # LLM 研判报告（llm_report）


# ----------------------------- 生命周期 -----------------------------
@app.on_event("startup")
def _startup():
    engine = get_engine()
    try:
        engine.load()
    except Exception as e:  # 加载失败不应让进程崩溃，交由 /health 暴露状态
        print(f"[startup][WARN] 模型/数据加载异常: {e}")


# ----------------------------- 路由 -----------------------------
@app.get("/health")
def health():
    engine = get_engine()
    return {
        "status": "ok" if engine.ready else "degraded",
        "model_loaded": engine.model is not None,
        "data_loaded": engine.bundle is not None,
        "device": engine.device,
        "ratio": engine.ratio,
        "llm_provider": config.LLM_PROVIDER,
    }


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    engine = get_engine()
    if not engine.ready:
        raise HTTPException(status_code=503, detail="DATA_NOT_LOADED: 模型或数据依赖未挂载")

    t0 = time.time()
    try:
        metrics = engine.analyze(req.news_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"INFERENCE_ERROR: {e}")

    metrics_dict = metrics.to_dict()
    report = generate_llm_report(metrics_dict) if req.with_report else None

    return AnalyzeResponse(
        news_id=req.news_id,
        elapsed_ms=int((time.time() - t0) * 1000),
        metrics=metrics_dict,
        report=report,
    )
