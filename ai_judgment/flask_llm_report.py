# coding: utf-8
"""
Flask 蓝图 —— LLM 研判报告接口（可直接挂到现有 Flask 后端）
============================================================

实现与前端/队友约定的接口：

    POST /api/v1/cases/<case_id>/llm-report          # 优先读缓存，无则真实生成
    POST /api/v1/cases/<case_id>/llm-report?regenerate=1   # 强制重新生成

设计要点（对应队友提出的需求）：
- 浏览器只传 case_id；后端按 case_id 读已有案例缓存，整理为 LLM 输入（见 report_service.build_llm_input）。
- API Key 只在后端（环境变量），绝不下发前端。
- 报告缓存：首次真实调用 LLM，成功后按 case_id 落盘；再次打开优先读缓存；保留"重新生成"。
- 调用超时可配；超时/失败时**仍返回相同结构的 JSON**（provider=mock(fallback...)），不让前端崩。

—— 如何接入你现有 Flask 应用（两步）——
    from ai_judgment.flask_llm_report import make_llm_report_blueprint

    # load_case_cache: 你已有的"按 case_id 读案例缓存"的函数，返回 dict 或 None
    bp = make_llm_report_blueprint(load_case_cache=your_load_case_cache)
    app.register_blueprint(bp)

如果你的案例缓存就是一个个 JSON 文件，也可以不传 load_case_cache，
改用环境变量 TGRFN_CASE_CACHE_DIR 指向缓存目录（文件名 <case_id>.json）。
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

from flask import Blueprint, jsonify, request

from .report_service import generate_report

# 报告缓存目录（可用环境变量覆盖）
REPORT_CACHE_DIR = os.getenv(
    "TGRFN_REPORT_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), "_report_cache"),
)
# 案例缓存目录（仅当未传 load_case_cache 时，用作默认的按文件读取）
CASE_CACHE_DIR = os.getenv("TGRFN_CASE_CACHE_DIR", "")


def _default_load_case_cache(case_id: str) -> Optional[dict]:
    """缺省实现：从 TGRFN_CASE_CACHE_DIR/<case_id>.json 读案例缓存。"""
    if not CASE_CACHE_DIR:
        return None
    path = os.path.join(CASE_CACHE_DIR, f"{case_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _report_cache_path(case_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in case_id)
    return os.path.join(REPORT_CACHE_DIR, f"{safe}.json")


def _read_report_cache(case_id: str) -> Optional[dict]:
    path = _report_cache_path(case_id)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _write_report_cache(case_id: str, report: dict) -> None:
    os.makedirs(REPORT_CACHE_DIR, exist_ok=True)
    with open(_report_cache_path(case_id), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def make_llm_report_blueprint(
    load_case_cache: Optional[Callable[[str], Optional[dict]]] = None,
    *,
    url_prefix: str = "",
) -> Blueprint:
    """构造 LLM 研判报告蓝图。

    load_case_cache: 传入你已有的"按 case_id 读案例缓存"函数；不传则用文件目录缺省实现。
    """
    loader = load_case_cache or _default_load_case_cache
    bp = Blueprint("llm_report", __name__, url_prefix=url_prefix)

    @bp.route("/api/v1/cases/<case_id>/llm-report", methods=["POST"])
    def llm_report(case_id: str):
        body = request.get_json(silent=True) or {}
        regenerate = (
            request.args.get("regenerate") in ("1", "true", "yes")
            or bool(body.get("regenerate"))
        )

        # 1) 命中缓存直接返回（除非强制重生成）
        if not regenerate:
            cached = _read_report_cache(case_id)
            if cached:
                cached["cached"] = True
                return jsonify({"code": 0, "message": "ok", "case_id": case_id, "report": cached})

        # 2) 读案例缓存
        case_cache = loader(case_id)
        if not case_cache:
            return jsonify({
                "code": 404, "message": "CASE_NOT_FOUND",
                "case_id": case_id, "report": None,
            }), 404

        # 3) 生成（内部已含超时与失败降级，永远返回结构化报告）
        report = generate_report(case_cache)
        report["cached"] = False

        # 4) 仅在真实 LLM 成功时落盘缓存（降级版不缓存，便于稍后重试拿到真报告）
        if str(report.get("provider", "")).startswith(("anthropic", "openai")):
            try:
                _write_report_cache(case_id, report)
            except Exception:
                pass

        return jsonify({"code": 0, "message": "ok", "case_id": case_id, "report": report})

    return bp
