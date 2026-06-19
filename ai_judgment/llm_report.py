# coding: utf-8
"""
LLM 研判报告生成 (Module 2)
===========================

把量化指标（internals.JudgmentMetrics.to_dict()）交给大模型，生成结构化中文
研判报告。提示词来自 prompts.py（单一数据源）。

设计原则：LLM 只负责自然语言解释，不重算、不推翻深度模型的数值结论；
一切"事实依据"来自 internals.compute_judgment_metrics。

支持的 provider（config.LLM_PROVIDER）：
  - "anthropic" : Claude（默认推荐，claude-sonnet-4-6）
  - "openai"    : GPT 系列
  - "mock"      : 无密钥离线实现，规则文本拼装，保证字段契约一致（联调用）

任何真实调用失败都会降级为 mock，并在报告里标注 _provider / _error，保证
/api/analyze 永远返回结构一致的 report，不让前端因 LLM 抖动而崩。
"""
from __future__ import annotations

import json
import re
from typing import Dict

from . import config
from .prompts import SYSTEM_PROMPT, build_user_prompt, EXPECTED_FIELDS


# ---------------------------------------------------------------------------
# JSON 解析兜底
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> Dict:
    """从 LLM 文本里稳健地抽取 JSON 对象（容忍 ```json 包裹或前后多余文本）。"""
    text = text.strip()
    # 去掉 markdown 代码块围栏
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 退而求其次：截取第一个 { 到最后一个 }
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def _normalize(report: Dict, provider: str) -> Dict:
    """补全缺失字段，保证契约一致。"""
    for f in EXPECTED_FIELDS:
        report.setdefault(f, "" if f != "Evidence" else [])
    if isinstance(report.get("Evidence"), str):
        report["Evidence"] = [report["Evidence"]]
    report["_provider"] = provider
    return report


# ---------------------------------------------------------------------------
# mock 实现（无密钥可用）
# ---------------------------------------------------------------------------
def _mock_report(metrics: Dict) -> Dict:
    tier_map = {"SAFE": "安全", "SUSPICIOUS": "存疑", "HIGH_RISK": "高危"}
    risk = tier_map.get(metrics["preliminary_risk_level"], "存疑")
    evidence = list(metrics.get("triggered_rules", [])) or ["未触发显著异常规则。"]
    is_fake = metrics["predicted_label"] == "FAKE"
    conclusion = (
        f"模型判别为「{'虚假' if is_fake else '真实'}」，假新闻风险 "
        f"{metrics['fake_probability']:.0%}，量化置信度 {metrics['confidence']:.0%}。"
    )
    prop = (
        f"参与传播节点数 {metrics['num_neighbors']}，结构冲突指数 "
        f"{metrics['structural_conflict_index']:.2f}。"
        + ("检测到内容与传播图结构显著不一致，存在水军操纵 / 异常传播图谱嫌疑。"
           if metrics["structural_conflict_alert"] else "传播结构与内容基本一致。")
        + ("当前处于早期 / 稀疏传播阶段，结论置信度已下调，建议持续监测。"
           if metrics["sparse_propagation"] else "")
    )
    rec = ("高度警惕，请勿转发，建议交由人工复核。" if risk == "高危"
           else "建议人工复核后再做判断。" if risk == "存疑"
           else "暂未发现明显异常，可正常参考。")
    return _normalize({
        "Risk_Level": risk,
        "Conclusion": conclusion,
        "Evidence": evidence,
        "Propagation_Analysis": prop,
        "Recommendation": rec,
    }, provider="mock")


# ---------------------------------------------------------------------------
# Anthropic / OpenAI 实现
# ---------------------------------------------------------------------------
def _anthropic_report(metrics: Dict) -> Dict:
    import anthropic
    client = anthropic.Anthropic(api_key=config.LLM_API_KEY, timeout=config.LLM_TIMEOUT_S)
    resp = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(metrics)}],
    )
    text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
    return _normalize(_extract_json(text), provider="anthropic")


def _openai_report(metrics: Dict) -> Dict:
    from openai import OpenAI
    client = OpenAI(api_key=config.LLM_API_KEY, timeout=config.LLM_TIMEOUT_S)
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(metrics)},
        ],
    )
    text = resp.choices[0].message.content
    return _normalize(_extract_json(text), provider="openai")


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------
def generate_llm_report(metrics: Dict) -> Dict:
    """
    输入量化指标 dict，输出结构化研判报告 dict
    （字段：Risk_Level / Conclusion / Evidence / Propagation_Analysis / Recommendation）。
    真实调用失败时自动降级为 mock，保证返回结构稳定。
    """
    provider = config.LLM_PROVIDER.lower()
    if provider == "mock" or not config.LLM_API_KEY:
        return _mock_report(metrics)
    try:
        if provider == "anthropic":
            return _anthropic_report(metrics)
        if provider == "openai":
            return _openai_report(metrics)
        return _mock_report(metrics)
    except Exception as e:  # 网络 / 解析 / 密钥异常一律降级，不影响主链路
        fallback = _mock_report(metrics)
        fallback["_provider"] = f"mock(fallback from {provider})"
        fallback["_error"] = str(e)[:200]
        return fallback
