# coding: utf-8
"""
LLM 研判报告生成 —— Module 1 预留接口 / Module 2 正式实现
=========================================================

设计原则（关键）：
  大模型**只负责把量化指标翻译成自然语言研判报告**，不重新计算、不质疑模型
  的概率与结构冲突数值。一切"事实依据"来自 internals.compute_judgment_metrics。
  这样可避免 LLM 幻觉覆盖深度模型的真实计算结果。

Module 1 阶段：提供 provider="mock" 的离线实现，使整条 API 链路（含 /api/analyze
返回 report 字段）在无密钥时即可联调。Module 2 将填充 anthropic / openai 的
真实调用与 System Prompt（System Prompt 详见 PROMPTS.md）。
"""
from __future__ import annotations

import json
from typing import Dict

from . import config


def _build_user_payload(metrics: Dict) -> str:
    """把量化指标打包成给 LLM 的 user 消息（结构化、只读事实）。"""
    return json.dumps(metrics, ensure_ascii=False, indent=2)


def _mock_report(metrics: Dict) -> Dict:
    """无密钥时的占位报告：直接由规则文本拼装，保证字段契约一致。"""
    tier_map = {"SAFE": "安全", "SUSPICIOUS": "存疑", "HIGH_RISK": "高危"}
    risk = tier_map.get(metrics["preliminary_risk_level"], "存疑")
    evidence = metrics.get("triggered_rules", []) or ["未触发显著异常规则。"]
    conclusion = (
        f"模型判别为「{'真实' if metrics['predicted_label'] == 'REAL' else '虚假'}」，"
        f"假新闻风险 {metrics['fake_probability']:.0%}，量化置信度 {metrics['confidence']:.0%}。"
    )
    prop = (
        f"参与传播节点数 {metrics['num_neighbors']}，"
        f"结构冲突指数 {metrics['structural_conflict_index']:.2f}。"
        + ("检测到内容与传播图结构显著不一致，存在水军操纵/异常传播图谱嫌疑。"
           if metrics["structural_conflict_alert"] else "传播结构与内容基本一致。")
        + ("当前处于早期/稀疏传播阶段，结论置信度已下调，建议持续监测。"
           if metrics["sparse_propagation"] else "")
    )
    return {
        "Risk_Level": risk,
        "Conclusion": conclusion,
        "Evidence": evidence,
        "Propagation_Analysis": prop,
        "_provider": "mock",
    }


def generate_llm_report(metrics: Dict) -> Dict:
    """
    输入量化指标 dict，输出结构化 LLM 研判报告 dict
    （字段：Risk_Level / Conclusion / Evidence / Propagation_Analysis）。

    Module 1：mock 实现可用。Module 2：接 anthropic/openai。
    """
    provider = config.LLM_PROVIDER.lower()
    if provider == "mock" or not config.LLM_API_KEY:
        return _mock_report(metrics)

    # —— Module 2 正式实现占位（接入时取消注释并补全） ——
    # if provider == "anthropic":
    #     return _anthropic_report(metrics)
    # if provider == "openai":
    #     return _openai_report(metrics)
    return _mock_report(metrics)
