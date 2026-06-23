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
    """规则化生成一份**详细具体**的中文研判报告（无密钥可用）。

    逐项把量化指标翻译成可核查的自然语言证据，字段契约与真实 LLM 输出完全一致，
    保证离线 / 降级场景下前端也能拿到内容充实的报告。
    """
    tier_map = {"SAFE": "安全", "SUSPICIOUS": "存疑", "HIGH_RISK": "高危"}
    risk = tier_map.get(metrics["preliminary_risk_level"], "存疑")
    is_fake = metrics["predicted_label"] == "FAKE"
    label_cn = "虚假" if is_fake else "真实"

    fake_p = metrics["fake_probability"]
    conf = metrics["confidence"]
    margin = metrics.get("decision_margin", 0.0)
    sci = metrics["structural_conflict_index"]
    attn = metrics.get("modality_attention", {}) or {}
    dom = metrics.get("dominant_modality", "unknown")
    dom_cn = {"text": "文本", "image": "视觉", "time": "时间", "unknown": "未知"}.get(dom, dom)
    n_nb = metrics["num_neighbors"]

    # —— Conclusion：标签 + 风险 + 置信度 + 最关键一项异常 ——
    if metrics["structural_conflict_alert"]:
        key_point = "传播结构与内容表示出现显著冲突，疑似存在协同操纵痕迹。"
    elif metrics["modality_dominance_alert"]:
        key_point = f"{dom_cn}模态注意力异常集中（占比 {attn.get(dom, 0):.0%}），是本次判别的主要依据。"
    elif metrics["sparse_propagation"]:
        key_point = "目前传播规模偏小、处于早期阶段，结论需结合后续观测进一步确认。"
    else:
        key_point = "各项异常检测均未触发，内容与传播行为基本自洽。"
    conclusion = (
        f"模型判别本条新闻为「{label_cn}」，校准后假新闻风险 {fake_p:.0%}，"
        f"量化置信度 {conf:.0%}。{key_point}"
    )

    # —— Evidence：逐条挂靠具体数值 ——
    evidence = []
    evidence.append(
        f"判别依据：判别间隔 decision_margin={margin:.2f}"
        f"（{'<0 偏向虚假' if margin < 0 else '≥0 偏向真实'}），"
        f"经训练集正负样本比 ratio={metrics.get('ratio_baseline', 1.0)} 校准后，"
        f"假新闻风险为 {fake_p:.0%}。"
    )
    if metrics["structural_conflict_alert"]:
        evidence.append(
            f"结构冲突告警：内容↔传播图结构冲突指数 {sci:.2f}，高于告警阈值，"
            "表明新闻文本本身与其传播路径行为相互矛盾，是水军操纵 / 异常传播图谱的典型信号。"
        )
    else:
        evidence.append(f"结构一致：内容↔传播图结构冲突指数仅 {sci:.2f}，未触发告警，传播行为与内容基本吻合。")
    if attn:
        attn_txt = "、".join(
            f"{ {'text':'文本','image':'视觉','time':'时间'}.get(k, k) } {v:.0%}"
            for k, v in attn.items()
        )
        evidence.append(f"模态贡献：三模态注意力归一化占比为 {attn_txt}，由「{dom_cn}」模态主导判别。")
    if metrics["modality_dominance_alert"]:
        if dom == "image":
            evidence.append("视觉异常：图像模态注意力异常占优，提示配图可能经过篡改 / 张冠李戴，建议做反向图搜溯源。")
        elif dom == "time":
            evidence.append("时间异常：时间模态注意力异常占优，传播呈典型谣言式爆发后骤减的时间衰减形态。")
        elif dom == "text":
            evidence.append("文本驱动：文本模态注意力异常占优，判别主要由措辞 / 语义触发，建议核查文本事实与信源。")
    evidence.append(
        f"传播规模：参与传播聚合的邻居节点数 {n_nb} 个，"
        + (f"低于稀疏阈值，属早期 / 稀疏传播，置信度已下调。" if metrics["sparse_propagation"]
           else "传播规模充分，统计意义较稳定。")
    )

    # —— Propagation_Analysis：连续成段 ——
    seg = [f"本案例共聚合 {n_nb} 个传播邻居节点参与研判。"]
    if metrics["structural_conflict_alert"]:
        seg.append(
            f"结构冲突指数高达 {sci:.2f}，模型在'内容语义表示'与'传播图结构表示'之间检测到显著不对齐，"
            "存在水军操纵或异常传播图谱嫌疑——即内容看似平常，但其被转发 / 互动的结构呈现非自然协同特征。"
        )
    else:
        seg.append(f"结构冲突指数为 {sci:.2f}，处于正常区间，传播图结构与内容语义基本一致，未见明显协同操纵痕迹。")
    if metrics["modality_dominance_alert"] and dom == "image":
        seg.append("从模态看，视觉特征贡献异常突出，存在图像篡改 / 移花接木的高度嫌疑，应优先对配图做溯源核验。")
    elif metrics["modality_dominance_alert"] and dom == "time":
        seg.append("从模态看，时间特征贡献异常突出，传播曲线呈爆发后快速衰减的典型谣言形态。")
    elif dom == "text":
        seg.append("从模态看，判别主要由文本内容驱动，异常更多体现在措辞与语义层面，建议结合事实核查。")
    else:
        seg.append("各模态贡献相对均衡，未见单一模态异常主导。")
    if metrics["sparse_propagation"]:
        seg.append("需要强调：当前数据处于早期 / 稀疏传播阶段，样本量有限，模型已主动下调置信度，建议纳入持续监测、待传播充分后复判。")
    else:
        seg.append("当前传播已较为充分，结论的统计稳定性较好。")
    prop = "".join(seg)

    # —— Recommendation：分层处置 ——
    if risk == "高危":
        rec = ("总体口径：高度警惕，请勿转发扩散。建议立即交由内容安全团队人工复核，"
               "并追踪异常账号集群、对涉事图片 / 信源做二次溯源核验。")
    elif risk == "存疑":
        rec = ("总体口径：暂不建议扩散，建议人工复核后再做判断。可优先核查文本事实与原始信源，"
               "并关注后续传播是否出现协同放大迹象。")
    else:
        rec = ("总体口径：暂未发现明显异常，可正常参考。建议保留常规抽样复核，"
               "对突发的传播放大保持关注即可。")

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
