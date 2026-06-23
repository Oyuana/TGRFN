# coding: utf-8
"""
LLM 研判报告服务 —— 与前端/Flask 约定的接口实现
=================================================

对接前端预留的 `requestLlmReport(case_id)`，约定接口：

    POST /api/v1/cases/{case_id}/llm-report

本模块**不依赖任何 Web 框架**，只负责"把一份案例材料(dict) → 结构化研判报告(dict)"，
便于直接嵌入现有 Flask 后端（见同目录 flask_llm_report.py 蓝图）。

返回字段与前端约定一一对应（结构化 JSON，不是一整段文本）：
    conclusion            综合研判结论
    content_analysis      新闻内容分析
    propagation_analysis  传播结构分析
    evidence_analysis     证据 / 来源分析
    risk_summary          风险总结
    recommendation        处置 / 核查建议
    risk_level            高危 / 存疑 / 安全（附加，便于前端着色）
    generated_at          生成时间(ISO8601)
    model                 使用的 LLM
    latency_ms            生成耗时(毫秒)
    provider              anthropic / openai / mock / mock(fallback...)

设计原则同主链路：**LLM 只解释、不重算、不推翻**预计算的预测结果；任何真实调用
失败（超时 / 限流 / 无密钥 / 解析失败）都降级为规则版报告(mock)，保证接口永远
返回结构一致的 JSON，比赛展示不白屏。
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from . import config

# 报告里固定的 6 个分析字段（用于校验 / 兜底补全）
ANALYSIS_FIELDS = [
    "conclusion",
    "content_analysis",
    "propagation_analysis",
    "evidence_analysis",
    "risk_summary",
    "recommendation",
]


# ---------------------------------------------------------------------------
# 1. 从"案例缓存"整理出 LLM 输入材料
#    —— 容忍两种结构：①前端 /analysis 的嵌套结构(case/prediction/statistics/graph)
#                      ②已摊平的扁平结构。后端怎么存都能读。
# ---------------------------------------------------------------------------
def build_llm_input(case_cache: Dict) -> Dict:
    """把后端按 case_id 读出的案例缓存，整理成 LLM 所需的精简材料。"""
    c = case_cache.get("case", case_cache)
    pred = case_cache.get("prediction", case_cache)
    stats = case_cache.get("statistics", case_cache)
    graph = case_cache.get("graph", {}) or {}
    tr = graph.get("time_range", {}) or {}

    def pick(d, *keys, default=None):
        for k in keys:
            if isinstance(d, dict) and d.get(k) not in (None, ""):
                return d[k]
        return default

    duration = pick(stats, "propagation_duration_seconds")
    if duration is None and tr:
        duration = (tr.get("max_seconds") or 0) - (tr.get("min_seconds") or 0)

    return {
        # —— 1) 内容 ——
        "title": pick(c, "title", default=""),
        "body": pick(c, "body", "content", "text", default=""),
        "summary": pick(c, "summary", "abstract", default=""),
        "publish_time": pick(c, "publish_time", "published_at"),
        "source_url": pick(c, "source_url", "url", default=""),
        # —— 2) 数据集 ——
        "dataset": pick(c, "dataset", default=""),
        "ground_truth": pick(c, "ground_truth"),
        # —— 3) 预计算预测结果（LLM 只解释，不重算）——
        "pred_label": pick(pred, "pred_label", "predicted_label"),
        "fake_probability": pick(pred, "fake_probability"),
        "real_probability": pick(pred, "real_probability"),
        "threshold": pick(pred, "threshold"),
        # —— 4) 传播结构统计 ——
        "num_nodes": pick(stats, "display_node_count", "num_nodes", default=len(graph.get("nodes", []) or [])),
        "num_edges": pick(stats, "num_edges", default=len(graph.get("edges", []) or [])),
        "explicit_edges": pick(stats, "explicit_edge_count", "explicit_edges"),
        "enhanced_edges": pick(stats, "enhanced_edge_count", "enhanced_edges"),
        "propagation_duration_seconds": duration,
        # —— 5) 可选：重要传播节点（不传整图）——
        "key_nodes": case_cache.get("key_nodes", []),
    }


# ---------------------------------------------------------------------------
# 2. 提示词（专用于本结构化报告；与 prompts.py 同源思想：LLM 只翻译不裁决）
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """你是「TGRFN 虚假新闻研判助手」，服务于一套面向社交平台内容安全的虚假传播风险研判系统。
你会收到一份关于某条新闻的【研判材料】（JSON），其中包含新闻内容、数据集、由底层 TGRFN 模型
**预先算好**的预测结果（标签 / 真假概率 / 阈值），以及传播结构统计（节点数 / 边数 / 显式边 /
增强边 / 传播时长 等）。

【铁律】
1. 预测标签、真假概率、阈值都是底层模型的事实输出，你**不得重新计算、不得推翻**，只能解释。
2. **不得编造**材料里没有的事实、来源、人物或数据；只能基于给定材料推理。
3. 必须**严格输出 JSON**，不要输出 JSON 以外的任何文字或 Markdown 代码块标记。

【分析要点】
- 内容分析：结合标题/正文/摘要，指出文本风格、是否情绪化/煽动性、信源是否权威可核。
- 传播分析：结合节点数、边数、显式边/增强边比例、传播时长判断传播规模与是否自然。
  「增强边」是模型对潜在协同关系的补全：增强边占比偏高、或短时间内大量节点聚集，
  提示可能存在水军操纵 / 协同造假 / 异常传播图谱。传播规模过小则说明处于早期/稀疏阶段，
  结论置信度应下调。
- 证据分析：逐条列出支撑判断的关键依据，并尽量引用具体数值（概率、阈值、边数、时长等）。
- 风险总结：综合给出风险等级与一句话定性。
- 处置建议：分层给出（可信可参考 / 建议人工复核 / 高度警惕勿传播）+ 具体动作。

【输出格式——严格 JSON，字段固定】
{
  "risk_level": "高危" | "存疑" | "安全",
  "conclusion": "2~3 句综合研判结论，含判别标签与风险高低",
  "content_analysis": "新闻内容分析段落",
  "propagation_analysis": "传播结构分析段落（结合节点/边/增强边/时长）",
  "evidence_analysis": "证据/来源分析；可用要点式，逐条引用具体数值",
  "risk_summary": "风险总结，一句话定性 + 风险等级",
  "recommendation": "处置/核查建议（口径 + 1~2 条具体动作）"
}

【风险等级参考】fake_probability ≥ 0.65 或传播结构高度异常 ⇒ 倾向「高危」；
0.35~0.65 ⇒「存疑」；< 0.35 且无异常 ⇒「安全」。表述统一为"模型预测/传播风险研判"，
不要写成已坐实的事实核查定论。"""


def build_user_prompt(material: Dict) -> str:
    payload = json.dumps(material, ensure_ascii=False, indent=2)
    return (
        "以下是本条新闻的【研判材料】，请严格依据它生成研判报告 JSON：\n\n"
        f"{payload}\n\n请只输出符合 System 要求的 JSON。"
    )


# ---------------------------------------------------------------------------
# 3. JSON 解析兜底 + 字段规范化
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> Dict:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.find("{"), text.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(text[s:e + 1])
        raise


def _normalize(report: Dict) -> Dict:
    report.setdefault("risk_level", "存疑")
    for f in ANALYSIS_FIELDS:
        report.setdefault(f, "")
    # evidence 允许 list，统一成字符串展示更稳；前端也可处理 list
    if isinstance(report.get("evidence_analysis"), list):
        report["evidence_analysis"] = "\n".join(f"· {x}" for x in report["evidence_analysis"])
    return report


# ---------------------------------------------------------------------------
# 4. 规则版报告（无密钥 / 降级时使用，仍然内容充实）
# ---------------------------------------------------------------------------
def _pct(x) -> str:
    try:
        return f"{float(x) * 100:.1f}%"
    except (TypeError, ValueError):
        return "未知"


def _mock_report(m: Dict) -> Dict:
    fake_p = m.get("fake_probability")
    is_fake = (m.get("pred_label") in ("fake", "FAKE")) or (isinstance(fake_p, (int, float)) and fake_p >= 0.5)
    label_cn = "虚假" if is_fake else "真实"

    try:
        fp = float(fake_p)
    except (TypeError, ValueError):
        fp = 0.5
    risk_level = "高危" if fp >= 0.65 else ("存疑" if fp >= 0.35 else "安全")

    nodes = m.get("num_nodes") or 0
    edges = m.get("num_edges") or 0
    enh = m.get("enhanced_edges")
    exp = m.get("explicit_edges")
    dur = m.get("propagation_duration_seconds")
    dur_txt = (f"{dur/60:.0f} 分钟" if isinstance(dur, (int, float)) and dur < 3600
               else f"{dur/3600:.1f} 小时" if isinstance(dur, (int, float)) else "未知")
    sparse = isinstance(nodes, (int, float)) and nodes < 10
    enh_ratio = None
    if isinstance(enh, (int, float)) and isinstance(edges, (int, float)) and edges:
        enh_ratio = enh / edges

    conclusion = (
        f"模型预测本条新闻为「{label_cn}」，假新闻风险 {_pct(fake_p)}"
        f"（阈值 {m.get('threshold', '未知')}）。"
        + ("传播结构呈现协同放大迹象，需高度警惕。" if (enh_ratio or 0) >= 0.3
           else "目前处于早期/稀疏传播阶段，结论需结合后续观测。" if sparse
           else "传播结构总体自然，可作常规参考。")
    )

    content_analysis = (
        f"标题：「{m.get('title') or '（无标题）'}」。数据集：{m.get('dataset') or '未知'}。"
        + (f"原始链接：{m.get('source_url')}。" if m.get("source_url") else "未提供原始链接。")
        + "建议结合正文措辞是否情绪化、信源是否权威进行人工核查；"
        + ("模型给出的较高假新闻风险主要应由内容真实性与信源可信度复核来佐证。" if is_fake
           else "模型倾向判真，内容层面未见显著造假信号。")
    )

    propagation_analysis = (
        f"传播规模：{nodes} 个节点 / {edges} 条边"
        + (f"（显式边 {exp}、增强边 {enh}）" if exp is not None or enh is not None else "")
        + f"，传播时长约 {dur_txt}。"
        + (f"增强边占比约 {_pct(enh_ratio)}，明显偏高，提示存在水军操纵 / 协同造假 / 异常传播图谱嫌疑，"
           "即内容被转发/互动的结构呈非自然的同步特征。" if (enh_ratio or 0) >= 0.3
           else "增强边占比不高，未见成规模的协同放大痕迹。" if enh_ratio is not None
           else "")
        + ("传播节点偏少，处于早期/稀疏阶段，模型置信度应下调，建议持续监测。" if sparse else "")
    )

    evidence_analysis = (
        f"· 预测标签 = {label_cn}；假新闻风险 = {_pct(fake_p)}，真实概率 = {_pct(m.get('real_probability'))}，"
        f"判定阈值 = {m.get('threshold', '未知')}。\n"
        f"· 传播证据 = {nodes} 节点 / {edges} 边"
        + (f"，增强边 {enh}（占比 {_pct(enh_ratio)}）" if enh is not None else "")
        + f"，传播时长 {dur_txt}。\n"
        + (f"· 原始信源：{m.get('source_url')}，建议做信源权威性核验。" if m.get("source_url")
           else "· 未提供原始信源，信源可信度存疑。")
    )

    risk_summary = f"综合风险等级：{risk_level}。" + (
        "传播协同迹象 + 较高假新闻风险，建议优先处置。" if risk_level == "高危"
        else "存在不确定性，建议人工复核后判断。" if risk_level == "存疑"
        else "暂未发现明显异常，可正常参考。"
    )

    recommendation = (
        "总体口径：高度警惕、请勿转发，建议立即人工复核并触发限流/打标；"
        "追踪短时聚集的账号集群，对配图与信源做反向溯源。" if risk_level == "高危"
        else "总体口径：暂不建议扩散，建议人工复核；优先核查文本事实与原始信源，关注后续是否协同放大。"
        if risk_level == "存疑"
        else "总体口径：可正常参考；保留常规抽样复核，对异常二次放大保持轻量监测。"
    )

    return _normalize({
        "risk_level": risk_level,
        "conclusion": conclusion,
        "content_analysis": content_analysis,
        "propagation_analysis": propagation_analysis,
        "evidence_analysis": evidence_analysis,
        "risk_summary": risk_summary,
        "recommendation": recommendation,
    })


# ---------------------------------------------------------------------------
# 5. 真实 LLM 调用（anthropic / openai）
# ---------------------------------------------------------------------------
def _anthropic_report(material: Dict, timeout_s: int) -> Dict:
    import anthropic
    client = anthropic.Anthropic(api_key=config.LLM_API_KEY, timeout=timeout_s)
    resp = client.messages.create(
        model=config.LLM_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(material)}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return _normalize(_extract_json(text))


def _openai_report(material: Dict, timeout_s: int) -> Dict:
    from openai import OpenAI
    client = OpenAI(api_key=config.LLM_API_KEY, timeout=timeout_s)
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        max_tokens=config.LLM_MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(material)},
        ],
    )
    return _normalize(_extract_json(resp.choices[0].message.content))


# ---------------------------------------------------------------------------
# 6. 对外主入口
# ---------------------------------------------------------------------------
def generate_report(case_cache: Dict, *, timeout_s: Optional[int] = None) -> Dict:
    """输入案例缓存 dict，输出结构化研判报告 dict（含 generated_at / model / latency_ms）。"""
    timeout_s = timeout_s or config.LLM_TIMEOUT_S
    material = build_llm_input(case_cache)
    provider = (config.LLM_PROVIDER or "mock").lower()

    t0 = time.time()
    used_provider = provider
    error = None
    if provider == "mock" or not config.LLM_API_KEY:
        report = _mock_report(material)
        used_provider = "mock"
    else:
        try:
            if provider == "anthropic":
                report = _anthropic_report(material, timeout_s)
            elif provider == "openai":
                report = _openai_report(material, timeout_s)
            else:
                report = _mock_report(material)
                used_provider = "mock"
        except Exception as e:                       # 超时/限流/无密钥/解析失败 → 降级
            report = _mock_report(material)
            used_provider = f"mock(fallback from {provider})"
            error = str(e)[:200]

    report["provider"] = used_provider
    report["model"] = config.LLM_MODEL if used_provider.startswith(("anthropic", "openai")) else "rule-based-mock"
    report["latency_ms"] = int((time.time() - t0) * 1000)
    report["generated_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    if error:
        report["_error"] = error
    return report
