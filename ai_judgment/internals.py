# coding: utf-8
"""
量化研判内核 (Module 1 核心)
================================

把 DPSG 模型输出的**原始中间变量**转换成一组结构化、可解释的量化指标，
作为后续大模型 (LLM) 研判的"事实依据"。本模块是**纯函数 + 无外部依赖**
（仅依赖标准库 math/dataclasses），因此可以脱离 PyTorch 单独运行和单测。

输入（来自 model.py 改造后的 forward(return_internals=True)）：
  - probability      : float   sigmoid 输出 p ∈ (0,1)，越接近 1 越"真"
  - dist_mse         : float   MSELoss(align_c, align_g)，即 GossipCop.py 的 loss_dis
  - rep_scale        : float   align_c / align_g 的平均能量（用于把 MSE 归一化）
  - modality_energy  : dict    {"text":..,"image":..,"time":..} 三模态表示能量
  - num_neighbors    : int     参与传播聚合的邻居节点数（判断稀疏传播）

输出：JudgmentMetrics —— 见下方 dataclass。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _logit(p: float, eps: float = 1e-7) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    # 数值稳定的 sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _softmax(d: Dict[str, float]) -> Dict[str, float]:
    if not d:
        return {}
    m = max(d.values())
    exp = {k: math.exp(v - m) for k, v in d.items()}
    s = sum(exp.values()) or 1.0
    return {k: v / s for k, v in exp.items()}


# ---------------------------------------------------------------------------
# 输出数据结构
# ---------------------------------------------------------------------------
@dataclass
class JudgmentMetrics:
    """供 API 直接序列化为 JSON 的量化研判结果。"""
    # —— 概率 / 判别 ——
    fake_probability: float            # [0,1] 假新闻风险（已按 ratio 校准的判别风险）
    model_raw_score: float             # 模型原始 sigmoid 输出 p（越大越"真"）
    decision_margin: float             # logit(p) - log(ratio)，>=0 判真，<0 判假
    predicted_label: str               # "REAL" / "FAKE"
    ratio_baseline: float              # 使用的训练集正负样本比基准

    # —— 结构冲突（水军 / 异常传播图谱）——
    structural_conflict_index: float   # 归一化后的 align_c↔align_g 冲突度，越大越异常
    structural_conflict_alert: bool

    # —— 模态注意力贡献 ——
    modality_attention: Dict[str, float]  # {"text","image","time"} 归一化后求和=1
    dominant_modality: str
    modality_dominance_alert: bool

    # —— 传播规模 / 置信度修正 ——
    num_neighbors: int
    sparse_propagation: bool

    # —— 量化预分级（最终分级由 LLM 结合证据给出）——
    preliminary_risk_level: str        # "SAFE" / "SUSPICIOUS" / "HIGH_RISK"
    confidence: float                  # [0,1] 该量化结论的置信度

    # —— 给提示词用的人类可读触发器列表 ——
    triggered_rules: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def compute_judgment_metrics(
    probability: float,
    dist_mse: float,
    rep_scale: float,
    modality_energy: Dict[str, float],
    num_neighbors: int,
    *,
    ratio: float,
    fake_risk_scale: float = 1.0,
    risk_tiers: Optional[Dict[str, float]] = None,
    structural_conflict_alert_th: float = 0.5,
    modality_dominance_th: float = 0.6,
    sparse_threshold: int = 10,
) -> JudgmentMetrics:
    """把模型中间变量汇聚成量化研判指标。纯函数，便于单测。"""
    risk_tiers = risk_tiers or {"SAFE": 0.35, "SUSPICIOUS": 0.65}
    rules = []

    # 1) 判别间隔：真实规则是 odds=p/(1-p) 与 ratio 比较 (GossipCop.py:1078,1239)
    #    等价于 logit(p) 与 log(ratio) 比较。margin>=0 -> 真，<0 -> 假。
    margin = _logit(probability) - math.log(max(ratio, 1e-7))
    predicted_label = "REAL" if margin >= 0 else "FAKE"
    # 把"偏离 ratio"的间隔映射成 [0,1] 假新闻风险：margin 越负风险越高。
    fake_probability = _sigmoid(-margin * fake_risk_scale)

    # 2) 结构冲突指数：用平均表示能量归一化 MSE，得到尺度无关的相对冲突度。
    structural_conflict_index = float(dist_mse) / (float(rep_scale) + 1e-7)
    conflict_alert = structural_conflict_index >= structural_conflict_alert_th
    if conflict_alert:
        rules.append(
            "STRUCTURAL_CONFLICT_HIGH: 内容表示与传播图结构表示严重不一致，"
            "疑似水军操纵 / 异常传播图谱"
        )

    # 3) 模态注意力贡献：三模态表示能量 -> softmax 归一化（求和=1）
    modality_attention = _softmax({k: float(v) for k, v in modality_energy.items()})
    if modality_attention:
        dominant_modality = max(modality_attention, key=modality_attention.get)
        dominance = modality_attention[dominant_modality]
    else:
        dominant_modality, dominance = "unknown", 0.0
    dominance_alert = dominance >= modality_dominance_th
    if dominance_alert:
        if dominant_modality == "image":
            rules.append("MODALITY_IMAGE_DOMINANT: 视觉特征注意力占比异常，疑似图像篡改")
        elif dominant_modality == "time":
            rules.append("MODALITY_TIME_DOMINANT: 时间注意力占比异常，呈典型谣言爆发式时间衰减")
        elif dominant_modality == "text":
            rules.append("MODALITY_TEXT_DOMINANT: 文本特征主导判别，建议核查文本内容真实性")

    # 4) 传播稀疏 -> 置信度下调
    sparse = num_neighbors < sparse_threshold
    if sparse:
        rules.append(
            f"SPARSE_PROPAGATION: 传播邻居数={num_neighbors} 低于阈值"
            f"{sparse_threshold}，处于早期/稀疏阶段，结论置信度下调"
        )

    # 5) 量化预分级（最终分级交给 LLM；这里给保守的量化基线）
    if fake_probability < risk_tiers["SAFE"]:
        prelim = "SAFE"
    elif fake_probability < risk_tiers["SUSPICIOUS"]:
        prelim = "SUSPICIOUS"
    else:
        prelim = "HIGH_RISK"
    # 结构冲突会把分级上调一档
    if conflict_alert and prelim == "SAFE":
        prelim = "SUSPICIOUS"
    elif conflict_alert and prelim == "SUSPICIOUS":
        prelim = "HIGH_RISK"

    # 6) 置信度：间隔越大越自信；稀疏传播打 7 折
    confidence = min(1.0, abs(margin) / 3.0 + 0.3)
    if sparse:
        confidence *= 0.7

    return JudgmentMetrics(
        fake_probability=round(fake_probability, 4),
        model_raw_score=round(float(probability), 4),
        decision_margin=round(margin, 4),
        predicted_label=predicted_label,
        ratio_baseline=round(float(ratio), 4),
        structural_conflict_index=round(structural_conflict_index, 4),
        structural_conflict_alert=conflict_alert,
        modality_attention={k: round(v, 4) for k, v in modality_attention.items()},
        dominant_modality=dominant_modality,
        modality_dominance_alert=dominance_alert,
        num_neighbors=int(num_neighbors),
        sparse_propagation=sparse,
        preliminary_risk_level=prelim,
        confidence=round(confidence, 4),
        triggered_rules=rules,
    )


# ---------------------------------------------------------------------------
# 自测：无需 PyTorch / 数据即可运行  ->  python -m ai_judgment.internals
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    # 用例 A：高度疑似假新闻 + 结构冲突 + 图像主导
    a = compute_judgment_metrics(
        probability=0.18, dist_mse=0.9, rep_scale=1.0,
        modality_energy={"text": 0.5, "image": 3.0, "time": 0.4},
        num_neighbors=25, ratio=1.0,
    )
    # 用例 B：可信新闻 + 低冲突 + 文本主导 + 稀疏传播
    b = compute_judgment_metrics(
        probability=0.92, dist_mse=0.05, rep_scale=1.0,
        modality_energy={"text": 2.0, "image": 0.3, "time": 0.3},
        num_neighbors=4, ratio=1.0,
    )
    print("CASE A (疑似假):")
    print(json.dumps(a.to_dict(), ensure_ascii=False, indent=2))
    print("\nCASE B (可信/稀疏):")
    print(json.dumps(b.to_dict(), ensure_ascii=False, indent=2))
    assert a.predicted_label == "FAKE" and a.structural_conflict_alert
    assert a.dominant_modality == "image"
    assert b.predicted_label == "REAL" and b.sparse_propagation
    print("\n[OK] internals self-test passed.")
