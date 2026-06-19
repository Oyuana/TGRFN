# coding: utf-8
"""
集中式配置 —— AI 智能研判分析系统 (Module 1: Backend & API)

所有"魔法数字"、路径、阈值、外部密钥读取都集中在这里，便于联调与部署时统一调整。
任何模块都应从这里读取配置，而不要在业务代码里硬编码。
"""
import os


# ---------------------------------------------------------------------------
# 1. 模型 / 设备
# ---------------------------------------------------------------------------
# 训练好的权重 checkpoint 路径（save_checkpoint 产出的 .tar）。
# 部署时通过环境变量覆盖，避免写死训练机的绝对路径。
MODEL_CHECKPOINT_PATH = os.getenv(
    "TGRFN_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "weights", "best_model_0.tar"),
)

# 推理设备："cuda"/"cuda:0"/"cpu"；默认自动探测。
DEVICE = os.getenv("TGRFN_DEVICE", "auto")

# DPSG 构造超参（与 GossipCop.py 训练时保持一致，见 GossipCop.py:1016-1020）
MODEL_HPARAMS = {
    "input_dim": [768, 512, 3, 29],
    "n_hidden_dim": 128,
    "u_hidden_dim": 128,
    "p_hidden_dim": 128,
    "out_embed_d": 208,
    "d_model": 208,
    "attn_heads": 8,
    "enc_layers": 1,
    "npu": 30,
}


# ---------------------------------------------------------------------------
# 2. 研判规则阈值（量化层 —— 给 LLM 的"事实依据"在这里产生）
# ---------------------------------------------------------------------------
# 训练集正负样本比 ratio = 真新闻数 / 假新闻数 （GossipCop.py:230）。
# 判别规则：odds = p/(1-p)；odds >= ratio -> 真，否则 -> 假。
# 部署时应填入对应数据集真实统计值；此处给出 GossipCop 的经验默认。
DATASET_RATIO = float(os.getenv("TGRFN_DATASET_RATIO", "1.0"))

# 把"判别间隔 margin"映射为 [0,1] 假新闻风险值的缩放系数（越大越陡峭）。
FAKE_RISK_SCALE = 1.0

# 假新闻风险分级阈值（基于 fake_risk ∈ [0,1]）
RISK_TIERS = {
    "SAFE": 0.35,        # fake_risk <  0.35      -> 基本可信
    "SUSPICIOUS": 0.65,  # 0.35 <= risk < 0.65    -> 存疑，建议复核
    # fake_risk >= 0.65   -> 高危
}

# 结构冲突指数（align_c 与 align_g 的相对 MSE）告警阈值。
# 超过该值，提示词将引导大模型研判"水军操纵 / 异常传播图谱"。
STRUCTURAL_CONFLICT_ALERT = float(os.getenv("TGRFN_CONFLICT_ALERT", "0.5"))

# 单一模态注意力占比告警阈值。某一模态归一化贡献超过该值即视为"该模态主导"，
# 引导大模型指出"视觉篡改嫌疑"或"爆发式时间衰减"等异常。
MODALITY_DOMINANCE_ALERT = float(os.getenv("TGRFN_MODALITY_ALERT", "0.6"))

# 传播稀疏判定：参与传播的邻居节点数低于该值时，结论置信度下调并提示"早期/稀疏传播"。
SPARSE_PROPAGATION_THRESHOLD = int(os.getenv("TGRFN_SPARSE_THRESHOLD", "10"))


# ---------------------------------------------------------------------------
# 3. 大模型 (LLM) 接入  —— Module 2 使用，Module 1 预留
# ---------------------------------------------------------------------------
# 供应商："anthropic" / "openai" / "mock"（mock 用于无密钥本地联调）
LLM_PROVIDER = os.getenv("TGRFN_LLM_PROVIDER", "mock")
LLM_MODEL = os.getenv("TGRFN_LLM_MODEL", "claude-sonnet-4-6")
LLM_API_KEY = os.getenv("TGRFN_LLM_API_KEY", "")
LLM_MAX_TOKENS = int(os.getenv("TGRFN_LLM_MAX_TOKENS", "1024"))
LLM_TIMEOUT_S = int(os.getenv("TGRFN_LLM_TIMEOUT", "30"))


def resolve_device():
    """把 DEVICE 配置解析为实际可用的 torch 设备字符串。"""
    if DEVICE != "auto":
        return DEVICE
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
