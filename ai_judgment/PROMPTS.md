# Module 2 · AI 研判规则与大模型提示词设计

本文件是**设计说明**；运行时的权威提示词在 `prompts.py`（单一数据源），二者如有
出入以 `prompts.py` 为准。

## 1. 核心设计原则

> **LLM 只解释，不计算。**

底层 DPSG 模型 + 规则引擎（`internals.py`）已经算定全部数值事实
（假新闻风险、结构冲突指数、模态贡献、传播稀疏度等）。大模型的职责仅是把这些
量化指标**翻译成普通用户能看懂的中文研判报告**，不得重算、不得推翻、不得编造
材料外的证据。这样可以避免 LLM 幻觉覆盖深度模型的真实计算结果。

## 2. 研判规则（提示词据此组织解释）

| 规则 | 触发量化字段 | LLM 应得出的研判方向 |
|------|--------------|----------------------|
| **A. 分类与置信度** | `decision_margin` / `fake_probability` / `ratio_baseline` | 判别基于 `odds=p/(1-p)` 与训练集正负样本比 `ratio` 比较，而非 0.5 阈值；偏离 ratio 越大置信度越高 |
| **B. 结构冲突（水军）** | `structural_conflict_index`, `structural_conflict_alert` | 告警为真时，明确指出"水军操纵 / 异常传播图谱"——内容与传播路径行为矛盾 |
| **C. 视觉异常** | `dominant_modality="image"` + `modality_dominance_alert` | 指出"视觉特征存在严重篡改嫌疑" |
| **C. 时间异常** | `dominant_modality="time"` + `modality_dominance_alert` | 指出"具有典型谣言的爆发式时间衰减特征" |
| **D. 传播稀疏** | `sparse_propagation` | 说明结论基于早期/稀疏数据，置信度已下调，建议持续监测 |

这些规则与底层模型的对应关系：
- `fake_probability` ← 真实判别规则 `odds vs ratio`（`GossipCop.py:1078/1239`）
- `structural_conflict_index` ← `MSELoss(align_c, align_g)`，即训练 `loss_dis`（`GossipCop.py:1084`）
- `modality_attention` ← text/image/time 三模态表示能量经 softmax 归一化

## 3. 输出契约（LLM 必须返回的 JSON）

```json
{
  "Risk_Level": "高危 | 存疑 | 安全",
  "Conclusion": "一句话核心结论",
  "Evidence": ["证据点1", "证据点2"],
  "Propagation_Analysis": "传播结构 / 水军嫌疑 / 时间特征分析",
  "Recommendation": "给用户的处置建议"
}
```

`llm_report.py` 会对该 JSON 做兜底：缺字段自动补全、Evidence 字符串自动转数组、
解析失败或调用异常自动降级为 mock，保证 `/api/analyze` 永远返回结构一致的 report。

## 4. 风险等级映射

- `fake_probability >= 0.65` 或触发结构冲突告警 ⇒ **高危**
- `0.35 <= fake_probability < 0.65` ⇒ **存疑**
- `fake_probability < 0.35` 且无告警 ⇒ **安全**

规则引擎在 `preliminary_risk_level` 给出预分级作为 LLM 的重要参考，最终等级由
LLM 结合所有告警综合判断。

## 5. Provider 切换（环境变量）

| 变量 | 说明 | 示例 |
|------|------|------|
| `TGRFN_LLM_PROVIDER` | `anthropic` / `openai` / `mock` | `anthropic` |
| `TGRFN_LLM_MODEL` | 模型 ID | `claude-sonnet-4-6` |
| `TGRFN_LLM_API_KEY` | 密钥（留空自动走 mock） | `sk-...` |

无密钥时全链路用 mock 即可联调；接真实模型只需配置以上三个环境变量，代码零改动。
