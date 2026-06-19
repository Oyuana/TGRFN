# Module 3 · 前后端数据接口契约

前后端解耦开发的唯一依据。后端实现见 `api.py`，前端按本契约对接即可。
字段与 `internals.JudgmentMetrics` / `api.AnalyzeResponse` 保持一一对应。

- Base URL（开发）：`http://localhost:8000`
- 编码：UTF-8，`Content-Type: application/json`
- 已开启 CORS（开发期 `allow_origins=*`，生产应收敛）

---

## 1. `POST /api/analyze` —— 研判主接口

### Request

```json
{
  "news_id": "gossipcop-123456",
  "with_report": true
}
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `news_id` | string | 是 | 待研判新闻 ID |
| `with_report` | bool | 否 | 是否调用 LLM 生成自然语言报告，默认 `true`；置 `false` 仅返回量化指标（更快） |

### Response `200`

返回分两部分：**`metrics`（底层模型量化指标）** + **`report`（LLM 文本报告）**。

```json
{
  "code": 0,
  "message": "ok",
  "news_id": "gossipcop-123456",
  "elapsed_ms": 142,
  "metrics": {
    "fake_probability": 0.82,
    "model_raw_score": 0.18,
    "decision_margin": -1.5163,
    "predicted_label": "FAKE",
    "ratio_baseline": 1.0,
    "structural_conflict_index": 0.9,
    "structural_conflict_alert": true,
    "modality_attention": { "text": 0.0691, "image": 0.838, "time": 0.0929 },
    "dominant_modality": "image",
    "modality_dominance_alert": true,
    "num_neighbors": 6,
    "sparse_propagation": true,
    "preliminary_risk_level": "HIGH_RISK",
    "confidence": 0.56,
    "triggered_rules": [
      "STRUCTURAL_CONFLICT_HIGH: ...",
      "MODALITY_IMAGE_DOMINANT: ...",
      "SPARSE_PROPAGATION: ..."
    ]
  },
  "report": {
    "Risk_Level": "高危",
    "Conclusion": "模型判别为「虚假」，假新闻风险 82%……",
    "Evidence": ["……", "……"],
    "Propagation_Analysis": "参与传播节点数 6……",
    "Recommendation": "高度警惕，请勿转发，建议交由人工复核。",
    "_provider": "anthropic"
  }
}
```

### `metrics` 字段字典

| 字段 | 类型 | 含义 | 前端用途 |
|------|------|------|----------|
| `fake_probability` | float [0,1] | 经 ratio 校准的假新闻风险 | **仪表盘**（置信度/风险） |
| `model_raw_score` | float [0,1] | 模型原始 sigmoid 输出（越大越"真"） | 辅助展示 |
| `decision_margin` | float | `logit(p)-log(ratio)`，≥0 判真，<0 判假 | 辅助/调试 |
| `predicted_label` | "REAL"/"FAKE" | 模型判别标签 | 结论区 |
| `ratio_baseline` | float | 训练集正负样本比基准 | 说明文案 |
| `structural_conflict_index` | float ≥0 | 内容↔传播图结构冲突度（loss_dis 归一化） | **警示条**（水军嫌疑度） |
| `structural_conflict_alert` | bool | 是否触发结构冲突告警 | 警示条高亮 |
| `modality_attention` | object | `{text,image,time}` 归一化贡献，和=1 | **雷达图/饼图** |
| `dominant_modality` | string | 主导模态 | 雷达图标注 |
| `modality_dominance_alert` | bool | 单模态是否异常主导 | 告警标记 |
| `num_neighbors` | int | 参与传播聚合的邻居节点数 | 传播规模 |
| `sparse_propagation` | bool | 是否早期/稀疏传播 | 置信度提示 |
| `preliminary_risk_level` | "SAFE"/"SUSPICIOUS"/"HIGH_RISK" | 规则引擎预分级 | 兜底等级 |
| `confidence` | float [0,1] | 量化结论置信度 | 仪表盘副指标 |
| `triggered_rules` | string[] | 命中的规则文本 | 证据列表兜底 |

### `report` 字段字典（LLM 生成）

| 字段 | 类型 | 含义 |
|------|------|------|
| `Risk_Level` | "高危"/"存疑"/"安全" | 综合风险等级（**核心结论区主标**） |
| `Conclusion` | string | 一句话结论 |
| `Evidence` | string[] | 关键证据列表 |
| `Propagation_Analysis` | string | 传播/水军/时间分析段落 |
| `Recommendation` | string | 处置建议 |
| `_provider` | string | 来源：`anthropic`/`openai`/`mock`/`mock(fallback...)` |
| `_error` | string? | 仅降级时出现，LLM 调用错误摘要 |

### 错误响应

| HTTP | detail 前缀 | 含义 | 前端处理 |
|------|------------|------|----------|
| `404` | `NEWS_NOT_FOUND` | news_id 不存在 | 提示"未找到该新闻" |
| `503` | `DATA_NOT_LOADED` | 模型/数据未挂载 | 提示"服务未就绪" |
| `500` | `INFERENCE_ERROR` | 推理异常 | 提示"研判失败，请重试" |

FastAPI 错误体格式：`{ "detail": "<前缀>: <信息>" }`。

---

## 2. `GET /health` —— 健康检查

```json
{
  "status": "ok",           // ok | degraded
  "model_loaded": true,
  "data_loaded": true,
  "device": "cuda",
  "ratio": 1.0,
  "llm_provider": "anthropic"
}
```

前端可在进入页面时探活，`status=degraded` 时禁用提交并提示后端未就绪。

---

## 3. 约定补充

- 所有浮点已在后端 `round` 到 4 位，前端展示时按需再格式化（如百分比取整）。
- `modality_attention` 三项之和恒为 1（softmax 归一），可直接喂饼图/雷达图。
- 同一 `news_id` 在 eval 模式下结果可复现（已修复推理噪声门控）。
