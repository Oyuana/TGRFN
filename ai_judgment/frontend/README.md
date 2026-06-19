# Module 4 · 前端展示 (React + TailwindCSS + Recharts)

TGRFN AI 智能研判分析系统的前端。严格对接 `../API_CONTRACT.md`。

## 页面结构

1. **核心结论区** `ConclusionBanner` —— 醒目展示 LLM 风险等级（高危/存疑/安全）+ 一句话结论 + 建议。
2. **量化指标可视化区**
   - `ConfidenceGauge` —— 仪表盘，模型预测置信度 / 假新闻风险 `fake_probability`。
   - `ConflictBar` —— 警示条，传播结构冲突指数（水军嫌疑度）`structural_conflict_index`。
   - `ModalityRadar` —— 雷达图，注意力权重贡献（文本 vs 图像 vs 时间）`modality_attention`。
3. **智能分析报告区** `ReportSection` —— LLM 生成的证据分析 + 传播链路分析。

## 联调入口

`src/api.js` 的 **`fetchReportData(newsId, withReport)`** 是唯一的后端调用点，
对应 `POST /api/analyze`。开发期 `vite.config.js` 已把 `/api`、`/health` 代理到
`http://localhost:8000`。

## 启动

```bash
cd ai_judgment/frontend
npm install
npm run dev          # http://localhost:5173
```

后端无需密钥即可联调（LLM 走 mock）。生产部署时配置后端环境变量切换真实大模型，
前端零改动。如需自定义后端地址，设 `VITE_API_BASE`。
