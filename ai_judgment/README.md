# TGRFN · AI 智能研判分析系统

在 TGRFN 时序图假新闻检测模型之上，提取**模型中间变量**（多模态注意力、内容
表示 vs 图结构表示的对齐距离）与**损失函数信号**（`loss_dis` / MSE），结合大
语言模型生成普通用户可读的研判报告。

> 本目录是 **Module 1（后端深度学习改造 + API 封装）** 的产出，配合 `prompts.py` /
> `PROMPTS.md`（Module 2）、`API_CONTRACT.md`（Module 3）。
>
> **Module 4（前端）** 的正式落地基于仓库根目录下既有的可视化大屏页面
> `index3.0_topnav_transparent_logo_final_logo_fixed (1).html`（单文件 HTML +
> ECharts，无需构建步骤），而非另起一个独立的 React 工程。该页面在「研判界面」
> 的「判定」「报告」标签页中新增了：
> - 研判等级横幅（高危/存疑/安全，对齐 `internals.py` 阈值与结构冲突上调规则）
> - 「调用后端 API 研判」按钮，按 `API_CONTRACT.md` 调用 `POST /api/analyze`，
>   将真实 `metrics`/`report` 覆盖页面上的 Mock 演示数据，并在「报告」页展示完整
>   LLM 文本报告（结论/证据/传播分析/建议）
> - 启动时自动 `GET /health` 探活，展示"数据源：Mock"或"后端在线"状态
>
> 后端地址默认 `http://localhost:8000`；部署到其它地址时在浏览器控制台执行
> `window.TGRFN_API_BASE = 'https://your-host:port'` 后刷新页面即可（`api.py` 已
> 开启 `allow_origins=*` 的 CORS，无需额外配置）。

## 目录结构

| 文件 | 职责 | 是否可独立运行 |
|------|------|----------------|
| `config.py` | 集中配置：模型路径、判别阈值、LLM 密钥（全部走环境变量） | ✅ |
| `internals.py` | **量化研判内核**：中间变量 → 结构化指标（纯函数） | ✅ `python -m ai_judgment.internals` |
| `model.py` | import 无副作用的 DPSG 架构；`forward(return_internals=True)` 额外吐中间量 | 需 torch |
| `inference.py` | 单例推理引擎：加载权重 + 数据，`analyze(news_id)` | 需 torch + 数据 |
| `llm_report.py` | LLM 报告生成（Module 1: mock；Module 2: anthropic/openai） | ✅（mock） |
| `api.py` | FastAPI：`POST /api/analyze`、`GET /health` | 需 fastapi |

## 核心设计

### 1. 从模型抽取的"事实依据"
- **判别风险 `fake_probability`**：基于真实判别规则 `odds = p/(1-p)` 与训练集正负
  样本比 `ratio` 比较（对应 `GossipCop.py:1078/1239`），而非简单的 `p>0.5`。
- **结构冲突指数**：`MSELoss(align_c, align_g)` —— 即训练里的 `loss_dis`
  （`GossipCop.py:1084`）。内容表示与传播图结构表示越不一致，越疑似"水军操纵 /
  异常传播图谱"。
- **模态注意力贡献**：text / image / time 三模态表示能量经 softmax 归一化。某模态
  占比过高触发"视觉篡改嫌疑""爆发式时间衰减"等告警。
- **传播稀疏度**：邻居规模低于阈值时下调置信度并提示"早期/稀疏传播"。

### 2. 为什么新建 `model.py` 而不改训练脚本
`models/train_and_evaluation/GossipCop.py` 在 **import 时**即加载训练机绝对路径
数据并执行完整训练循环，无法被 API 进程安全引用。因此把模型架构原样抽取为
import 无副作用、数据注入式的 `model.py`，并新增**向后兼容**的
`forward(return_internals=True)`——训练脚本无需任何改动即可继续使用旧的
`(pred, dist)` 返回。后续可让训练脚本反向 import 本模块以消除重复定义。

### 3. LLM 只解释、不计算
所有数值在 `internals.py` 算定，LLM 仅把指标翻译成自然语言报告，避免幻觉覆盖
模型真实计算结果。

## 快速开始

```bash
# 1) 仅验证量化内核（无需 torch）
python -m ai_judgment.internals

# 2) 启动 API（需安装 ai_judgment/requirements.txt）
export TGRFN_MODEL_PATH=/path/to/best_model_0.tar
export TGRFN_DATASET_RATIO=1.0          # 填对应数据集真实正负样本比
uvicorn ai_judgment.api:app --port 8000
```

## 唯一集成点

`inference.py::InferenceEngine.load_data_bundle()` 默认返回 `None`。部署时在此
复用 `GossipCop.py` 的 `neighbor_loader` / `data_loader` 装配 `DataStore` 与各
`news_id` 的邻居顺序、时间戳即可，其余链路已全部就绪（数据未挂载时 API 以 503
明确报错，不会静默出错）。
