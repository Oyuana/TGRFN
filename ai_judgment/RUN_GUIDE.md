# 实战运行手册 · 用真实数据集跑通研判

> 适用场景：四个模块已就绪，你拿到队员/学姐**训练好的权重 + 预处理数据**，
> 想在自己机器上跑通「真实模型 → 量化指标 → LLM 报告 → 前端展示」全链路。
>
> 核心结论：**业务代码基本不用改**，主要工作是 ①装环境 ②摆数据 ③配环境变量。
> 唯一可能要看的代码是 `data_adapter.py`（仅当队员的落盘格式与 GossipCop.py 不一致时）。

---

## 〇、只想"展示最佳效果"？直接打开前端（零依赖）

如果你**还没跑起来后端 / 模型**，但要做演示或答辩，直接用浏览器打开
`ai_judgment/前端.html` 即可——它内置了 3 个精选演示案例（高危造假 / 早期存疑 / 安全真实），
自带传播拓扑演化、量化指标和**详细 AI 研判报告**，无需任何后端服务。

- 默认 `USE_LIVE_API = false`（前端脚本顶部），即纯前端演示模式，开箱即用。
- 想接真实后端时，把它改成 `true`：会优先请求后端，后端不可用时**自动回退**到内置演示数据，演示不会"白屏"。
- 内置案例 / 报告文案集中在前端文件里的 `window.DEMO`（`PROFILES` 段），可按需改文案、加案例。

> 想要"开箱即出最好结果"：打开后默认载入的就是高危造假案例
> `gossipcop-847028`，研判等级=高危、含 5 条证据 + 传播分析 + 处置建议，最适合截图 / 路演。

---

## 一、需要找队员要的东西

| 物料 | 对应 GossipCop.py | 说明 |
|------|-------------------|------|
| 训练好的权重 `best_model_*.tar` | `save_checkpoint` 产物 | 推理必需；没有它结果是随机的 |
| `n_neighbors.txt` | 第 108 行 `neighbor_loader` 入参 | 传播邻居顺序 |
| `original_adj` | 第 110 行 `json.load` | 邻接表 |
| `n_add_time.txt` / `p_add_time.txt` | 第 36/42 行 | 新闻/帖子时间戳 |
| `normalized_news_nodes/` | 第 210 行 `data_loader` | 新闻节点 embedding（含 label）|
| `normalized_post_nodes/` | 第 208 行 | 帖子节点 embedding |
| `normalized_user_nodes/` | 第 209 行 | 用户节点 embedding |
| 训练时的超参 | 第 1016-1020 行 | 核对 `config.MODEL_HPARAMS` 是否一致 |

> 同时确认数据集的 **id 前缀**（GossipCop 是 `gossipcop-`）。若不同，设
> `TGRFN_NEWS_ID_PREFIX`。

---

## 二、装环境

```bash
# 1) 主项目依赖（含 torch，版本与队员训练环境对齐最稳妥）
pip install -r requirements.txt
# 2) API/服务依赖
pip install -r ai_judgment/requirements.txt
# 3) 仅用真实大模型时再装（mock 模式不需要）
pip install anthropic        # 或 openai
```

先验证纯量化内核（不需要 torch / 数据）：

```bash
python -m ai_judgment.internals      # 打印两个自测用例，能跑通即说明判别逻辑 OK
```

---

## 三、配环境变量（关键步骤，指向你的真实数据）

```bash
# —— 模型权重 ——
export TGRFN_MODEL_PATH=/abs/path/to/best_model_0.tar
export TGRFN_DEVICE=auto                 # 有 GPU 自动用 cuda，没有就 cpu

# —— 数据 7 件套（替换成你机器上的真实路径；目录路径以 / 结尾）——
export TGRFN_NEIGHBORS_FILE=/abs/.../fnn_gossipcop_u50/n_neighbors.txt
export TGRFN_ADJ_FILE=/abs/.../fnn_gossipcop_u50/original_adj
export TGRFN_N_TIME_FILE=/abs/.../fnn_gossipcop_u50/n_add_time.txt
export TGRFN_P_TIME_FILE=/abs/.../fnn_gossipcop_u50/p_add_time.txt
export TGRFN_NEWS_DIR=/abs/.../normalized_news_nodes/
export TGRFN_POST_DIR=/abs/.../normalized_post_nodes/
export TGRFN_USER_DIR=/abs/.../normalized_user_nodes/

# —— ratio：不设则由 data_adapter 从 news label 自动统计（推荐让它自动算）——
# export TGRFN_DATASET_RATIO=1.0

# —— LLM：先用 mock 跑通；要真实报告再切 anthropic/openai 并给 key ——
export TGRFN_LLM_PROVIDER=mock           # 或 anthropic / openai
# export TGRFN_LLM_API_KEY=sk-...
```

> 7 个路径只要有**任意一个为空或文件不存在**，后端就判定"数据未挂载"，
> `/api/analyze` 返回 503、`/health` 显示 `degraded`——这是预期的明确报错，
> 不会给出假结果。

---

## 四、起服务 + 看前端

```bash
# 后端
uvicorn ai_judgment.api:app --host 0.0.0.0 --port 8000

# 自检
curl http://localhost:8000/health
# 期望 model_loaded=true, data_loaded=true, ratio 为真实统计值

# 单条研判（news_id 用你数据集里真实存在的 id）
curl -X POST http://localhost:8000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"news_id":"gossipcop-123456","with_report":true}'
```

**前端**：直接用浏览器打开根目录的
`index3.0_topnav_transparent_logo_final_logo_fixed (1).html`，进入「研判界面」
→「判定」标签 → 点「调用后端 API 研判」。页面启动会自动 `/health` 探活。
若后端不在 `localhost:8000`，先在浏览器控制台执行
`window.TGRFN_API_BASE='http://你的host:端口'` 再刷新。

> 注意：前端 mock 案例的 id 是 `GC-837` 等演示值。要让「调用后端」命中真实结果，
> 把 `cases` 里的 `id` 换成你数据集中真实存在的 `news_id`（或新增一条案例），
> 否则后端会返回 404 NEWS_NOT_FOUND 并自动回退到 mock 展示。

---

## 五、什么时候才需要改代码

| 情况 | 改哪里 |
|------|--------|
| 数据落盘格式与 GossipCop.py 不同（分隔符/字段顺序/batch 命名） | `data_adapter.py` 的 `_neighbor_loader` / `_load_*_nodes` |
| 训练超参与默认不同（维度、头数、层数、npu） | `config.MODEL_HPARAMS`（必须与训练一致，否则权重加载错位） |
| 权重 state_dict 键名差异较大 | `inference.load`（已用 `strict=False` 容忍少量差异，差异大需排查）|
| 想要真实 LLM 报告 | 设 `TGRFN_LLM_PROVIDER` + `TGRFN_LLM_API_KEY`，代码零改动 |
| 阈值/分级口径调整 | `config.py`（RISK_TIERS / 各告警阈值，或对应环境变量）|

---

## 六、自检顺序（出问题时逐层排除）

1. `python -m ai_judgment.internals` —— 量化逻辑（不依赖数据/torch）
2. `/health` 的 `model_loaded` —— 权重路径 & 超参对不对
3. `/health` 的 `data_loaded` & `ratio` —— 7 个数据路径对不对、label 能不能统计出比例
4. `/api/analyze` 单条 —— 端到端推理；报 404 就是 news_id 不在数据集里
5. 前端「调用后端 API 研判」—— 跨域/地址；失败会显示红字并回退 mock

> 当前沙箱无 GPU/torch，`model.py`/`inference.py` 未在真实 PyTorch 下跑过 forward。
> 首次在真实环境联调时，重点核对第 2、3 步两个 `*_loaded` 标志与 ratio 数值。
