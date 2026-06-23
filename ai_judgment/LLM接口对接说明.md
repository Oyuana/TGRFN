# LLM 研判报告接口 · 对接说明

> 给你（看不懂没关系，先看这段）：
> 队友那条消息就是想跟你**敲定"生成 LLM 研判报告"这个接口长什么样**，并问了你 8 个问题。
> 我已经把接口和代码都写好了，放在 `ai_judgment/report_service.py` 和 `ai_judgment/flask_llm_report.py`。
> 你**不用自己配置**——只要把下面【给队友的回复】整段发给他，他按里面的两步把代码接进去就行。

---

## 0. 一张图看懂这件事

```text
浏览器(前端)  ──POST /api/v1/cases/{case_id}/llm-report──▶  Flask 后端(172.16.110.110:8010)
   只传 case_id                                              │  ① 按 case_id 读已有案例缓存
   不传整篇新闻/拓扑图                                        │  ② 整理成 LLM 输入
   不碰 API Key                                              │  ③ 调用大模型(Key 在服务器)
                                                             │  ④ 结果按 case_id 缓存
   ◀────────────── 结构化 JSON 研判报告 ──────────────────────┘
```

**关键：API Key 只放在 Flask 服务器，浏览器永远不直接调用大模型。** 这正是队友强调的，我们的实现完全遵守。

---

## 1. 接口规范（已实现，双方照此对接）

**请求**
```
POST /api/v1/cases/<case_id>/llm-report          # 优先读缓存，没有才真生成
POST /api/v1/cases/<case_id>/llm-report?regenerate=1   # 点"重新生成"时用
Body(可选): { "regenerate": true }
```
前端只传 `case_id`。

**成功响应 `200`（结构化 JSON，不是一整段文本）**
```jsonc
{
  "code": 0, "message": "ok", "case_id": "gossipcop-847028",
  "report": {
    "risk_level": "高危",                 // 高危/存疑/安全，便于前端着色（附加项）
    "conclusion": "综合研判结论…",
    "content_analysis": "新闻内容分析…",
    "propagation_analysis": "传播结构分析…",
    "evidence_analysis": "证据/来源分析…",
    "risk_summary": "风险总结…",
    "recommendation": "处置/核查建议…",
    "generated_at": "2026-06-23T20:14:37+08:00",
    "model": "claude-sonnet-4-6",         // 使用的 LLM；降级时为 rule-based-mock
    "latency_ms": 4200,                   // 生成耗时
    "provider": "anthropic",              // anthropic/openai/mock/mock(fallback...)
    "cached": false                       // 是否来自缓存
  }
}
```
队友要求的 6 个字段（conclusion / content_analysis / propagation_analysis / evidence_analysis /
risk_summary / recommendation）+ generated_at / model / latency_ms **全部覆盖**，并多给了
risk_level / provider / cached 方便前端。

**失败/超时响应**：HTTP 仍 `200`，`report` 结构**完全一样**，只是 `provider` 变成
`mock(fallback...)`、带 `_error` 摘要——前端照常渲染兜底文案，**不会白屏**。
`case_id` 不存在 → `404 {"code":404,"message":"CASE_NOT_FOUND"}`。

---

## 2. 队友只需两步接入（代码已写好）

**第一步**：把 `ai_judgment/` 目录放进他的项目（或 `pip install -e .`）。

**第二步**：在他的 Flask 启动文件里加 3 行——
```python
from ai_judgment.flask_llm_report import make_llm_report_blueprint

# load_case_cache 用他自己"按 case_id 读案例缓存"的函数（返回 dict 或 None）
bp = make_llm_report_blueprint(load_case_cache=他已有的读缓存函数)
app.register_blueprint(bp)
```
> 如果案例缓存就是一堆 `<case_id>.json` 文件，连函数都不用传：设环境变量
> `TGRFN_CASE_CACHE_DIR=/缓存目录` 即可。

**配置（都在服务器环境变量，不下发前端）**
```bash
export TGRFN_LLM_PROVIDER=anthropic            # 或 openai；不配则用 mock(规则版，可离线演示)
export TGRFN_LLM_API_KEY=sk-xxxx               # 大模型 Key，只在服务器
export TGRFN_LLM_MODEL=claude-sonnet-4-6
export TGRFN_LLM_TIMEOUT=20                     # 调用超时(秒)
export TGRFN_REPORT_CACHE_DIR=./_report_cache   # 报告缓存目录(可选)
```
**缓存行为**：首次真实生成成功后按 `case_id` 落盘；再次打开优先读缓存；点"重新生成"`?regenerate=1`
会忽略缓存重算。（降级版报告不缓存，方便稍后重试拿到真报告。）

依赖：`flask`（他后端已有）；用真实大模型还需 `pip install anthropic`（或 `openai`）。
不配 Key 时走规则版 mock，无需联网也能返回完整结构化报告。

---

## 3. 前端（本仓库 `ai_judgment/前端.html` 已改好，无需再动）

本仓库的 `前端.html` 已经：① 删除内置演示数据，只走真实后端接口；② 选中案例时自动调用
`POST /api/v1/cases/{id}/llm-report` 生成研判报告（命中缓存秒回），并在「报告」标签页渲染
6 段结构化内容 + 风险等级 + 模型/耗时/生成时间，附「重新生成」「导出 PDF」按钮；
③ 判定页与 PDF 导出同步用上报告里的结论与处置建议。
> 若前后端**分离部署**，把 `前端.html` 顶部的 `const API_BASE = ''` 改成后端地址
> （例如 `'http://172.16.110.110:8010'`）即可。

如果队友用的是他自己那份前端，核心就是把预留的 `requestLlmReport(caseId)` 改成真正请求接口：
```javascript
async function requestLlmReport(caseId, regenerate = false) {
  const url = `/api/v1/cases/${encodeURIComponent(caseId)}/llm-report` + (regenerate ? '?regenerate=1' : '');
  const res = await fetch(url, { method: 'POST', headers: {'Content-Type':'application/json'}, body: '{}' });
  const data = await res.json();
  return data.report;   // 直接拿 6 段结构化字段渲染到报告区
}
```
渲染时用 `report.risk_level` 着色，依次显示 conclusion / content_analysis / propagation_analysis /
evidence_analysis / risk_summary / recommendation，底部小字标 `model · latency_ms · generated_at`。

---

## 4. 案例缓存字段映射（我已做容错，建议核对）

我按前端 `/analysis` 的结构做了兼容读取，**字段名不一致也大多能读**，但请队友核对这些字段名：

| LLM 需要的信息 | 我读取的字段（任一命中即可） |
|----------------|------------------------------|
| 标题 / 正文 / 摘要 | `case.title` / `case.body`(或 content/text) / `case.summary` |
| 发布时间 / 原始链接 | `case.publish_time` / `case.source_url`(或 url) |
| 数据集 | `case.dataset` |
| 预测标签 / 真假概率 / 阈值 | `prediction.pred_label` / `fake_probability` / `real_probability` / `threshold` |
| 节点数 / 边数 / 显式边 / 增强边 | `statistics.display_node_count` / `num_edges` / `explicit_edge_count` / `enhanced_edge_count` |
| 传播时长 | `statistics.propagation_duration_seconds`，或由 `graph.time_range` 自动算 |
| 重要传播节点(可选) | `key_nodes`（不传整图） |

> 若字段名差别较大，把真实缓存的一个样例 JSON 发我，我 5 分钟改好映射。

---

## 5. 【给队友的回复】——把下面整段复制发给他

> 收到，接口我这边定好了，按你的建议来，已经把可直接接入的代码写好（`ai_judgment/report_service.py` + `flask_llm_report.py`）。逐条回你的问题：
>
> **接口**：就用你建议的 `POST /api/v1/cases/{case_id}/llm-report`，前端只传 case_id，由 Flask 按 id 读案例缓存→整理→调大模型→缓存→返回。返回**结构化 JSON**，字段：conclusion / content_analysis / propagation_analysis / evidence_analysis / risk_summary / recommendation / risk_level / generated_at / model / latency_ms / provider / cached，你要的都在。
>
> 1. **LLM 跑在哪台/哪个端口**：建议**不单独起服务**，直接把 LLM 调用做成 Flask 蓝图，跑在你现有的 172.16.110.110:8010 同一个进程里——少一跳、少一台机器，Key 也只在这一个后端。接入就两步：`make_llm_report_blueprint(load_case_cache=你的读缓存函数)` + `app.register_blueprint(bp)`。
> 2. **172.16.110.110 能否直接访问**：这是内网，我在外网打不开它和 health 接口，需要你在内网/VPN 里自测 health。我也不需要连它——LLM 调用就放在这台 Flask 里，只要**这台服务器能访问外网**（api.anthropic.com / api.openai.com 或你们的代理）即可。
> 3. **是否需要 token**：前端→Flask 这跳，比赛 demo 可先不加；要加就用一个内部 header（如 X-Internal-Token），前端配置带上。Flask→大模型那跳用 API Key。
> 4. **API Key 谁保管**：就由 172.16.110.110 的 Flask 保管，放环境变量 `TGRFN_LLM_API_KEY`，绝不下发前端。
> 5. **返回什么格式**：结构化 JSON（同上），不是 markdown / 纯文本。
> 6. **一次生成多久**：真实大模型约 3~10 秒，已设超时（默认 20s，可调）。
> 7. **超时/失败返回什么**：返回**完全相同结构**的 JSON，只是 provider=mock(fallback…)、带 _error，HTTP 仍 200，前端照常渲染兜底文案不白屏；case_id 不存在返回 404 + {code:404,message:"CASE_NOT_FOUND"}。
> 8. **要不要发缓存字段结构**：**要**。我按前端 /analysis 的 case/prediction/statistics/graph 结构做了兼容读取，但麻烦你把真实缓存的一个样例 JSON 发我对一下字段名（尤其正文/摘要、threshold、explicit/enhanced 边数、传播时长的字段名），我对齐一下最稳。
>
> 缓存：首次真实生成成功后按 case_id 落盘，再次打开优先读缓存，保留 `?regenerate=1` 重新生成。Key 没配时会走规则版兜底（可离线演示）。你那边把缓存读取函数名发我，或确认用 `TGRFN_CASE_CACHE_DIR` 目录读 `<case_id>.json` 即可。

---

## 附：相关文件

| 文件 | 作用 |
|------|------|
| `ai_judgment/report_service.py` | 报告生成核心（材料整理 + 提示词 + 大模型调用 + 规则兜底），不依赖框架 |
| `ai_judgment/flask_llm_report.py` | Flask 蓝图：`POST /api/v1/cases/<id>/llm-report` + 缓存 + 重生成 + 超时降级 |
| `ai_judgment/config.py` | 集中配置：LLM 厂商 / Key / 模型 / 超时（全部环境变量） |
