# coding: utf-8
"""
TGRFN · AI 智能研判分析系统
============================

在 TGRFN 时序图假新闻检测模型之上，提取模型中间变量与损失函数信号，
结合大语言模型生成易读研判报告的端到端系统。

模块划分：
  config       —— 集中配置（路径 / 阈值 / LLM 密钥）
  model        —— import 无副作用的 DPSG 架构，forward 支持 return_internals
  internals    —— 量化研判内核（纯函数，可单测）：中间变量 -> 结构化指标
  inference    —— 单例推理引擎：加载权重 + 数据，对外 analyze(news_id)
  llm_report   —— LLM 研判报告生成（Module 1: mock；Module 2: anthropic/openai）
  api          —— FastAPI 服务：POST /api/analyze
"""
__version__ = "1.0.0"
