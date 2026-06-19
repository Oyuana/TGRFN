// ============================================================================
// 前后端联调入口 (Module 4)
// ----------------------------------------------------------------------------
// 与 ai_judgment/API_CONTRACT.md 严格对应。后端联调只需关注本文件。
// 开发期 vite 已把 /api、/health 代理到 http://localhost:8000。
// ============================================================================

const BASE = import.meta.env.VITE_API_BASE || ''

/**
 * 研判主接口：POST /api/analyze
 * @param {string} newsId   待研判新闻 ID
 * @param {boolean} withReport 是否调用 LLM 生成文本报告（默认 true）
 * @returns {Promise<{news_id, elapsed_ms, metrics, report}>}
 */
export async function fetchReportData(newsId, withReport = true) {
  const resp = await fetch(`${BASE}/api/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ news_id: newsId, with_report: withReport }),
  })

  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`
    try {
      const body = await resp.json()
      detail = body.detail || detail
    } catch (_) { /* ignore */ }
    throw new Error(detail)
  }
  return resp.json()
}

/** 健康检查：GET /health */
export async function fetchHealth() {
  const resp = await fetch(`${BASE}/health`)
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`)
  return resp.json()
}
