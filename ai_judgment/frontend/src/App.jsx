import { useState } from 'react'
import { fetchReportData } from './api'
import ConclusionBanner from './components/ConclusionBanner'
import ConfidenceGauge from './components/ConfidenceGauge'
import ConflictBar from './components/ConflictBar'
import ModalityRadar from './components/ModalityRadar'
import ReportSection from './components/ReportSection'

export default function App() {
  const [newsId, setNewsId] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [data, setData] = useState(null)

  async function handleAnalyze() {
    if (!newsId.trim()) return
    setLoading(true); setError(''); setData(null)
    try {
      // —— 联调入口：调用后端 POST /api/analyze ——
      const res = await fetchReportData(newsId.trim())
      setData(res)
    } catch (e) {
      setError(e.message || '研判失败，请重试')
    } finally {
      setLoading(false)
    }
  }

  const metrics = data?.metrics
  const report = data?.report

  return (
    <div className="min-h-screen max-w-5xl mx-auto px-4 py-8">
      <header className="mb-6">
        <h1 className="text-2xl font-bold text-slate-800">TGRFN · AI 智能研判分析系统</h1>
        <p className="text-sm text-slate-400 mt-1">时序图关系假新闻检测 · 模型量化 + 大模型研判</p>
      </header>

      {/* 输入区 */}
      <div className="flex gap-3 mb-8">
        <input
          value={newsId}
          onChange={(e) => setNewsId(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
          placeholder="输入新闻 ID，例如 gossipcop-123456"
          className="flex-1 px-4 py-3 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-300"
        />
        <button
          onClick={handleAnalyze}
          disabled={loading}
          className="px-6 py-3 rounded-xl bg-indigo-500 text-white font-medium hover:bg-indigo-600 disabled:opacity-50"
        >
          {loading ? '研判中…' : '开始研判'}
        </button>
      </div>

      {error && (
        <div className="mb-6 rounded-xl bg-red-50 border border-red-200 text-red-600 px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {data && (
        <div className="space-y-6">
          {/* 1. 核心结论区 */}
          <ConclusionBanner report={report} />

          {/* 2. 量化指标可视化区 */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white rounded-2xl shadow-sm p-6 flex flex-col items-center justify-center">
              <ConfidenceGauge value={metrics.fake_probability} />
              <p className="text-xs text-slate-400 mt-2">
                模型判别：{metrics.predicted_label === 'FAKE' ? '虚假' : '真实'} ·
                置信度 {Math.round(metrics.confidence * 100)}%
              </p>
            </div>
            <div className="bg-white rounded-2xl shadow-sm p-6 flex items-center">
              <div className="w-full">
                <ConflictBar
                  index={metrics.structural_conflict_index}
                  alert={metrics.structural_conflict_alert}
                />
                {metrics.sparse_propagation && (
                  <p className="mt-4 text-xs text-amber-500">
                    ⓘ 早期/稀疏传播（邻居 {metrics.num_neighbors} 个），置信度已下调
                  </p>
                )}
              </div>
            </div>
            <div className="bg-white rounded-2xl shadow-sm p-6">
              <ModalityRadar
                attention={metrics.modality_attention}
                dominant={metrics.dominant_modality}
                alert={metrics.modality_dominance_alert}
              />
            </div>
          </div>

          {/* 3. 智能分析报告区 */}
          <ReportSection report={report} />

          <p className="text-center text-xs text-slate-300">
            耗时 {data.elapsed_ms} ms · ratio 基准 {metrics.ratio_baseline}
          </p>
        </div>
      )}
    </div>
  )
}
