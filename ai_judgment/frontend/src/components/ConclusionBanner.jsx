// 核心结论区：醒目展示风险等级 + 一句话结论
const STYLE = {
  高危: { bg: 'bg-red-50', border: 'border-red-300', text: 'text-red-600', icon: '⛔', tag: 'bg-red-500' },
  存疑: { bg: 'bg-amber-50', border: 'border-amber-300', text: 'text-amber-600', icon: '⚠️', tag: 'bg-amber-500' },
  安全: { bg: 'bg-green-50', border: 'border-green-300', text: 'text-green-600', icon: '✅', tag: 'bg-green-500' },
}

export default function ConclusionBanner({ report }) {
  if (!report) return null
  const s = STYLE[report.Risk_Level] || STYLE['存疑']
  return (
    <div className={`rounded-2xl border-2 ${s.bg} ${s.border} p-6`}>
      <div className="flex items-center gap-3 mb-3">
        <span className="text-3xl">{s.icon}</span>
        <span className={`text-white text-sm font-bold px-3 py-1 rounded-full ${s.tag}`}>
          {report.Risk_Level}
        </span>
        <span className={`text-xl font-bold ${s.text}`}>研判结论</span>
      </div>
      <p className="text-lg text-slate-700 leading-relaxed">{report.Conclusion}</p>
      {report.Recommendation && (
        <p className={`mt-3 text-sm font-medium ${s.text}`}>建议：{report.Recommendation}</p>
      )}
    </div>
  )
}
