// 智能分析报告区：优雅展示 LLM 生成的证据分析与传播链路分析
export default function ReportSection({ report }) {
  if (!report) return null
  return (
    <div className="bg-white rounded-2xl shadow-sm p-6 space-y-5">
      <div className="flex items-center gap-2">
        <span className="text-xl">🧠</span>
        <h3 className="text-lg font-bold text-slate-700">智能分析报告</h3>
        {report._provider && (
          <span className="ml-auto text-xs text-slate-400">来源：{report._provider}</span>
        )}
      </div>

      <div>
        <h4 className="text-sm font-semibold text-slate-500 mb-2">关键证据</h4>
        <ul className="space-y-2">
          {(report.Evidence || []).map((e, i) => (
            <li key={i} className="flex gap-2 text-sm text-slate-600">
              <span className="text-indigo-400 mt-0.5">▪</span>
              <span>{e}</span>
            </li>
          ))}
        </ul>
      </div>

      <div>
        <h4 className="text-sm font-semibold text-slate-500 mb-2">传播链路分析</h4>
        <p className="text-sm text-slate-600 leading-relaxed bg-slate-50 rounded-lg p-3">
          {report.Propagation_Analysis}
        </p>
      </div>

      {report._error && (
        <p className="text-xs text-amber-500">注：LLM 调用降级（{report._error}），以上为规则引擎兜底文本。</p>
      )}
    </div>
  )
}
