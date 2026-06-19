// 警示条：传播结构冲突指数（水军嫌疑度）
// index 通常 ∈ [0, ~2]，alert 决定是否高亮告警
export default function ConflictBar({ index = 0, alert = false, threshold = 0.5 }) {
  const ratio = Math.min(index / (threshold * 2), 1) // 归一到 [0,1] 仅用于条宽
  const pct = Math.round(ratio * 100)
  const color = alert ? '#ef4444' : index >= threshold * 0.6 ? '#f59e0b' : '#22c55e'

  return (
    <div>
      <div className="flex justify-between items-baseline mb-2">
        <span className="text-sm font-medium text-slate-600">传播结构冲突指数（水军嫌疑度）</span>
        <span className="text-lg font-bold" style={{ color }}>{index.toFixed(2)}</span>
      </div>
      <div className="h-4 w-full rounded-full bg-slate-100 overflow-hidden">
        <div className="h-full rounded-full transition-all"
             style={{ width: `${pct}%`, background: color }} />
      </div>
      <p className={`mt-2 text-xs ${alert ? 'text-red-500 font-semibold' : 'text-slate-400'}`}>
        {alert
          ? '⚠ 内容与传播图结构严重不一致，存在水军操纵 / 异常传播图谱嫌疑'
          : `阈值 ${threshold}，当前低于告警线，传播结构与内容基本一致`}
      </p>
    </div>
  )
}
