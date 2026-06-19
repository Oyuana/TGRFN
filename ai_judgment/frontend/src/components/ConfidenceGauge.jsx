import { RadialBarChart, RadialBar, PolarAngleAxis, ResponsiveContainer } from 'recharts'

// 仪表盘：展示「模型预测置信度 / 假新闻风险」(fake_probability)
// value ∈ [0,1]
export default function ConfidenceGauge({ value = 0, label = '假新闻风险' }) {
  const pct = Math.round(value * 100)
  const color = value >= 0.65 ? '#ef4444' : value >= 0.35 ? '#f59e0b' : '#22c55e'
  const data = [{ name: label, value: pct, fill: color }]

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-48 h-48">
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart
            innerRadius="70%" outerRadius="100%" data={data}
            startAngle={220} endAngle={-40}
          >
            <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
            <RadialBar background dataKey="value" cornerRadius={12} />
          </RadialBarChart>
        </ResponsiveContainer>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-4xl font-bold" style={{ color }}>{pct}%</span>
          <span className="text-xs text-slate-400 mt-1">{label}</span>
        </div>
      </div>
    </div>
  )
}
