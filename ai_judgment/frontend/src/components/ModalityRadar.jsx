import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, Tooltip,
} from 'recharts'

// 雷达图：注意力权重贡献（文本 vs 图像 vs 时间）
// attention: { text, image, time }，三项之和=1
const LABELS = { text: '文本', image: '图像', time: '时间' }

export default function ModalityRadar({ attention = {}, dominant = '', alert = false }) {
  const data = Object.keys(LABELS).map((k) => ({
    modality: LABELS[k],
    value: Math.round((attention[k] || 0) * 100),
    key: k,
  }))

  return (
    <div>
      <div className="flex justify-between items-baseline mb-1">
        <span className="text-sm font-medium text-slate-600">模态注意力贡献</span>
        {dominant && (
          <span className={`text-xs px-2 py-0.5 rounded-full ${alert ? 'bg-red-100 text-red-600' : 'bg-slate-100 text-slate-500'}`}>
            主导：{LABELS[dominant] || dominant}{alert ? '（异常）' : ''}
          </span>
        )}
      </div>
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data} outerRadius="75%">
            <PolarGrid />
            <PolarAngleAxis dataKey="modality" tick={{ fontSize: 13 }} />
            <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fontSize: 10 }} />
            <Tooltip formatter={(v) => `${v}%`} />
            <Radar dataKey="value" stroke="#6366f1" fill="#6366f1" fillOpacity={0.45} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
      {alert && (
        <p className="text-xs text-red-500 font-semibold text-center">
          {dominant === 'image' && '⚠ 视觉特征注意力占比异常，疑似图像篡改'}
          {dominant === 'time' && '⚠ 时间注意力占比异常，呈典型谣言爆发式时间衰减'}
          {dominant === 'text' && '⚠ 文本特征高度主导，建议核查文本事实'}
        </p>
      )}
    </div>
  )
}
