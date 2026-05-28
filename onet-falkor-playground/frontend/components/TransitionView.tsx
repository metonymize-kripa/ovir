'use client';

import { useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

interface CompetencyGap {
  element_id: string;
  name: string;
  type: string;
  target_im: number;
  target_lv: number;
  target_score: number;
  source_im?: number;
  source_lv?: number;
  source_score?: number;
  delta: number;
}

interface TechGap {
  commodity_code: string;
  title: string;
  hot_tech: boolean;
  in_demand: boolean;
}

interface TransitionResult {
  source: { code: string; title: string; job_zone: number | null };
  target: { code: string; title: string; job_zone: number | null };
  similarity: number;
  missing: CompetencyGap[];
  deficient: CompetencyGap[];
  transferable: CompetencyGap[];
  tech_gaps: TechGap[];
  summary: {
    missing_count: number;
    deficient_count: number;
    transferable_count: number;
    tech_gap_count: number;
    source_competency_count: number;
    target_competency_count: number;
  };
}

const TYPE_COLOR: Record<string, string> = {
  Skill: '#6366f1',
  Ability: '#8b5cf6',
  Knowledge: '#06b6d4',
  WorkActivity: '#10b981',
};

const TYPE_BADGE: Record<string, string> = {
  Skill: 'bg-indigo-900 text-indigo-300',
  Ability: 'bg-violet-900 text-violet-300',
  Knowledge: 'bg-cyan-900 text-cyan-300',
  WorkActivity: 'bg-emerald-900 text-emerald-300',
};

type Tab = 'missing' | 'deficient' | 'transferable' | 'tech';

function SimilarityMeter({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color =
    pct >= 75 ? '#22c55e' : pct >= 50 ? '#f97316' : '#ef4444';

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="text-5xl font-bold mono" style={{ color }}>
        {pct}%
      </div>
      <div className="text-xs text-slate-400 text-center">
        structural cosine similarity
        <br />
        <span className="text-slate-600">(GraphBLAS im×lv vectors — no embeddings)</span>
      </div>
      <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div
      className="flex flex-col items-center gap-1 px-6 py-4 rounded-lg"
      style={{ background: 'var(--surface)', border: `1px solid ${color}30` }}
    >
      <span className="text-2xl font-bold mono" style={{ color }}>
        {value}
      </span>
      <span className="text-xs text-slate-400 text-center">{label}</span>
    </div>
  );
}

function GapRow({ gap, category }: { gap: CompetencyGap; category: Tab }) {
  const badgeClass = TYPE_BADGE[gap.type] ?? 'bg-slate-800 text-slate-300';

  return (
    <tr className="border-b border-white/5 hover:bg-white/3 transition-colors">
      <td className="py-2.5 pr-3">
        <div className="flex items-center gap-2">
          <span className={`text-xs px-1.5 py-0.5 rounded mono ${badgeClass}`}>
            {gap.type.slice(0, 3).toUpperCase()}
          </span>
          <span className="text-sm">{gap.name}</span>
        </div>
      </td>
      <td className="py-2.5 pr-3 text-right mono text-sm">
        {gap.source_score != null ? gap.source_score.toFixed(1) : '—'}
      </td>
      <td className="py-2.5 pr-3 text-right mono text-sm">
        {gap.target_score != null ? gap.target_score.toFixed(1) : '—'}
      </td>
      <td className="py-2.5 text-right mono text-sm">
        {category === 'missing' ? (
          <span className="text-red-400">+{gap.delta != null ? gap.delta.toFixed(1) : '?'}</span>
        ) : gap.delta > 0 ? (
          <span className="text-orange-400">+{gap.delta != null ? gap.delta.toFixed(1) : '?'}</span>
        ) : (
          <span className="text-emerald-400">{gap.delta != null ? gap.delta.toFixed(1) : '?'}</span>
        )}
      </td>
    </tr>
  );
}

function GapChart({ missing, deficient }: { missing: CompetencyGap[]; deficient: CompetencyGap[] }) {
  const top = [...missing.slice(0, 8), ...deficient.slice(0, 7)]
    .sort((a, b) => b.delta - a.delta)
    .slice(0, 12)
    .map((g) => ({
      name: g.name.length > 22 ? g.name.slice(0, 22) + '…' : g.name,
      source: g.source_score ?? 0,
      target: g.target_score,
      type: g.type,
      category: g.source_score === undefined ? 'missing' : 'deficient',
    }));

  if (top.length === 0) return null;

  return (
    <div
      className="rounded-lg p-4"
      style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
    >
      <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
        Top gaps by delta (im×lv score)
      </div>
      <ResponsiveContainer width="100%" height={320}>
        <BarChart
          data={top}
          layout="vertical"
          margin={{ top: 0, right: 16, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2d3d" horizontal={false} />
          <XAxis
            type="number"
            tick={{ fill: '#64748b', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={160}
          />
          <Tooltip
            contentStyle={{
              background: '#1a1d27',
              border: '1px solid #2a2d3d',
              borderRadius: 8,
              fontSize: 12,
            }}
            labelStyle={{ color: '#e2e8f0', fontWeight: 600 }}
            formatter={(value: number, name: string) => [
              value.toFixed(1),
              name === 'source' ? 'Source score' : 'Target score',
            ]}
          />
          <Legend
            formatter={(v) => (
              <span style={{ color: '#94a3b8', fontSize: 12 }}>
                {v === 'source' ? 'Source' : 'Target'}
              </span>
            )}
          />
          <Bar dataKey="source" fill="#312e81" radius={[0, 3, 3, 0]} maxBarSize={12}>
            {top.map((entry, i) => (
              <Cell
                key={i}
                fill={TYPE_COLOR[entry.type] ? `${TYPE_COLOR[entry.type]}50` : '#312e81'}
              />
            ))}
          </Bar>
          <Bar dataKey="target" fill="#6366f1" radius={[0, 3, 3, 0]} maxBarSize={12}>
            {top.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.category === 'missing' ? '#ef444480' : '#f9731680'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function TransitionView({ result }: { result: TransitionResult }) {
  const [tab, setTab] = useState<Tab>('missing');

  const tabs: { key: Tab; label: string; count: number; color: string }[] = [
    { key: 'missing', label: 'Missing', count: result.summary.missing_count, color: '#ef4444' },
    { key: 'deficient', label: 'Deficient', count: result.summary.deficient_count, color: '#f97316' },
    { key: 'transferable', label: 'Transferable', count: result.summary.transferable_count, color: '#22c55e' },
    { key: 'tech', label: 'Tech gaps', count: result.summary.tech_gap_count, color: '#6366f1' },
  ];

  const activeItems: CompetencyGap[] =
    tab === 'missing'
      ? result.missing
      : tab === 'deficient'
      ? result.deficient
      : tab === 'transferable'
      ? result.transferable
      : [];

  return (
    <div className="flex flex-col gap-6">
      {/* Header */}
      <div
        className="rounded-lg p-4 flex items-center justify-between gap-4"
        style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
      >
        <div className="flex flex-col gap-0.5">
          <span className="text-xs text-slate-400">From</span>
          <span className="font-semibold">{result.source.title}</span>
          <span className="mono text-xs text-indigo-400">{result.source.code}</span>
        </div>
        <div className="text-slate-600 text-2xl">→</div>
        <div className="flex flex-col gap-0.5">
          <span className="text-xs text-slate-400">To</span>
          <span className="font-semibold">{result.target.title}</span>
          <span className="mono text-xs text-indigo-400">{result.target.code}</span>
        </div>
      </div>

      {/* Similarity + stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div
          className="md:col-span-1 rounded-lg p-5"
          style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
        >
          <SimilarityMeter value={result.similarity} />
        </div>
        <div className="md:col-span-3 grid grid-cols-2 sm:grid-cols-4 gap-3">
          <StatCard label="Missing" value={result.summary.missing_count} color="#ef4444" />
          <StatCard label="Deficient" value={result.summary.deficient_count} color="#f97316" />
          <StatCard label="Transferable" value={result.summary.transferable_count} color="#22c55e" />
          <StatCard label="Tech gaps" value={result.summary.tech_gap_count} color="#6366f1" />
        </div>
      </div>

      {/* Chart */}
      <GapChart missing={result.missing} deficient={result.deficient} />

      {/* Tab table */}
      <div
        className="rounded-lg overflow-hidden"
        style={{ border: '1px solid var(--border)' }}
      >
        {/* Tab bar */}
        <div
          className="flex border-b"
          style={{ background: 'var(--surface)', borderColor: 'var(--border)' }}
        >
          {tabs.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className="px-4 py-3 text-sm flex items-center gap-2 transition-colors"
              style={{
                borderBottom: tab === t.key ? `2px solid ${t.color}` : '2px solid transparent',
                color: tab === t.key ? t.color : '#64748b',
              }}
            >
              {t.label}
              <span
                className="text-xs px-1.5 py-0.5 rounded-full mono"
                style={{
                  background: tab === t.key ? `${t.color}20` : '#ffffff10',
                  color: tab === t.key ? t.color : '#64748b',
                }}
              >
                {t.count}
              </span>
            </button>
          ))}
        </div>

        {/* Content */}
        <div style={{ background: 'var(--bg)' }} className="p-4">
          {tab === 'tech' ? (
            result.tech_gaps.length === 0 ? (
              <p className="text-sm text-slate-500 py-4 text-center">No technology gaps.</p>
            ) : (
              <div className="flex flex-wrap gap-2">
                {result.tech_gaps.map((t) => (
                  <span
                    key={t.commodity_code}
                    className="text-xs px-2.5 py-1.5 rounded-md flex items-center gap-1.5"
                    style={{
                      background: 'var(--surface)',
                      border: `1px solid ${t.hot_tech ? '#6366f150' : 'var(--border)'}`,
                      color: t.hot_tech ? '#a5b4fc' : '#94a3b8',
                    }}
                  >
                    {t.hot_tech && (
                      <span className="text-orange-400" title="Hot technology">
                        ★
                      </span>
                    )}
                    {t.in_demand && (
                      <span className="text-emerald-400" title="In demand">
                        ↑
                      </span>
                    )}
                    {t.title}
                  </span>
                ))}
              </div>
            )
          ) : activeItems.length === 0 ? (
            <p className="text-sm text-slate-500 py-4 text-center">No items in this category.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="text-xs text-slate-500 uppercase tracking-wider border-b border-white/5">
                    <th className="pb-2 text-left font-medium">Competency</th>
                    <th className="pb-2 text-right font-medium pr-3">Source</th>
                    <th className="pb-2 text-right font-medium pr-3">Target</th>
                    <th className="pb-2 text-right font-medium">Delta</th>
                  </tr>
                </thead>
                <tbody>
                  {activeItems.map((gap) => (
                    <GapRow key={`${gap.type}-${gap.element_id}`} gap={gap} category={tab} />
                  ))}
                </tbody>
              </table>
              <p className="text-xs text-slate-600 mt-3">
                Scores = im × lv (importance × required level). Source column blank = competency absent in source occupation.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
