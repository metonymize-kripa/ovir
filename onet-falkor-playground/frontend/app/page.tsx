'use client';

import { useState } from 'react';
import OccupationSearch, { OccupationOption } from '@/components/OccupationSearch';
import TransitionView from '@/components/TransitionView';

type TransitionResult = Awaited<ReturnType<typeof fetchTransition>>;

async function fetchTransition(sourceCode: string, targetCode: string) {
  const res = await fetch('/api/transition', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_code: sourceCode, target_code: targetCode }),
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

export default function Home() {
  const [source, setSource] = useState<OccupationOption | null>(null);
  const [target, setTarget] = useState<OccupationOption | null>(null);
  const [result, setResult] = useState<TransitionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function compute() {
    if (!source || !target) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await fetchTransition(source.code, target.code);
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  const canCompute = !!source && !!target && !loading;

  return (
    <main className="min-h-screen px-4 py-10 max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span
            className="text-xs mono px-2 py-0.5 rounded"
            style={{ background: 'var(--accent-dim)', color: 'var(--accent)' }}
          >
            USE CASE A
          </span>
          <span className="text-xs text-slate-500">FalkorDB × O*NET 30.0</span>
        </div>
        <h1 className="text-2xl font-bold mb-1">Career Transition Engine</h1>
        <p className="text-sm text-slate-400 max-w-2xl">
          Computes the exact structural competency delta between two O*NET occupations via GraphBLAS
          matrix traversal. Similarity is cosine distance over weighted im×lv vectors — no
          embeddings, no text matching.
        </p>
      </div>

      {/* Search panel */}
      <div
        className="rounded-xl p-5 mb-6"
        style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
      >
        <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-4 items-end">
          <OccupationSearch
            label="Source occupation (current role)"
            value={source}
            onChange={setSource}
            disabled={loading}
          />

          <div className="flex justify-center pb-1">
            <span className="text-2xl text-slate-600">→</span>
          </div>

          <OccupationSearch
            label="Target occupation (desired role)"
            value={target}
            onChange={setTarget}
            disabled={loading}
          />
        </div>

        <div className="flex items-center justify-between mt-5 pt-4 border-t" style={{ borderColor: 'var(--border)' }}>
          <p className="text-xs text-slate-500">
            {source && target
              ? `${source.code} → ${target.code}`
              : 'Select two occupations to compute the transition delta.'}
          </p>
          <button
            onClick={compute}
            disabled={!canCompute}
            className="px-5 py-2 rounded-lg text-sm font-semibold transition-all"
            style={{
              background: canCompute ? 'var(--accent)' : 'var(--border)',
              color: canCompute ? 'white' : 'var(--muted)',
              cursor: canCompute ? 'pointer' : 'not-allowed',
            }}
          >
            {loading ? 'Computing…' : 'Compute Transition'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div
          className="rounded-lg px-4 py-3 mb-6 text-sm"
          style={{ background: '#450a0a', border: '1px solid #7f1d1d', color: '#fca5a5' }}
        >
          Error: {error}. Check that the backend is running at localhost:8000 and the graph is
          loaded.
        </div>
      )}

      {/* Loading skeleton */}
      {loading && (
        <div className="flex flex-col gap-4 animate-pulse">
          <div className="h-20 rounded-xl" style={{ background: 'var(--surface)' }} />
          <div className="grid grid-cols-4 gap-3">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="h-20 rounded-lg" style={{ background: 'var(--surface)' }} />
            ))}
          </div>
          <div className="h-72 rounded-lg" style={{ background: 'var(--surface)' }} />
        </div>
      )}

      {/* Results */}
      {result && !loading && <TransitionView result={result} />}

      {/* Footer hint */}
      {!result && !loading && (
        <div
          className="rounded-xl p-6 text-center"
          style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
        >
          <p className="text-slate-500 text-sm mb-3">Try a sample transition:</p>
          <div className="flex flex-wrap gap-2 justify-center">
            {[
              ['15-2031.00', '15-1252.00', 'Operations Research → Software Dev'],
              ['13-2011.00', '15-2051.00', 'Accountant → Data Scientist'],
              ['11-1011.00', '15-1299.09', 'Chief Executive → AI/ML Specialist'],
              ['29-1141.00', '15-1211.01', 'RN → Health Informatics'],
            ].map(([src, tgt, label]) => (
              <button
                key={label}
                onClick={() => {
                  // Fetch both occupations then set
                  Promise.all([
                    fetch(`/api/occupation/${src}`).then((r) => r.json()),
                    fetch(`/api/occupation/${tgt}`).then((r) => r.json()),
                  ]).then(([s, t]) => {
                    if (s?.occupation) {
                      setSource(s.occupation);
                    }
                    if (t?.occupation) {
                      setTarget(t.occupation);
                    }
                  });
                }}
                className="text-xs px-3 py-1.5 rounded-lg transition-colors hover:bg-white/5"
                style={{ border: '1px solid var(--border)', color: '#94a3b8' }}
              >
                {label}
              </button>
            ))}
          </div>
        </div>
      )}
    </main>
  );
}
