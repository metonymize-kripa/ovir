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

interface JdMatchResult {
  code: string;
  title: string;
  match_count: number;
  sample_tasks: string[];
}

export default function Home() {
  const [source, setSource] = useState<OccupationOption | null>(null);
  const [target, setTarget] = useState<OccupationOption | null>(null);
  const [result, setResult] = useState<TransitionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // JD matching states
  const [searchTab, setSearchTab] = useState<'search' | 'jd'>('search');
  const [jdText, setJdText] = useState('');
  const [jdMatches, setJdMatches] = useState<JdMatchResult[]>([]);
  const [jdLoading, setJdLoading] = useState(false);

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

  async function matchJd() {
    if (!jdText.trim()) return;
    setJdLoading(true);
    setError(null);
    setJdMatches([]);
    try {
      const res = await fetch('/api/match-jd', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jd_text: jdText }),
      });
      if (!res.ok) throw new Error('Failed to match Job Description');
      const data = await res.json();
      setJdMatches(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setJdLoading(false);
    }
  }

  const canCompute = !!source && !!target && !loading;

  return (
    <main className="min-h-screen px-4 py-10 max-w-5xl mx-auto">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span
            className="text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded"
            style={{ background: 'var(--accent-dim)', color: 'var(--accent)' }}
          >
            TRANSITION CORE
          </span>
          <span className="text-xs text-slate-500">FalkorDB × O*NET 30.0 GraphBLAS engine</span>
        </div>
        <h1 className="text-3xl font-extrabold mb-1.5 tracking-tight">O*NET Career Transition Engine</h1>
        <p className="text-sm text-slate-400 max-w-3xl leading-relaxed">
          Computes the exact structural competency delta between two occupations via single-traversal
          matrix math. Compare structural capability deltas against raw text matching, select active tasks
          to test automation exposure, and map raw job descriptions directly to federal codes.
        </p>
      </div>

      {/* Input tabs */}
      <div className="flex gap-2 mb-4 border-b border-white/5">
        <button
          onClick={() => setSearchTab('search')}
          className={`px-4 py-2.5 text-xs font-bold uppercase tracking-wider transition-colors border-b-2 ${searchTab === 'search' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-500'}`}
        >
          🔍 Select Career Pair
        </button>
        <button
          onClick={() => setSearchTab('jd')}
          className={`px-4 py-2.5 text-xs font-bold uppercase tracking-wider transition-colors border-b-2 ${searchTab === 'jd' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-500'}`}
        >
          📝 JD Free-Text Matcher (Use Case B)
        </button>
      </div>

      {/* Search panel */}
      {searchTab === 'search' ? (
        <div
          className="rounded-xl p-6 mb-6 shadow-xl"
          style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
        >
          <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-6 items-end">
            <OccupationSearch
              label="Source occupation (current role)"
              value={source}
              onChange={setSource}
              disabled={loading}
            />

            <div className="flex justify-center pb-3">
              <span className="text-2xl text-slate-600 font-light">→</span>
            </div>

            <OccupationSearch
              label="Target occupation (desired role)"
              value={target}
              onChange={setTarget}
              disabled={loading}
            />
          </div>

          <div className="flex items-center justify-between mt-6 pt-5 border-t" style={{ borderColor: 'var(--border)' }}>
            <p className="text-xs text-slate-500">
              {source && target
                ? `${source.code} → ${target.code}`
                : 'Select both a source and target occupation to calculate structural deltas.'}
            </p>
            <button
              onClick={compute}
              disabled={!canCompute}
              className="px-6 py-2.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all"
              style={{
                background: canCompute ? 'var(--accent)' : 'var(--border)',
                color: canCompute ? 'white' : 'var(--muted)',
                cursor: canCompute ? 'pointer' : 'not-allowed',
              }}
            >
              {loading ? 'Computing…' : 'Compute delta'}
            </button>
          </div>
        </div>
      ) : (
        <div
          className="rounded-xl p-6 mb-6 shadow-xl flex flex-col gap-4"
          style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
        >
          <div className="flex flex-col gap-2">
            <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Paste Non-Standard Job Description
            </label>
            <textarea
              value={jdText}
              onChange={(e) => setJdText(e.target.value)}
              placeholder="Paste job details (e.g., 'Build custom WebGPU canvas pipelines, manage AWS servers, and coordinate stakeholder meetings.')"
              className="w-full h-32 rounded-lg p-3 text-sm outline-none resize-none leading-relaxed"
              style={{
                background: 'var(--bg)',
                border: '1px solid var(--border)',
                color: 'var(--text)',
              }}
            />
          </div>

          <button
            onClick={matchJd}
            disabled={jdLoading || !jdText.trim()}
            className="px-6 py-2.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all self-end"
            style={{
              background: jdText.trim() ? 'var(--accent)' : 'var(--border)',
              color: jdText.trim() ? 'white' : 'var(--muted)',
              cursor: jdText.trim() ? 'pointer' : 'not-allowed',
            }}
          >
            {jdLoading ? 'Analyzing labor graph…' : 'Match to O*NET Occupations'}
          </button>

          {/* JD Match Results */}
          {jdMatches.length > 0 && (
            <div className="flex flex-col gap-4 border-t pt-5" style={{ borderColor: 'var(--border)' }}>
              <div className="text-xs font-bold text-slate-400 uppercase tracking-wider">
                Top 5 Matching Occupations in O*NET via Tasks Traversal
              </div>
              <div className="grid grid-cols-1 gap-3">
                {jdMatches.slice(0, 5).map((match) => (
                  <div
                    key={match.code}
                    className="rounded-lg p-4 flex flex-col md:flex-row justify-between items-start md:items-center gap-4 border"
                    style={{ background: 'var(--bg)', borderColor: 'var(--border)' }}
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="mono text-xs text-indigo-400 bg-indigo-500/10 px-1.5 py-0.2 rounded font-bold">
                          {match.code}
                        </span>
                        <span className="text-[10px] text-emerald-400 bg-emerald-500/10 px-1.5 py-0.2 rounded font-semibold">
                          {match.match_count} Tasks Overlap
                        </span>
                      </div>
                      <span className="font-semibold text-sm block mb-1 text-slate-200">{match.title}</span>
                      <div className="text-[11px] text-slate-500 leading-normal line-clamp-2">
                        <strong>Overlapping task context:</strong> {match.sample_tasks.join('; ')}
                      </div>
                    </div>

                    <div className="flex gap-2 w-full md:w-auto">
                      <button
                        onClick={async () => {
                          const occRes = await fetch(`/api/occupation/${match.code}`).then(r => r.json());
                          if (occRes?.occupation) {
                            setSource(occRes.occupation);
                            setSearchTab('search');
                          }
                        }}
                        className="flex-1 md:flex-initial px-3 py-1.5 rounded bg-white/5 border border-white/10 text-xs font-semibold hover:bg-white/10 text-slate-300"
                      >
                        Set as Source
                      </button>
                      <button
                        onClick={async () => {
                          const occRes = await fetch(`/api/occupation/${match.code}`).then(r => r.json());
                          if (occRes?.occupation) {
                            setTarget(occRes.occupation);
                            setSearchTab('search');
                          }
                        }}
                        className="flex-1 md:flex-initial px-3 py-1.5 rounded bg-indigo-500/20 border border-indigo-500/40 text-xs font-semibold hover:bg-indigo-500/30 text-indigo-300"
                      >
                        Set as Target
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div
          className="rounded-lg px-4 py-3 mb-6 text-sm"
          style={{ background: '#450a0a', border: '1px solid #7f1d1d', color: '#fca5a5' }}
        >
          Error: {error}. Check that the backend is running at 127.0.0.1:8000 and the graph is
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
          <p className="text-slate-500 text-sm mb-3 font-medium">Try a sample career transition:</p>
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
                  Promise.all([
                    fetch(`/api/occupation/${src}`).then((r) => r.json()),
                    fetch(`/api/occupation/${tgt}`).then((r) => r.json()),
                  ]).then(([s, t]) => {
                    if (s?.occupation) setSource(s.occupation);
                    if (t?.occupation) setTarget(t.occupation);
                  });
                }}
                className="text-xs px-3 py-1.5 rounded-lg transition-all hover:bg-white/5 border text-slate-400 hover:text-slate-200"
                style={{ border: '1px solid var(--border)' }}
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
