'use client';

import { useEffect, useRef, useState } from 'react';

export interface OccupationOption {
  code: string;
  title: string;
  description: string;
  job_zone: number | null;
}

interface Props {
  label: string;
  value: OccupationOption | null;
  onChange: (occ: OccupationOption | null) => void;
  disabled?: boolean;
}

export default function OccupationSearch({ label, value, onChange, disabled }: Props) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<OccupationOption[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClick);
    return () => document.removeEventListener('mousedown', handleClick);
  }, []);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (!query.trim()) {
      setResults([]);
      setOpen(false);
      return;
    }
    debounceRef.current = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
        const data = await res.json();
        setResults(data);
        setOpen(data.length > 0);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 250);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [query]);

  function select(occ: OccupationOption) {
    onChange(occ);
    setQuery('');
    setOpen(false);
  }

  function clear() {
    onChange(null);
    setQuery('');
  }

  const jzColors: Record<number, string> = {
    1: 'bg-slate-600',
    2: 'bg-blue-700',
    3: 'bg-indigo-700',
    4: 'bg-violet-700',
    5: 'bg-purple-700',
  };

  return (
    <div className="flex flex-col gap-2" ref={containerRef}>
      <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
        {label}
      </label>

      {value ? (
        <div
          style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
          className="rounded-lg p-3 flex items-start justify-between gap-3"
        >
          <div className="flex flex-col gap-1 min-w-0">
            <div className="flex items-center gap-2">
              <span
                className="mono text-xs px-1.5 py-0.5 rounded"
                style={{ background: 'var(--accent-dim)', color: 'var(--accent)' }}
              >
                {value.code}
              </span>
              {value.job_zone && (
                <span
                  className={`text-xs px-1.5 py-0.5 rounded text-white ${jzColors[value.job_zone] ?? 'bg-slate-600'}`}
                >
                  Zone {value.job_zone}
                </span>
              )}
            </div>
            <span className="font-semibold text-sm">{value.title}</span>
            <span className="text-xs text-slate-400 line-clamp-2">{value.description}</span>
          </div>
          <button
            onClick={clear}
            disabled={disabled}
            className="text-slate-500 hover:text-slate-200 text-lg leading-none flex-shrink-0 mt-0.5"
          >
            ×
          </button>
        </div>
      ) : (
        <div className="relative">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={disabled}
            placeholder="Search by title or O*NET code…"
            className="w-full rounded-lg px-3 py-2.5 text-sm outline-none"
            style={{
              background: 'var(--surface)',
              border: '1px solid var(--border)',
              color: 'var(--text)',
            }}
            onFocus={() => results.length > 0 && setOpen(true)}
          />
          {loading && (
            <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-500">
              …
            </span>
          )}

          {open && results.length > 0 && (
            <div
              className="absolute z-50 w-full mt-1 rounded-lg overflow-hidden shadow-xl"
              style={{ background: 'var(--surface)', border: '1px solid var(--border)' }}
            >
              {results.map((occ) => (
                <button
                  key={occ.code}
                  onClick={() => select(occ)}
                  className="w-full text-left px-3 py-2.5 hover:bg-white/5 border-b border-white/5 last:border-0 flex flex-col gap-0.5"
                >
                  <div className="flex items-center gap-2">
                    <span className="mono text-xs text-indigo-400">{occ.code}</span>
                    {occ.job_zone && (
                      <span className="text-xs text-slate-500">Zone {occ.job_zone}</span>
                    )}
                  </div>
                  <span className="text-sm font-medium">{occ.title}</span>
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
