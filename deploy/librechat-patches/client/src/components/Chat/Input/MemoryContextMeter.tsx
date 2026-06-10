import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Flame, BrainCircuit } from 'lucide-react';
import { Constants } from 'librechat-data-provider';
import { useAuthContext } from '~/hooks/AuthContext';

/**
 * MemoryContextMeter — Claude-Code-style context meter for TCMM memory,
 * mounted in the chat-input footer row.
 *
 * Collapsed: a small chip showing total memory usage as a % of the model
 * context window. Expanded (click): a popover with one progress bar per
 * memory tier — immutable / stable / working / volatile — plus effective
 * heat for the stable and working tiers.
 *
 * Data: GET /api/veilguard-client/memory-status?conversationId=…
 * (JWT-authenticated LibreChat proxy → TCMM /memory_status, which is
 * peek-only: no instance creation, no recall, no DB work — safe to poll.)
 */

const BASE =
  ((window as any).__VEILGUARD_CONFIG__ || {}).agentProxyUrl ||
  '/api/veilguard-client';

interface TierStat {
  tokens: number;
  blocks: number;
  heat_avg?: number;
  heat_max?: number;
  max_blocks?: number;
}

interface MemoryStatus {
  status: string;
  cold: boolean;
  context_window?: number;
  budget_tokens?: number;
  total_tokens: number;
  tiers: Record<string, TierStat>;
}

function formatK(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

const EMPTY: TierStat = { tokens: 0, blocks: 0 };

function useMemoryStatus(conversationId: string | null, isSubmitting: boolean, interval = 12000) {
  const [data, setData] = useState<MemoryStatus | null>(null);
  const { token } = useAuthContext();
  const convRef = useRef(conversationId);
  convRef.current = conversationId;

  const load = useCallback(() => {
    const cid = convRef.current;
    if (!token || !cid || cid === Constants.NEW_CONVO) {
      setData(null);
      return;
    }
    fetch(`${BASE}/memory-status?conversationId=${encodeURIComponent(cid)}`, {
      headers: { Authorization: `Bearer ${token}` },
      signal: AbortSignal.timeout(5000),
    })
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        // Ignore responses that raced a conversation switch.
        if (convRef.current === cid) setData(d);
      })
      .catch(() => undefined);
  }, [token]);

  // Poll + refetch on conversation switch.
  useEffect(() => {
    setData(null);
    load();
    const id = setInterval(load, interval);
    return () => clearInterval(id);
  }, [conversationId, interval, load]);

  // The turn just finished: TCMM ingests in the background — refresh
  // shortly after so the meter reflects the new turn without waiting a
  // full poll interval.
  const prevSubmitting = useRef(isSubmitting);
  useEffect(() => {
    if (prevSubmitting.current && !isSubmitting) {
      const t = setTimeout(load, 4000);
      return () => clearTimeout(t);
    }
    prevSubmitting.current = isSubmitting;
  }, [isSubmitting, load]);
  useEffect(() => {
    prevSubmitting.current = isSubmitting;
  }, [isSubmitting]);

  return data;
}

function HeatBadge({ tier }: { tier: TierStat }) {
  if (tier.heat_avg == null) {
    return null;
  }
  const hot = tier.heat_avg >= 1.0;
  return (
    <span
      className={`flex items-center gap-0.5 tabular-nums ${hot ? 'text-orange-400' : 'text-text-tertiary'}`}
      title={`effective heat — avg ${tier.heat_avg}, max ${tier.heat_max ?? '–'}`}
    >
      <Flame size={9} />
      {tier.heat_avg.toFixed(1)}
    </span>
  );
}

function TierRow({
  label,
  tier,
  max,
  color,
}: {
  label: string;
  tier: TierStat;
  max: number;
  color: string;
}) {
  const pct = max > 0 ? Math.min(100, (tier.tokens / max) * 100) : 0;
  return (
    <div className="flex items-center gap-2 text-[10px]">
      <span className="w-16 shrink-0 text-right text-text-secondary">{label}</span>
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-tertiary">
        <div
          className={`h-full rounded-full transition-all duration-700 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="w-10 shrink-0 text-right tabular-nums text-text-secondary">
        {formatK(tier.tokens)}
      </span>
      <span className="w-9 shrink-0">
        <HeatBadge tier={tier} />
      </span>
    </div>
  );
}

export default function MemoryContextMeter({
  conversationId,
  isSubmitting,
}: {
  conversationId: string | null;
  isSubmitting: boolean;
}) {
  const [open, setOpen] = useState(false);
  const data = useMemoryStatus(conversationId, isSubmitting);
  const rootRef = useRef<HTMLDivElement>(null);

  // Close on outside click.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener('mousedown', onDown);
    return () => document.removeEventListener('mousedown', onDown);
  }, [open]);

  if (!data || data.cold || !data.total_tokens) {
    return null;
  }

  const window_ = data.context_window || 200000;
  const tiers = data.tiers || {};
  const immutable = tiers.immutable ?? EMPTY;
  const stable = tiers.stable ?? EMPTY;
  // Working folds in transient (fresh tool output awaiting citation).
  const transient = tiers.transient ?? EMPTY;
  const working: TierStat = {
    ...(tiers.working ?? EMPTY),
    tokens: (tiers.working?.tokens ?? 0) + transient.tokens,
    blocks: (tiers.working?.blocks ?? 0) + transient.blocks,
  };
  // Volatile folds in shadow (both are recalled-memory display tiers).
  const shadow = tiers.shadow ?? EMPTY;
  const volatile_: TierStat = {
    tokens: (tiers.volatile?.tokens ?? 0) + shadow.tokens,
    blocks: (tiers.volatile?.blocks ?? 0) + shadow.blocks,
  };

  const total = data.total_tokens;
  const pct = Math.min(100, (total / window_) * 100);
  const pctLabel = pct < 1 ? '<1' : Math.round(pct);

  const segments: Array<[TierStat, string]> = [
    [immutable, 'bg-cyan-500'],
    [stable, 'bg-blue-500'],
    [working, 'bg-amber-500'],
    [volatile_, 'bg-purple-500'],
  ];

  return (
    <div ref={rootRef} className="relative flex items-center">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-label="Memory context"
        title={`Memory context — ${formatK(total)} / ${formatK(window_)} tokens`}
        className="flex items-center gap-1.5 rounded-full border border-border-light px-2 py-1 text-[10px] text-text-secondary transition-colors hover:bg-surface-tertiary"
      >
        <BrainCircuit size={12} />
        {/* mini stacked bar */}
        <span className="flex h-1.5 w-12 overflow-hidden rounded-full bg-surface-tertiary">
          {segments.map(([t, color], i) => (
            <span
              key={i}
              className={`h-full ${color}`}
              style={{ width: `${Math.min(100, (t.tokens / window_) * 100)}%` }}
            />
          ))}
        </span>
        <span className="tabular-nums">{pctLabel}%</span>
      </button>

      {open && (
        <div className="absolute bottom-full left-0 z-50 mb-2 w-80 rounded-xl border border-border-medium bg-surface-primary p-3 shadow-lg">
          <div className="mb-2 flex items-baseline justify-between">
            <span className="text-xs font-semibold text-text-primary">Memory context</span>
            <span className="text-[10px] tabular-nums text-text-secondary">
              {formatK(total)} / {formatK(window_)} ({pctLabel}%)
            </span>
          </div>
          {/* overall stacked bar */}
          <div className="mb-3 flex h-2 w-full overflow-hidden rounded-full bg-surface-tertiary">
            {segments.map(([t, color], i) => (
              <div
                key={i}
                className={`h-full ${color}`}
                style={{ width: `${Math.min(100, (t.tokens / window_) * 100)}%` }}
              />
            ))}
          </div>
          <div className="space-y-1.5">
            <TierRow label="Immutable" tier={immutable} max={window_} color="bg-cyan-500" />
            <TierRow label="Stable" tier={stable} max={window_} color="bg-blue-500" />
            <TierRow label="Working" tier={working} max={window_} color="bg-amber-500" />
            <TierRow label="Volatile" tier={volatile_} max={window_} color="bg-purple-500" />
          </div>
          <div className="mt-2 border-t border-border-light pt-1.5 text-[9px] text-text-tertiary">
            {immutable.blocks + stable.blocks + working.blocks} live blocks ·{' '}
            {volatile_.blocks} recalled · <Flame size={8} className="inline" /> = effective heat
            (stable/working)
          </div>
        </div>
      )}
    </div>
  );
}
