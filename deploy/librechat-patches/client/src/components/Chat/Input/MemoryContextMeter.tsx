import React, { useState, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
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
 * The popover renders through a PORTAL to document.body with fixed
 * positioning: the chat-input container clips overflowing children
 * (rounded corners + overflow-hidden ancestors), which cut the popover
 * to its bottom slice — only the Volatile row was visible in-place.
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
  // Exact provider-billed input tokens for the last turn (new +
  // cache_read + cache_create) — the same figure the admin dashboard
  // shows. Preferred as the headline; tier bars stay estimates.
  last_turn_input_tokens?: number;
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
      className={`flex items-center justify-end gap-0.5 tabular-nums ${hot ? 'text-orange-400' : 'text-text-tertiary'}`}
      title={`effective heat — avg ${tier.heat_avg}, max ${tier.heat_max ?? '–'}`}
    >
      <Flame size={10} className={hot ? 'drop-shadow-[0_0_3px_rgba(251,146,60,0.6)]' : ''} />
      {tier.heat_avg.toFixed(1)}
    </span>
  );
}

interface TierSpec {
  key: string;
  label: string;
  hint: string;
  dot: string;
  bar: string;
}

// One visual identity per tier, used by the chip, the overview bar and
// the per-tier rows so colors always correspond.
const TIER_SPECS: TierSpec[] = [
  {
    key: 'immutable',
    label: 'Immutable',
    hint: 'pinned preamble, persona & tool definitions — 24h cache tier',
    dot: 'bg-cyan-400',
    bar: 'bg-gradient-to-r from-cyan-400 to-cyan-600',
  },
  {
    key: 'stable',
    label: 'Stable',
    hint: 'promoted, cache-stable conversation memory',
    dot: 'bg-blue-400',
    bar: 'bg-gradient-to-r from-blue-400 to-blue-600',
  },
  {
    key: 'working',
    label: 'Working',
    hint: 'recent turns & fresh tool output (incl. transient)',
    dot: 'bg-amber-400',
    bar: 'bg-gradient-to-r from-amber-400 to-amber-600',
  },
  {
    key: 'volatile',
    label: 'Volatile',
    hint: 'recalled memory staged for this turn (incl. shadow)',
    dot: 'bg-purple-400',
    bar: 'bg-gradient-to-r from-purple-400 to-purple-600',
  },
];

function TierRow({
  spec,
  tier,
  max,
}: {
  spec: TierSpec;
  tier: TierStat;
  max: number;
}) {
  const pct = max > 0 ? Math.min(100, (tier.tokens / max) * 100) : 0;
  // Tiny-but-nonzero tiers still get a visible sliver.
  const width = tier.tokens > 0 ? Math.max(2, pct) : 0;
  return (
    <div className="flex items-center gap-2.5 text-[11px]" title={spec.hint}>
      <span className={`h-2 w-2 shrink-0 rounded-full ${spec.dot}`} />
      <span className="w-[4.4rem] shrink-0 text-text-secondary">{spec.label}</span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-tertiary">
        <div
          className={`h-full rounded-full transition-all duration-700 ${spec.bar}`}
          style={{ width: `${width}%` }}
        />
      </div>
      <span className="w-11 shrink-0 text-right tabular-nums text-text-primary">
        {formatK(tier.tokens)}
      </span>
      <span className="w-7 shrink-0 text-right tabular-nums text-text-tertiary">
        {tier.blocks}
      </span>
      <span className="w-10 shrink-0">
        <HeatBadge tier={tier} />
      </span>
    </div>
  );
}

const POPOVER_WIDTH = 352;

export default function MemoryContextMeter({
  conversationId,
  isSubmitting,
}: {
  conversationId: string | null;
  isSubmitting: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ left: number; bottom: number } | null>(null);
  const data = useMemoryStatus(conversationId, isSubmitting);
  const btnRef = useRef<HTMLButtonElement>(null);
  const popRef = useRef<HTMLDivElement>(null);

  const toggle = useCallback(() => {
    if (btnRef.current) {
      const r = btnRef.current.getBoundingClientRect();
      setPos({
        left: Math.max(8, Math.min(r.left, window.innerWidth - POPOVER_WIDTH - 8)),
        bottom: window.innerHeight - r.top + 8,
      });
    }
    setOpen((o) => !o);
  }, []);

  // Close on outside click / Escape. The popover lives in a portal, so
  // check both the button and the portal node.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (btnRef.current?.contains(t) || popRef.current?.contains(t)) {
        return;
      }
      setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  // Conversation switched: stale popover would show the old session.
  useEffect(() => {
    setOpen(false);
  }, [conversationId]);

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

  const byKey: Record<string, TierStat> = {
    immutable,
    stable,
    working,
    volatile: volatile_,
  };

  const total = data.last_turn_input_tokens || data.total_tokens;
  const pct = Math.min(100, (total / window_) * 100);
  const pctLabel = pct < 1 ? '<1' : Math.round(pct);

  return (
    <>
      <button
        ref={btnRef}
        type="button"
        onClick={toggle}
        aria-label="Memory context"
        title={`Memory context — ${formatK(total)} / ${formatK(window_)} tokens`}
        className="flex items-center gap-1.5 rounded-full border border-border-light px-2 py-1 text-[10px] text-text-secondary transition-colors hover:bg-surface-tertiary"
      >
        <BrainCircuit size={12} />
        {/* mini stacked bar */}
        <span className="flex h-1.5 w-12 overflow-hidden rounded-full bg-surface-tertiary">
          {TIER_SPECS.map((s) => (
            <span
              key={s.key}
              className={`h-full ${s.bar}`}
              style={{ width: `${Math.min(100, (byKey[s.key].tokens / window_) * 100)}%` }}
            />
          ))}
        </span>
        <span className="tabular-nums">{pctLabel}%</span>
      </button>

      {open &&
        pos &&
        createPortal(
          <div
            ref={popRef}
            style={{ position: 'fixed', left: pos.left, bottom: pos.bottom, width: POPOVER_WIDTH }}
            className="z-[1000] rounded-2xl border border-border-medium bg-surface-primary p-4 shadow-2xl"
          >
            <div className="mb-3 flex items-center justify-between">
              <span className="flex items-center gap-1.5 text-sm font-semibold text-text-primary">
                <BrainCircuit size={15} className="text-cyan-400" />
                Memory
              </span>
              <span className="text-[11px] tabular-nums text-text-secondary">
                {formatK(total)} / {formatK(window_)} · {pctLabel}%
              </span>
            </div>

            {/* overview stacked bar */}
            <div className="mb-3.5 flex h-2.5 w-full overflow-hidden rounded-full bg-surface-tertiary">
              {TIER_SPECS.map((s) => (
                <div
                  key={s.key}
                  className={`h-full ${s.bar} transition-all duration-700`}
                  style={{ width: `${Math.min(100, (byKey[s.key].tokens / window_) * 100)}%` }}
                />
              ))}
            </div>

            {/* per-tier rows — always all four, even at zero */}
            <div className="space-y-2">
              {TIER_SPECS.map((s) => (
                <TierRow key={s.key} spec={s} tier={byKey[s.key]} max={window_} />
              ))}
            </div>

            {/* column key + footer */}
            <div className="mt-3 flex items-center justify-between border-t border-border-light pt-2 text-[9px] text-text-tertiary">
              <span>
                {immutable.blocks + stable.blocks + working.blocks} live ·{' '}
                {volatile_.blocks} recalled
              </span>
              <span className="flex items-center gap-1">
                tokens · blocks · <Flame size={8} className="inline" /> heat
              </span>
            </div>
          </div>,
          document.body,
        )}
    </>
  );
}
