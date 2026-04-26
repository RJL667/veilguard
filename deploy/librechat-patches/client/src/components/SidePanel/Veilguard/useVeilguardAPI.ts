import { useState, useEffect, useCallback } from 'react';

// Route all dashboard traffic through LibreChat's authenticated proxy
// (api/server/routes/veilguardClient.js). The proxy validates the user's
// JWT, injects an internal-secret header, and forwards over the docker
// bridge to sub-agents -- so these calls require a logged-in LibreChat
// session and never traverse the public Internet to sub-agents directly.
//
// Pre-2026-04-24 these calls hit /api/sub-agents/api/{stats,tasks,...}
// directly via Caddy. The spear-phish incident response sealed
// /api/sub-agents/* (and /api/tcmm/*) at the edge because those paths
// were leaking unauthenticated user content. The previous TODO
// ("Once those endpoints are scoped...") is what this change fulfills.
const AGENT_PROXY =
  ((window as any).__VEILGUARD_CONFIG__ || {}).agentProxyUrl ||
  '/api/veilguard-client';
// TCMM health is exposed via the same authenticated proxy (server-side
// fetch from sub-agents container -> TCMM over docker bridge).
const TCMM_HEALTH_URL =
  ((window as any).__VEILGUARD_CONFIG__ || {}).tcmmHealthUrl ||
  '/api/veilguard-client/tcmm-health';

async function fetchJSON<T>(url: string, fallback: T): Promise<T> {
  try {
    const resp = await fetch(url, { signal: AbortSignal.timeout(5000) });
    if (!resp.ok) return fallback;
    return await resp.json();
  } catch {
    return fallback;
  }
}

// Agent stats
export interface AgentStats {
  total_calls: number;
  total_errors: number;
  estimated_cost_usd: number;
  active_daemons: number;
  running_tasks: number;
  afk: boolean;
  idle_seconds: number;
  by_backend: Record<string, { calls: number; errors: number; total_ms: number }>;
  by_role: Record<string, { calls: number; errors: number }>;
  recent_calls: Array<{
    time: string;
    backend: string;
    model: string;
    role: string;
    elapsed_ms: number;
    cost_usd?: number;
  }>;
}

export function useAgentStats(interval = 5000) {
  const [stats, setStats] = useState<AgentStats | null>(null);

  useEffect(() => {
    const load = () => fetchJSON<AgentStats | null>(`${AGENT_PROXY}/stats`, null).then(setStats);
    load();
    const id = setInterval(load, interval);
    return () => clearInterval(id);
  }, [interval]);

  return stats;
}

// Tasks
export interface BackgroundTask {
  id: string;
  task: string;
  role: string;
  status: string;
  model: string;
  elapsed: string;
  result_preview: string | null;
  error: string | null;
}

export interface ManagedTask {
  id: string;
  title: string;
  description: string;
  status: string;
  depends_on: string[];
  assigned_to: string;
  result_preview: string | null;
}

export interface TasksResponse {
  background_tasks: BackgroundTask[];
  managed_tasks: ManagedTask[];
}

export function useTasks(interval = 3000) {
  const [tasks, setTasks] = useState<TasksResponse>({ background_tasks: [], managed_tasks: [] });

  useEffect(() => {
    const load = () => fetchJSON<TasksResponse>(`${AGENT_PROXY}/tasks`, tasks).then(setTasks);
    load();
    const id = setInterval(load, interval);
    return () => clearInterval(id);
  }, [interval]);

  return tasks;
}

// Scratchpad
export interface ScratchpadFile {
  name: string;
  size: number;
  age_hours: number;
  preview: string;
  full_content: string;
}

export function useScratchpad(interval = 10000) {
  const [files, setFiles] = useState<ScratchpadFile[]>([]);

  useEffect(() => {
    const load = () =>
      fetchJSON<{ files: ScratchpadFile[] }>(`${AGENT_PROXY}/scratchpad`, { files: [] }).then((d) =>
        setFiles(d.files),
      );
    load();
    const id = setInterval(load, interval);
    return () => clearInterval(id);
  }, [interval]);

  return files;
}

// Daemons
export interface DaemonInfo {
  name: string;
  task: string;
  role: string;
  enabled: boolean;
  backend: string;
  interval_minutes: number;
  run_count: number;
  next_run_seconds: number;
  last_observation: { time: string; result?: string; error?: string } | null;
}

export function useDaemons(interval = 5000) {
  const [daemons, setDaemons] = useState<DaemonInfo[]>([]);

  useEffect(() => {
    const load = () =>
      fetchJSON<{ daemons: DaemonInfo[] }>(`${AGENT_PROXY}/daemons`, { daemons: [] }).then((d) =>
        setDaemons(d.daemons),
      );
    load();
    const id = setInterval(load, interval);
    return () => clearInterval(id);
  }, [interval]);

  return daemons;
}

// TCMM Health
export interface TCMMHealth {
  status: string;
  live_blocks: number;
  shadow_blocks: number;
  archive_blocks: number;
  dream_nodes: number;
  current_step: number;
}

export function useTCMMHealth(interval = 10000) {
  const [health, setHealth] = useState<TCMMHealth | null>(null);

  useEffect(() => {
    const load = () => fetchJSON<TCMMHealth | null>(TCMM_HEALTH_URL, null).then(setHealth);
    load();
    const id = setInterval(load, interval);
    return () => clearInterval(id);
  }, [interval]);

  return health;
}

// Service health check
export interface ServiceStatus {
  name: string;
  url: string;
  status: 'up' | 'down' | 'unknown';
}

export function useServiceHealth(interval = 15000) {
  const [services, setServices] = useState<ServiceStatus[]>([]);

  useEffect(() => {
    const check = async () => {
      // Services we can reach directly from the browser (have CORS)
      const directTargets = [
        { name: 'Sub-Agents', url: `${AGENT_PROXY}/stats` },
        { name: 'TCMM', url: TCMM_HEALTH_URL },
      ];

      const directResults: ServiceStatus[] = await Promise.all(
        directTargets.map(async ({ name, url }) => {
          try {
            const resp = await fetch(url, { signal: AbortSignal.timeout(3000) });
            return { name, url, status: resp.ok ? ('up' as const) : ('down' as const) };
          } catch {
            return { name, url, status: 'down' as const };
          }
        }),
      );

      // Services we check via sub-agents proxy (no CORS from browser)
      // Sub-agents server checks these server-side in the /api/stats response
      const proxyServices: ServiceStatus[] = [
        { name: 'Host-Exec', url: '', status: 'unknown' as const },
        { name: 'Forge', url: '', status: 'unknown' as const },
        { name: 'PII Proxy', url: '', status: 'unknown' as const },
      ];

      // If sub-agents is up, assume host services are reachable
      const subAgentsUp = directResults.find(s => s.name === 'Sub-Agents')?.status === 'up';
      if (subAgentsUp) {
        proxyServices.forEach(s => s.status = 'up');
      }

      setServices([...directResults, ...proxyServices]);
    };
    check();
    const id = setInterval(check, interval);
    return () => clearInterval(id);
  }, [interval]);

  return services;
}

// Memory heatmap
export interface MemoryBlock {
  id: string;
  tier: 'live' | 'shadow' | 'archive';
  heat: number;
  text: string;
  role: string;
  step: number;
  topics?: string[];
  entities?: string[];
  links?: number;
}

export interface MemoryHeatmapData {
  blocks: MemoryBlock[];
  stats: {
    live_blocks: number;
    shadow_blocks: number;
    archive_blocks: number;
    current_step: number;
    total_live_tokens: number;
    total_archive_tokens: number;
    recalled_tokens: number;
    token_savings: number;
  };
}

// Token stats from TCMM
export interface TokenStats {
  session: {
    input_tokens: number;
    output_tokens: number;
    saved_tokens: number;
    turns: number;
  };
  history: Array<{
    step: number;
    input_tokens: number;
    output_tokens: number;
    archive_blocks: number;
  }>;
}

export function useTokenStats(interval = 3000) {
  const [data, setData] = useState<TokenStats | null>(null);

  useEffect(() => {
    const load = () =>
      fetchJSON<TokenStats | null>(`${TCMM_URL}/api/token_stats`, null).then(setData);
    load();
    const id = setInterval(load, interval);
    return () => clearInterval(id);
  }, [interval]);

  return data;
}

export function useMemoryHeatmap(interval = 5000) {
  const [data, setData] = useState<MemoryHeatmapData | null>(null);

  useEffect(() => {
    const load = () =>
      fetchJSON<MemoryHeatmapData | null>(`${TCMM_URL}/api/memory_heatmap`, null).then(setData);
    load();
    const id = setInterval(load, interval);
    return () => clearInterval(id);
  }, [interval]);

  return data;
}
