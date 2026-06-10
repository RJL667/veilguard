/**
 * Authenticated proxy to sub-agents' /api/client/* endpoints.
 *
 * The sub-agents server trusts the ``x-user-id`` header to decide which
 * LibreChat user is asking.  Inside the stack that's safe because the
 * header is injected server-side by the LibreChat MCP layer
 * (``{{LIBRECHAT_USER_ID}}`` template), but the Cowork side panel runs
 * in the browser and can't be trusted to set that header — a malicious
 * user could spoof any value to impersonate another user and steal
 * their daemon token.
 *
 * This route solves that by requiring a valid JWT (``requireJwtAuth``),
 * reading ``req.user.id`` from the decoded session, and forwarding the
 * request to sub-agents with ``x-user-id`` set from the server-side
 * session — the browser never sets it directly.
 */
const axios = require('axios');
const express = require('express');
const { requireJwtAuth } = require('../middleware/');

const router = express.Router();

// EventSource cannot set custom headers, so the SSE client passes the
// JWT as ?token=... query param.  This middleware promotes it to the
// Authorization header BEFORE requireJwtAuth runs, so passport-jwt sees
// it on the standard header it already inspects.  Scoped to /events only
// — every other route still uses Authorization headers from the React
// client.
router.use('/events', (req, res, next) => {
  if (!req.headers.authorization && req.query && req.query.token) {
    req.headers.authorization = `Bearer ${req.query.token}`;
  }
  next();
});

// Default points at host.docker.internal so the LibreChat container
// can reach sub-agents on the host, identical to the MCP URL in
// librechat.yaml.  Override with VEILGUARD_SUB_AGENTS_URL for different
// topologies (e.g. running LibreChat bare-metal).
const SUB_AGENTS_URL =
  process.env.VEILGUARD_SUB_AGENTS_URL || 'http://host.docker.internal:8809';

// Shared secret with sub-agents for admin routes (/api/client/status,
// /install, /register, /folders, /browse). Added 2026-04-24 after the
// register endpoint was found vending tokens unauthenticated via the
// contextvar fallback. Caddy already blocks those paths from the public
// domain; this header is defense-in-depth for docker-bridge callers.
// If unset we still send the call, but sub-agents returns 503 until the
// operator configures the env var on both sides — intentional fail-closed.
const INTERNAL_SECRET = process.env.VEILGUARD_INTERNAL_SECRET || '';

function proxy(path, method = 'GET') {
  return async (req, res) => {
    const userId = req.user && req.user.id ? String(req.user.id) : '';
    if (!userId) {
      return res.status(401).json({ error: 'Not authenticated' });
    }

    try {
      // Forward the original public Host so sub-agents can construct
      // a daemon-facing ws_url that actually resolves from the user's
      // laptop (e.g. veilguard.phishield.com), not host.docker.internal.
      const publicHost = req.get('x-forwarded-host') || req.get('host') || '';
      const result = await axios({
        method,
        url: `${SUB_AGENTS_URL}${path}`,
        headers: {
          'x-user-id': userId,
          'Content-Type': 'application/json',
          ...(publicHost ? { 'x-forwarded-host': publicHost } : {}),
          // Authenticate to sub-agents as a trusted internal caller.
          // Empty string is harmless (sub-agents compare_digest fails
          // closed), but logs a warn on startup if the var is missing.
          'x-internal-secret': INTERNAL_SECRET,
        },
        params: req.query || {},
        data: method === 'POST' ? req.body || {} : undefined,
        timeout: 8000,
        validateStatus: () => true, // forward non-2xx from upstream as-is
      });
      res.status(result.status).json(result.data);
    } catch (e) {
      res.status(502).json({
        error: 'sub-agents unreachable',
        detail: e.message,
      });
    }
  };
}

router.get('/status', requireJwtAuth, proxy('/api/client/status', 'GET'));
router.get('/install', requireJwtAuth, proxy('/api/client/install', 'GET'));
router.get('/register', requireJwtAuth, proxy('/api/client/register', 'GET'));
router.get('/folders', requireJwtAuth, proxy('/api/client/folders', 'GET'));
router.post('/folders', requireJwtAuth, proxy('/api/client/folders', 'POST'));
router.get('/browse', requireJwtAuth, proxy('/api/client/browse', 'GET'));

// --- TCMM /health proxy (added 2026-04-24) ---
// The Agent Platform's "TCMM Health" indicator polls block-count
// statistics. Caddy's /api/tcmm/* blanket-block keeps the heatmap and
// debug routes private; we expose just /health here, fetched by
// LibreChat over the docker bridge so TCMM never appears on the public
// domain. requireJwtAuth means a logged-in user is still required even
// for this status check.
const TCMM_URL =
  process.env.VEILGUARD_TCMM_URL || 'http://host.docker.internal:8811';

router.get('/tcmm-health', requireJwtAuth, async (req, res) => {
  try {
    const result = await axios({
      method: 'GET',
      url: `${TCMM_URL}/health`,
      timeout: 5000,
      validateStatus: () => true,
    });
    res.status(result.status).json(result.data);
  } catch (e) {
    res.status(502).json({ error: 'tcmm unreachable', detail: e.message });
  }
});

// --- TCMM per-conversation memory meter (added 2026-06-11) ---
// Feeds the context meter next to the chat input: per-tier token counts
// (immutable / stable / working / volatile) + heat. user_id comes from
// the JWT session server-side (same anti-spoofing rationale as the
// proxy() helper above); the browser only supplies the conversationId.
// TCMM's /memory_status is peek-only (no instance creation, no recall),
// so polling it is cheap and side-effect-free.
router.get('/memory-status', requireJwtAuth, async (req, res) => {
  const userId = req.user && req.user.id ? String(req.user.id) : '';
  if (!userId) {
    return res.status(401).json({ error: 'Not authenticated' });
  }
  try {
    const result = await axios({
      method: 'GET',
      url: `${TCMM_URL}/memory_status`,
      params: {
        conversation_id: String(req.query.conversationId || ''),
        user_id: userId,
      },
      timeout: 5000,
      validateStatus: () => true,
    });
    res.status(result.status).json(result.data);
  } catch (e) {
    res.status(502).json({ error: 'tcmm unreachable', detail: e.message });
  }
});

// --- Agent Platform side-panel routes (added 2026-04-24) ---
// The Veilguard Agent Platform panel (Dashboard / Tasks / Scratchpad /
// Daemons tabs) used to fetch /api/sub-agents/api/{stats,tasks,...}
// directly from the browser. Caddy's incident-response lockdown blocks
// the /api/sub-agents/* path for everything except the public update
// manifest, so those direct calls now 404 ("Connecting to sub-agents
// server...").
//
// These endpoints leak user-typed content (task descriptions, scratchpad
// files up to 5000 chars, daemon task strings), so the proper fix is to
// route them through the authenticated proxy used by the Cowork panel.
// requireJwtAuth ensures only logged-in LibreChat users hit them; the
// 'x-internal-secret' header injected by proxy() above lets sub-agents
// distinguish a trusted internal caller from a leaked-Caddy-config bypass.
//
// These routes are NOT user-scoped on the sub-agents side yet -- every
// authenticated LibreChat user sees the full system view. Per-user
// scoping is a separate piece of work.
router.get('/stats', requireJwtAuth, proxy('/api/stats', 'GET'));
router.get('/tasks', requireJwtAuth, proxy('/api/tasks', 'GET'));
router.get('/scratchpad', requireJwtAuth, proxy('/api/scratchpad', 'GET'));
router.get('/daemons', requireJwtAuth, proxy('/api/daemons', 'GET'));
router.get('/teams', requireJwtAuth, proxy('/api/teams', 'GET'));

// --- Phase 3/4 agent-runtime proxy (added 2026-05-27) ---
// agent-runtime (port 5000) owns the multi-agent platform's Phase 3/4
// surface: proposals, work_items, lessons, replay, documents.  The
// sidebar's "Work Queue" tab (ProposalsSection.tsx) fetches via this
// proxy so the X-Internal-Secret never reaches the browser.
const AGENT_RUNTIME_URL =
  process.env.VEILGUARD_AGENT_RUNTIME_URL ||
  'http://host.docker.internal:5000';

function proxyRuntime(path, method = 'GET') {
  return async function (req, res) {
    try {
      // Inject tenant_id + user_id from the JWT-authenticated user
      // when the path expects them (every Phase 3/4 endpoint does).
      const uid = req.user && req.user.id;
      const params = { ...req.query };
      if (uid) {
        if (!params.tenant_id) params.tenant_id = uid;
        if (!params.user_id) params.user_id = uid;
      }
      const result = await axios({
        method,
        url: `${AGENT_RUNTIME_URL}${path}`,
        params,
        data: ['POST', 'PUT', 'PATCH'].includes(method) ? req.body : undefined,
        headers: {
          'X-Internal-Secret': INTERNAL_SECRET || '',
          'Content-Type': 'application/json',
        },
        // 30s.  Was 8s — too tight when agent-runtime is mid-stream
        // (busy event loop starves the sync Lance reads inside
        // /proposals/{id} and /tasks/{id} for >8s, axios ECONNABORTEDs,
        // sidebar showed opaque "HTTP 502" 2026-05-27).  30s is still
        // bounded so a genuinely hung runtime fails cleanly.
        timeout: 30000,
        validateStatus: () => true,
      });
      res.status(result.status).json(result.data);
    } catch (e) {
      // Surface the real failure mode — was a recurring opaque "HTTP 502"
      // in the sidebar.  Now we get the axios error code (ECONNREFUSED /
      // ETIMEDOUT / etc.) + stack so the next regression is debuggable.
      const detail = {
        message: (e && e.message) || String(e),
        code:    e && e.code,
        url:     `${AGENT_RUNTIME_URL}${path}`,
      };
      console.error('[veilguard-client] proxyRuntime failure', detail);
      res.status(502).json({
        error:  'agent-runtime unreachable',
        detail,
      });
    }
  };
}

// Phase 3 — proposals queue + emergency lane
router.get('/proposals', requireJwtAuth, proxyRuntime('/proposals', 'GET'));
router.post('/proposals/emit', requireJwtAuth, proxyRuntime('/proposals/emit', 'POST'));
router.post('/proposals/:id/decision', requireJwtAuth, function (req, res) {
  return proxyRuntime(`/proposals/${req.params.id}/decision`, 'POST')(req, res);
});
router.post('/proposals/:id/convert', requireJwtAuth, function (req, res) {
  return proxyRuntime(`/proposals/${req.params.id}/convert`, 'POST')(req, res);
});

// Phase 3/4 — unified work-items view
router.get('/work_items', requireJwtAuth, proxyRuntime('/work_items', 'GET'));

// Phase 3 — per-tenant proactive-stream config
router.get('/proactive_config', requireJwtAuth, proxyRuntime('/proactive_config', 'GET'));
router.post('/proactive_config', requireJwtAuth, proxyRuntime('/proactive_config', 'POST'));

// Phase 3/4 — lessons (review queue + amendment candidates)
router.get('/lessons/review_queue', requireJwtAuth, proxyRuntime('/lessons/review_queue', 'GET'));
router.get('/lessons/amendment_candidates', requireJwtAuth, proxyRuntime('/lessons/amendment_candidates', 'GET'));
router.post('/lessons/:id/decision', requireJwtAuth, function (req, res) {
  return proxyRuntime(`/lessons/${req.params.id}/decision`, 'POST')(req, res);
});

// Phase 3/4 — proposal + task detail drill-in.
// IMPORTANT: these :id paths must register BEFORE the :id/ledger and
// :id/progress paths or Express's first-match routing serves the :id/...
// handlers for everything.
router.get('/proposals/:id', requireJwtAuth, function (req, res) {
  return proxyRuntime(`/proposals/${req.params.id}`, 'GET')(req, res);
});
router.get('/tasks/:id', requireJwtAuth, function (req, res) {
  return proxyRuntime(`/tasks/${req.params.id}`, 'GET')(req, res);
});

// Work Queue — read a task's deliverable artifact so outputs are clickable.
// Sandboxed server-side (workspace_fs._resolve). ?path=team/drafts/<file>.md
router.get('/workspace/file', requireJwtAuth, proxyRuntime('/workspace/file', 'GET'));

// Phase 3/4 — SSE push.  Stream-through proxy: we cannot use the
// regular axios JSON proxy because that buffers the full response.
// Instead pipe the upstream response straight to the browser, leaving
// the SSE message stream intact.  Heartbeat handled upstream in
// agent-runtime's event_stream() — every 25s.
router.get('/events', requireJwtAuth, async function (req, res) {
  try {
    const uid = req.user && req.user.id;
    const params = { ...req.query };
    if (uid) {
      if (!params.tenant_id) params.tenant_id = uid;
      if (!params.user_id)   params.user_id   = uid;
    }
    const qs = new URLSearchParams(params).toString();
    const upstream = await axios({
      method:       'GET',
      url:          `${AGENT_RUNTIME_URL}/events?${qs}`,
      headers:      {
        'X-Internal-Secret': INTERNAL_SECRET || '',
        Accept:              'text/event-stream',
      },
      responseType: 'stream',
      timeout:      0,             // no timeout — SSE is long-lived
      validateStatus: () => true,
    });
    // Pass status + headers straight through (minus content-length which
    // would lie about the stream length).
    res.status(upstream.status);
    res.set('Content-Type',      'text/event-stream');
    res.set('Cache-Control',     'no-cache, no-transform');
    res.set('X-Accel-Buffering', 'no');
    res.set('Connection',        'keep-alive');
    upstream.data.pipe(res);
    req.on('close', () => {
      try { upstream.data.destroy(); } catch (e) {}
    });
  } catch (e) {
    res.status(502).json({
      error:  'agent-runtime SSE unreachable',
      detail: (e && e.message) || String(e),
    });
  }
});

// Phase 4 — task ledgers + replay
router.get('/tasks/:id/ledger', requireJwtAuth, function (req, res) {
  return proxyRuntime(`/tasks/${req.params.id}/ledger`, 'GET')(req, res);
});
router.post('/tasks/:id/ledger', requireJwtAuth, function (req, res) {
  return proxyRuntime(`/tasks/${req.params.id}/ledger`, 'POST')(req, res);
});
router.post('/tasks/:id/progress', requireJwtAuth, function (req, res) {
  return proxyRuntime(`/tasks/${req.params.id}/progress`, 'POST')(req, res);
});
router.get('/replay/:id', requireJwtAuth, function (req, res) {
  return proxyRuntime(`/replay/${req.params.id}`, 'GET')(req, res);
});

// Phase 1 — sidebar agents view (lives on sub-agents :8809 not agent-runtime)
router.get('/agents', requireJwtAuth, proxy('/api/agents', 'GET'));

module.exports = router;
