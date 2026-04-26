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

module.exports = router;
