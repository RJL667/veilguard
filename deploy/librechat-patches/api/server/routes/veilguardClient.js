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

module.exports = router;
