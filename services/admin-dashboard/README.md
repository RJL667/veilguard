# Veilguard Admin Dashboard

Single-page admin console for Veilguard. Read-only view of:

- Service health (TCMM, Lance, FTS index, NLP adapter, worker threads)
- Recall performance counters (queues, NLP batches, success/skip)
- Lance tables (rows, fragments, on-disk size, FTS index status)
- PII redaction stats (last 24h totals, by-type breakdown, hourly time series)
- Recent redaction log (counts only — content never displayed)
- Per-user usage (messages, tokens, redaction count, last active)

## Auth

Validates the LibreChat JWT cookie (`token`) and requires
`role == "ADMIN"` in the LibreChat MongoDB `users` collection.

To grant admin to another user:

```js
db.users.updateOne({email: "user@example.com"}, {$set: {role: "ADMIN"}})
```

## Running locally

```bash
export JWT_SECRET=...   # same value as LibreChat's .env
export MONGO_URI=mongodb://localhost:27017/LibreChat
export TCMM_URL=http://localhost:8811
export ADMIN_LANCE_DIR=/path/to/tcmm-data/veilguard/tcmm.db
python server.py
```

Then sign into LibreChat in the same browser, and visit
`http://localhost:8820/`.

## Deployed (prod VM)

systemd unit at `veilguard-admin.service` listens on `127.0.0.1:8820`.
Caddy reverse-proxies `https://veilguard.phishield.com/admin/*` → it.

```
sudo cp veilguard-admin.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now veilguard-admin.service
```
