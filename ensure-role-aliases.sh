#!/bin/bash
# Veilguard — ensure lowercase role aliases exist in MongoDB.
#
# LibreChat's client-side data-provider lowercases role names when
# hitting `/api/roles/:roleName` (see packages/data-provider/src/
# api-endpoints.ts:364). The backend does a case-sensitive lookup,
# so it returns 404 for `/api/roles/admin` when the DB only has
# `ADMIN`. That makes every permission check fail on the client —
# useHasAccess returns false, the MCP selector badge is hidden, the
# tools dropdown is empty, and the Agents endpoint isn't rendered.
#
# Fix is a pair of lowercase role aliases that mirror the uppercase
# defaults. LibreChat's seed logic only touches ADMIN/USER on startup,
# so the aliases survive container restarts. Run this script after
# any fresh MongoDB init (new deploy, `docker compose down -v`, etc).
#
# Safe to re-run — uses upsert semantics.

set -euo pipefail

echo "[roles] ensuring lowercase aliases (admin, user) exist..."

sudo docker exec veilguard-mongodb-1 mongosh LibreChat --quiet --eval '
const admin = db.roles.findOne({name:"ADMIN"});
const user  = db.roles.findOne({name:"USER"});
if (!admin || !user) {
  print("[roles] ERROR: uppercase ADMIN/USER roles missing. Let LibreChat boot once to seed them.");
  quit(1);
}
const {_id: _aId, ...adminRest} = admin;
const {_id: _uId, ...userRest}  = user;
db.roles.updateOne({name:"admin"}, {$set: {...adminRest, name:"admin"}}, {upsert:true});
db.roles.updateOne({name:"user"},  {$set: {...userRest,  name:"user"}},  {upsert:true});
print("[roles] admin + user aliases OK");
print("");
db.roles.find({}, {name:1, "permissions.MCP_SERVERS.USE":1, "permissions.AGENTS.USE":1, _id:0})
  .forEach(r => print("  " + r.name + " -> MCP_SERVERS.USE=" + (r.permissions?.MCP_SERVERS?.USE ?? "undef") + ", AGENTS.USE=" + (r.permissions?.AGENTS?.USE ?? "undef")));
'
