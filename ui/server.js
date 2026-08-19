#!/usr/bin/env node
/*
 * Custom server for the AI Toolkit UI (replaces `next start --port 8675`).
 *
 * Adds two native features on top of the Next.js app:
 *   1. TLS — cert/key FILE PATHS are read from Settings (TLS_CERT / TLS_KEY).
 *      Same-protocol cert swaps apply in place (setSecureContext) within ~30s.
 *      A protocol switch (http<->https) recreates the server in place — the
 *      process never exits, so `concurrently` keeps it running.
 *   2. Password auth — the Ed25519 PUBLIC key is stored in Settings
 *      (AUTH_PUBLIC_KEY; legacy installs may still have AUTH_PASSWORD_HASH).
 *      The password itself is never stored or transmitted (login is an
 *      Ed25519 challenge-response, see src/server/auth.ts).
 *      No key  -> fully open (first-launch mode).
 *      Key set -> all /api routes require a session cookie (from
 *                 /api/auth/login) or a Bearer SESSION TOKEN (for curl/CLI
 *                 clients, e.g. obtained via scripts/ui_session.py).
 *
 * NOTE: this server is the ONLY supported way to run the UI (prod AND dev:
 * `node server.js --dev`). `next dev`/`next start` directly would bypass the
 * auth middleware, so it must not be used.
 */
const http = require('http');
const https = require('https');
const fs = require('fs');
const crypto = require('crypto');

process.env.NODE_ENV = process.env.NODE_ENV || 'production';
// --dev (or NODE_ENV=development) runs the app in Next dev mode (HMR etc.)
// while STILL enforcing the auth middleware - dev is not a bypass.
const dev = process.argv.includes('--dev') || process.env.NODE_ENV === 'development';

const next = require('next');
const { PrismaClient } = require('@prisma/client');
const { ensureSchema } = require('./db/ensureSchema');

// Shared with the API routes in this same process (stateless session tokens).
globalThis.__AITK_BOOT_SECRET__ = crypto.randomBytes(32).toString('hex');

const PORT = parseInt(process.env.PORT || '8675', 10);
const SESSION_COOKIE = 'AITK_SESSION';
const PUBLIC_PREFIXES = ['/api/img/', '/api/files/', '/api/auth'];
const SETTINGS_KEYS = ['AUTH_PUBLIC_KEY', 'AUTH_PASSWORD_HASH', 'TLS_CERT', 'TLS_KEY'];

const prisma = new PrismaClient();

// ---------- settings reads with a 10s TTL cache (no DB hit per request) ----------
// The revision counter is shared with the bundled @/server/settings (same
// process): flushCache() bumps it when the Settings page saves, so auth state
// (password set/cleared) applies immediately instead of waiting for the TTL.
const settingsCache = { value: null, ts: 0, rev: -1 };
async function getSettingsMap() {
  const now = Date.now();
  const rev = globalThis.__AITK_SETTINGS_REV__ || 0;
  if (!settingsCache.value || now - settingsCache.ts > 10000 || rev !== settingsCache.rev) {
    const rows = await prisma.settings.findMany({ where: { key: { in: SETTINGS_KEYS } } });
    settingsCache.value = Object.fromEntries(rows.map(r => [r.key, r.value]));
    settingsCache.ts = now;
    settingsCache.rev = rev;
  }
  return settingsCache.value;
}

// ---------- session token verification (mirrors src/server/auth.ts) ----------
function b64urlDecode(s) {
  return Buffer.from(s.replace(/-/g, '+').replace(/_/g, '/'), 'base64');
}
function verifySessionToken(token) {
  if (!token) return false;
  const parts = String(token).split('.');
  if (parts.length !== 2) return false;
  const [payload, sig] = parts;
  const expected = crypto.createHmac('sha256', globalThis.__AITK_BOOT_SECRET__).update(payload).digest();
  let actual;
  try {
    actual = b64urlDecode(sig);
  } catch {
    return false;
  }
  if (actual.length !== expected.length || !crypto.timingSafeEqual(actual, expected)) return false;
  try {
    const data = JSON.parse(b64urlDecode(payload).toString('utf8'));
    return typeof data.exp === 'number' && data.exp > Date.now();
  } catch {
    return false;
  }
}

function parseCookies(req) {
  const out = {};
  const header = req.headers.cookie;
  if (!header) return out;
  for (const part of header.split(';')) {
    const idx = part.indexOf('=');
    if (idx > -1) out[part.slice(0, idx).trim()] = decodeURIComponent(part.slice(idx + 1).trim());
  }
  return out;
}

async function authMiddleware(req, res) {
  const map = await getSettingsMap();
  const required = Boolean(map.AUTH_PUBLIC_KEY || map.AUTH_PASSWORD_HASH);
  if (!required) return true; // first-launch mode: no password configured

  const p = new URL(req.url, 'http://localhost').pathname;

  if (PUBLIC_PREFIXES.some(prefix => p.startsWith(prefix))) return true;

  if (p.startsWith('/api/')) {
    const cookies = parseCookies(req);
    if (verifySessionToken(cookies[SESSION_COOKIE])) return true;
    // Bearer carries a SESSION TOKEN (not the password) so curl/CLI clients
    // can authenticate without cookie handling (scripts/ui_session.py).
    const authHeader = req.headers.authorization || '';
    if (authHeader.startsWith('Bearer ') && verifySessionToken(authHeader.slice(7))) return true;
    res.statusCode = 401;
    res.setHeader('Content-Type', 'application/json');
    res.end(JSON.stringify({ error: 'Unauthorized' }));
    return false;
  }
  return true; // SPA shell + static assets load; AuthWrapper gates the UI
}

// ---------- TLS ----------
async function getTlsConfig() {
  const map = await getSettingsMap();
  const certPath = map.TLS_CERT || '';
  const keyPath = map.TLS_KEY || '';
  if (!certPath || !keyPath) return null;
  try {
    return { key: fs.readFileSync(keyPath), cert: fs.readFileSync(certPath), certPath, keyPath };
  } catch (e) {
    console.error(`[tls] cannot read cert/key (${keyPath}, ${tlsSigOf(certPath, keyPath)}): ${e.message}`);
    return null;
  }
}
function tlsSigOf(certPath, keyPath) {
  return `${certPath}:${keyPath}`;
}
function tlsSig(tls) {
  return tls ? `${tls.cert.length}:${tls.key.length}:${tls.certPath}:${tls.keyPath}` : '';
}

// ---------- request handling ----------
function makeHandler() {
  // turbopack in dev keeps the same build pipeline as `next dev --turbopack`
  const app = next({ dev, turbopack: dev });
  const handle = app.getRequestHandler();
  return {
    app,
    onReq: (req, res) => {
      authMiddleware(req, res)
        .then(allowed => {
          if (allowed) handle(req, res);
        })
        .catch(err => {
          console.error('[auth] middleware error:', err);
          if (!res.headersSent) {
            res.statusCode = 500;
            res.end('Internal error');
          }
        });
    },
  };
}

function makeServer(tls, onReq) {
  if (tls) return https.createServer({ key: tls.key, cert: tls.cert }, onReq);
  return http.createServer(onReq);
}

function listen(srv) {
  return new Promise((resolve, reject) => {
    const onError = (err) => reject(err);
    srv.once('error', onError);
    srv.listen(PORT, () => {
      srv.removeListener('error', onError);
      resolve();
    });
  });
}

function closeServer(srv) {
  return new Promise((resolve) => {
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(); } };
    try { if (srv.closeAllConnections) srv.closeAllConnections(); } catch {}
    try { srv.close(finish); } catch { finish(); }
    setTimeout(finish, 3000); // safety net
  });
}

// ---------- main ----------
let server = null;

(async () => {
  // Self-heal the SQLite schema (adds e.g. Job.job_config_encrypted to
  // pre-existing DBs so no manual `prisma db push` is needed).
  await ensureSchema();

  const { app, onReq } = makeHandler();
  await app.prepare();

  const tls = await getTlsConfig();
  server = makeServer(tls, onReq);
  await listen(server);
  console.log(
    tls
      ? `[server] TLS enabled (${tls.certPath} / ${tls.keyPath}) -> https on port ${PORT}`
      : `[server] plain HTTP on port ${PORT} (set TLS_CERT/TLS_KEY in Settings to enable TLS)`,
  );
  console.log(`[server] AI Toolkit UI listening on port ${PORT}`);

  // Poll for TLS changes so cert/key swaps apply without a process restart.
  let lastSig = tlsSig(tls);
  const poll = setInterval(async () => {
    try {
      const newTls = await getTlsConfig();
      const newSig = tlsSig(newTls);
      if (newSig === lastSig) return;
      lastSig = newSig;

      const isHttps = server instanceof https.Server;
      const newIsHttps = Boolean(newTls);

      if (isHttps && newIsHttps) {
        // Same protocol: swap the cert/key in place, no downtime.
        server.setSecureContext(newTls.key, newTls.cert);
        console.log('[server] TLS context reloaded (in place)');
      } else {
        // Protocol switch (http<->https): recreate the server in place.
        console.log(`[server] switching to ${newIsHttps ? 'HTTPS' : 'HTTP'} (in place)...`);
        const oldServer = server;
        await closeServer(oldServer);
        server = makeServer(newTls, onReq);
        await listen(server);
        console.log(`[server] now serving ${newIsHttps ? 'HTTPS' : 'HTTP'} on port ${PORT}`);
      }
    } catch (e) {
      console.error('[tls poll]', e);
    }
  }, 30000);
  if (poll.unref) poll.unref();

  process.on('SIGTERM', async () => {
    try { await prisma.$disconnect(); } catch {}
    process.exit(0);
  });
})().catch(e => {
  console.error('[server] failed to start:', e);
  process.exit(1);
});
