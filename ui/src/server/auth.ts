// Native password auth for the AI Toolkit UI.
//
// Public-key mode (default since the Ed25519 upgrade):
// - The browser derives an Ed25519 keypair from the password (PBKDF2-SHA256,
//   210k iterations, fixed salt -> 32-byte seed) and sends ONLY the 32-byte
//   public key, stored in Settings as AUTH_PUBLIC_KEY (base64). The password
//   itself is never transmitted to or stored on the server.
// - Login is a challenge-response: the server issues a one-time 32-byte
//   challenge (60s TTL, single-use), the browser signs it with the derived
//   private key, and the server verifies the signature against the stored
//   public key. A captured login cannot be replayed (challenge is consumed).
// - Successful login issues a stateless session token (HMAC-signed, 30-day
//   expiry) which the custom server (server.js) verifies on every /api
//   request (as the AITK_SESSION cookie or a Bearer token for curl/CLI).
// - If no key/hash is stored, the UI is fully open (first-launch mode).
//
// Legacy mode (migration only): old installs have AUTH_PASSWORD_HASH (scrypt
// "salt:hash") and no public key. While that state exists, login also accepts
// the plaintext password once; re-saving the password in Settings stores the
// public key and deletes the hash, completing the migration.
//
// Must stay in sync with:
//   - ui/src/utils/authKey.ts   (browser-side derivation + signing)
//   - scripts/ui_session.py     (CLI login helper, same derivation)

import { createHmac, createPublicKey, randomBytes, scryptSync, timingSafeEqual, verify } from 'crypto';

export const SESSION_COOKIE = 'AITK_SESSION';
export const SESSION_TTL_SECONDS = 60 * 60 * 24 * 30; // 30 days

// The boot secret lives on globalThis so server.js (plain Node) and the bundled
// API routes (same process) share the same signing key without any file/DB state.
function getBootSecret(): string {
  const g = globalThis as any;
  if (!g.__AITK_BOOT_SECRET__) {
    g.__AITK_BOOT_SECRET__ = randomBytes(32).toString('hex');
    console.warn('[auth] generated boot secret (server.js normally pre-sets it)');
  }
  return g.__AITK_BOOT_SECRET__ as string;
}

export function hashPassword(password: string): string {
  const salt = randomBytes(16);
  const hash = scryptSync(password, salt, 64);
  return `${salt.toString('hex')}:${hash.toString('hex')}`;
}

export function verifyPassword(password: string, stored: string): boolean {
  try {
    const [saltHex, hashHex] = stored.split(':');
    if (!saltHex || !hashHex) return false;
    const salt = Buffer.from(saltHex, 'hex');
    const expected = Buffer.from(hashHex, 'hex');
    const actual = scryptSync(password, salt, expected.length);
    return actual.length === expected.length && timingSafeEqual(actual, expected);
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// Ed25519 public-key auth
// ---------------------------------------------------------------------------

// SPKI DER prefix wrapping a raw 32-byte Ed25519 public key (RFC 8410).
const ED25519_SPKI_PREFIX = Buffer.from('302a300506032b6570032100', 'hex');

function ed25519PublicKey(b64: string) {
  const raw = Buffer.from(b64, 'base64');
  if (raw.length !== 32) return null;
  try {
    // createPublicKey throws if the 32 bytes are not a valid Ed25519 point.
    return createPublicKey({ key: Buffer.concat([ED25519_SPKI_PREFIX, raw]), format: 'der', type: 'spki' });
  } catch {
    return null;
  }
}

export function isValidEd25519PublicKey(b64: string): boolean {
  return ed25519PublicKey(b64) !== null;
}

export function verifyEd25519Signature(publicKeyB64: string, challengeHex: string, signatureB64: string): boolean {
  const publicKey = ed25519PublicKey(publicKeyB64);
  if (!publicKey) return false;
  let challenge: Buffer;
  let signature: Buffer;
  try {
    challenge = Buffer.from(challengeHex, 'hex');
    signature = Buffer.from(signatureB64, 'base64');
  } catch {
    return false;
  }
  if (challenge.length !== 32 || signature.length !== 64) return false;
  try {
    return verify(null, challenge, publicKey, signature);
  } catch {
    return false;
  }
}

// ---------------------------------------------------------------------------
// One-time login challenges (replay protection)
// ---------------------------------------------------------------------------

// Stored on globalThis so the bundled API routes and server.js (same process)
// share the same store, like the boot secret.
function getChallengeStore(): Map<string, number> {
  const g = globalThis as any;
  if (!(g.__AITK_AUTH_CHALLENGES__ instanceof Map)) {
    g.__AITK_AUTH_CHALLENGES__ = new Map<string, number>();
  }
  return g.__AITK_AUTH_CHALLENGES__;
}

const CHALLENGE_TTL_MS = 60_000;

export function issueChallenge(): string {
  const store = getChallengeStore();
  const now = Date.now();
  for (const [k, exp] of store) {
    if (exp < now) store.delete(k);
  }
  if (store.size > 1000) store.clear();
  const challenge = randomBytes(32).toString('hex');
  store.set(challenge, now + CHALLENGE_TTL_MS);
  return challenge;
}

export function consumeChallenge(challenge: string): boolean {
  const store = getChallengeStore();
  const exp = store.get(challenge);
  if (!exp || exp < Date.now()) return false;
  store.delete(challenge); // single-use
  return true;
}

function b64url(buf: Buffer): string {
  return buf.toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function b64urlDecode(s: string): Buffer {
  return Buffer.from(s.replace(/-/g, '+').replace(/_/g, '/'), 'base64');
}

export function signSessionToken(): string {
  const payload = b64url(Buffer.from(JSON.stringify({ exp: Date.now() + SESSION_TTL_SECONDS * 1000 })));
  const sig = createHmac('sha256', getBootSecret()).update(payload).digest();
  return `${payload}.${b64url(sig)}`;
}

export function verifySessionToken(token: string | undefined | null): boolean {
  if (!token) return false;
  const parts = String(token).split('.');
  if (parts.length !== 2) return false;
  const [payload, sig] = parts;
  const expected = createHmac('sha256', getBootSecret()).update(payload).digest();
  let actual: Buffer;
  try {
    actual = b64urlDecode(sig);
  } catch {
    return false;
  }
  if (actual.length !== expected.length || !timingSafeEqual(actual, expected)) return false;
  try {
    const data = JSON.parse(b64urlDecode(payload).toString('utf8'));
    return typeof data.exp === 'number' && data.exp > Date.now();
  } catch {
    return false;
  }
}
