// Client-side UI-password key derivation (Ed25519).
//
// The password is used ONLY in the browser to derive an Ed25519 keypair:
//     PBKDF2-HMAC-SHA256(password, SALT, 210_000 iterations) -> 32-byte seed
//     seed -> Ed25519 key (pure-JS RFC 8032, see utils/curve25519.ts -
//             WebCrypto no longer allows importing a raw private OKP key)
// Only the 32-byte PUBLIC key (base64) is sent to the server and stored in
// Settings (AUTH_PUBLIC_KEY). The private key and the password itself never
// leave the browser.
//
// Login is a challenge-response: GET /api/auth/challenge -> sign the
// challenge with the private key -> POST /api/auth/login. The server
// verifies the signature with the stored public key (see
// ui/src/server/auth.ts). The password is therefore never transmitted, and
// the server holds no password verifier.
//
// Must stay in sync with:
//   - ui/src/server/auth.ts (server-side verification, challenge store)
//   - scripts/ui_session.py (CLI login helper, same PBKDF2 parameters)

import { ed25519PublicKeyFromSeed, ed25519Sign } from '@/utils/curve25519';

// Must match the salt used by scripts/ui_session.py
const SALT = new TextEncoder().encode('AITK-AUTH-KEY-SALT-v1');
// Must match the iteration count used by scripts/ui_session.py
const ITERATIONS = 210_000;

function toBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function hexToBytes(hex: string): Uint8Array {
  const out = new Uint8Array(hex.length / 2);
  for (let i = 0; i < out.length; i++) {
    out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

// Derive the 32-byte Ed25519 seed from the password (PBKDF2-SHA256 via
// WebCrypto). The public key and the signatures are then computed from the
// seed with the pure-JS RFC 8032 implementation in utils/curve25519.ts -
// the WebCrypto spec no longer supports importing a raw Ed25519 PRIVATE key
// (raw import is public-key-only now, which is why older
// `importKey('raw', seed, { name: 'Ed25519' }, ..., ['sign'])` code broke
// on modern browsers).
async function deriveSeed(password: string): Promise<Uint8Array> {
  const baseKey = await crypto.subtle.importKey(
    'raw',
    new TextEncoder().encode(password),
    { name: 'PBKDF2' },
    false,
    ['deriveBits'],
  );

  return new Uint8Array(
    await crypto.subtle.deriveBits(
      { name: 'PBKDF2', hash: 'SHA-256', salt: SALT, iterations: ITERATIONS },
      baseKey,
      256,
    ),
  );
}

/**
 * Derive the UI-password Ed25519 public key (base64, 32 bytes raw) from the
 * password.
 */
export async function deriveAuthPublicKey(password: string): Promise<string> {
  const seed = await deriveSeed(password);
  return toBase64(await ed25519PublicKeyFromSeed(seed));
}

/**
 * Sign a server-issued challenge (64 hex chars) with the private key derived
 * from the password. Returns the 64-byte signature as base64.
 */
export async function signChallenge(password: string, challengeHex: string): Promise<string> {
  const seed = await deriveSeed(password);
  const challenge = hexToBytes(challengeHex);
  return toBase64(await ed25519Sign(seed, challenge));
}
