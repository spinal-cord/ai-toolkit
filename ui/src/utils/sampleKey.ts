// Client-side sample-encryption key derivation.
//
// The user's password is used ONLY in the browser to derive an X25519
// keypair:
//     PBKDF2-HMAC-SHA256(password, SALT, 210_000 iterations) -> 32-byte seed
//     seed -> X25519 private key (pure-JS RFC 7748 ladder, see
//             utils/curve25519.ts - WebCrypto no longer allows importing
//             a raw private OKP key, and clamping matches Python's
//             X25519PrivateKey.from_private_bytes)
// Only the 32-byte PUBLIC key (base64) is sent to the server and stored in
// Settings (SAMPLE_PUBLIC_KEY). The private key and the password itself
// never leave the browser, so the server can encrypt generated samples but
// can never decrypt them.
//
// Decryption is done offline with:
//     python scripts/decrypt_sample.py <password> <encrypted_sample>
// which re-derives the private key from the password (see
// toolkit/dataset_crypto.py:derive_sample_private_key - the parameters below
// MUST stay in sync).

import { x25519PublicFromSeed } from '@/utils/curve25519';

// Must match SAMPLE_KEY_SALT in toolkit/dataset_crypto.py
const SALT = new TextEncoder().encode('AITK-SAMPLE-KEY-SALT-v1');
// Must match SAMPLE_PBKDF2_ITERATIONS in toolkit/dataset_crypto.py
const ITERATIONS = 210_000;

function toBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

/**
 * Derive the sample-encryption X25519 public key (base64, 32 bytes raw)
 * from the user's password.
 *
 * Note: the public key is computed with the pure-JS RFC 7748 ladder in
 * utils/curve25519.ts because the WebCrypto spec no longer supports
 * importing a raw X25519 PRIVATE key (raw import is public-key-only now,
 * which is why older `importKey('raw', seed, { name: 'X25519' }, ...,
 * ['deriveBits'])` code broke on modern browsers).
 */
export async function deriveSamplePublicKey(password: string): Promise<string> {
  const encoder = new TextEncoder();

  const baseKey = await crypto.subtle.importKey(
    'raw',
    encoder.encode(password),
    { name: 'PBKDF2' },
    false,
    ['deriveBits'],
  );

  const seed = new Uint8Array(
    await crypto.subtle.deriveBits(
      { name: 'PBKDF2', hash: 'SHA-256', salt: SALT, iterations: ITERATIONS },
      baseKey,
      256,
    ),
  );

  return toBase64(x25519PublicFromSeed(seed));
}
