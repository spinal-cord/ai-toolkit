// Config encryption (X25519 + AES-256-GCM), browser-only.
//
// A keypair is generated in the browser (Settings -> Config Encryption):
//   - the PRIVATE key is stored in localStorage and NEVER sent to the server
//   - the PUBLIC key is stored in Settings (CONFIG_PUBLIC_KEY)
//
// When the user creates/edits a job config, the webui encrypts the config
// JSON with the public key and sends BOTH versions to the server:
//   - unencrypted  -> used to start training (worker/python)
//   - encrypted    -> stored on the server so it can be fetched + edited later
//
// Only the holder of the private key (this browser) can decrypt the stored
// config; the AI-toolkit itself cannot decrypt it.
//
// On-disk/blob format:
//   MAGIC(8) | ephemeral_pub(32) | nonce(12) | AES-GCM ciphertext(+16 tag)
// Key: ECDH(ephemeral_priv, recipient_pub) -> HKDF-SHA256 -> AES-256-GCM key.
// MAGIC is used as the GCM AAD so a corrupted/truncated blob fails auth.
//
// Must stay in sync with the crypto parameters (magic/salt/info) if you ever
// add a server-side or CLI decryption path.

// PKCS8 DER prefix wrapping a raw 32-byte X25519 private key (RFC 8410):
// SEQUENCE { INTEGER 0, SEQUENCE { OID id-X25519, OCTET STRING (raw key) } }.
// The WebCrypto spec no longer allows importing a raw X25519 PRIVATE key
// (raw import is public-key-only now), so the key stored in localStorage is
// wrapped in this PKCS8 envelope before importKey.
const X25519_PKCS8_PREFIX = new Uint8Array([
  0x30, 0x2e, 0x02, 0x01, 0x00, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x6e, 0x04, 0x22, 0x04, 0x20,
]);

const CONFIG_MAGIC = 'AITKCFG1';
const HKDF_SALT = 'AITK-CFG-SALT-v1';
const HKDF_INFO = 'AITK-CFG-KEY-v1';
const LS_PRIVATE_KEY = 'AITK_CONFIG_PRIVATE_KEY';

function toBase64(bytes: Uint8Array): string {
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function fromBase64(b64: string): Uint8Array {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function magicBytes(): Uint8Array {
  return new TextEncoder().encode(CONFIG_MAGIC);
}

// ---------------------------------------------------------------------------
// Private key storage (browser only - never sent to the server)
// ---------------------------------------------------------------------------
export function storeConfigPrivateKey(privateKeyB64: string): void {
  localStorage.setItem(LS_PRIVATE_KEY, privateKeyB64);
}

export function getConfigPrivateKey(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem(LS_PRIVATE_KEY);
}

export function hasConfigPrivateKey(): boolean {
  if (typeof window === 'undefined') return false;
  const v = localStorage.getItem(LS_PRIVATE_KEY);
  return v !== null && v.length > 0;
}

export function clearConfigPrivateKey(): void {
  if (typeof window === 'undefined') return;
  localStorage.removeItem(LS_PRIVATE_KEY);
}

// ---------------------------------------------------------------------------
// Key pair generation
// ---------------------------------------------------------------------------
export async function generateConfigKeypair(): Promise<{ privateKeyB64: string; publicKeyB64: string }> {
  const pair = (await crypto.subtle.generateKey({ name: 'X25519' }, true, ['deriveBits'])) as CryptoKeyPair;
  // The private key is stored as PKCS8 DER (base64): the WebCrypto spec no
  // longer allows 'raw' export of a private OKP key (raw is public-only).
  const priv = new Uint8Array(await crypto.subtle.exportKey('pkcs8', pair.privateKey));
  const pub = new Uint8Array(await crypto.subtle.exportKey('raw', pair.publicKey));
  return { privateKeyB64: toBase64(priv), publicKeyB64: toBase64(pub) };
}

async function hkdfToAesKey(ikm: ArrayBuffer | Uint8Array, usages: KeyUsage[]): Promise<CryptoKey> {
  const base = await crypto.subtle.importKey('raw', ikm as BufferSource, { name: 'HKDF' }, false, ['deriveKey']);
  return crypto.subtle.deriveKey(
    {
      name: 'HKDF',
      hash: 'SHA-256',
      salt: new TextEncoder().encode(HKDF_SALT) as BufferSource,
      info: new TextEncoder().encode(HKDF_INFO) as BufferSource,
    },
    base,
    { name: 'AES-GCM', length: 256 },
    false,
    usages,
  );
}

// Encrypt a config (JSON string) with the recipient X25519 public key.
// Returns a base64 blob: MAGIC(8) | ephemeral_pub(32) | nonce(12) | AES-GCM ct(+16 tag)
export async function encryptConfig(plaintext: string, recipientPubB64: string): Promise<string> {
  const recipientPub = (await crypto.subtle.importKey(
    'raw',
    fromBase64(recipientPubB64) as BufferSource,
    { name: 'X25519' },
    false,
    [],
  )) as CryptoKey;

  const eph = (await crypto.subtle.generateKey({ name: 'X25519' }, false, ['deriveBits'])) as CryptoKeyPair;
  // WebCrypto ECDH: the peer public key goes in algorithm.public; the 3rd arg
  // is the output length in bytes.
  const shared = await crypto.subtle.deriveBits(
    { name: 'X25519', public: recipientPub } as EcdhKeyDeriveParams,
    eph.privateKey,
    32,
  );
  const aesKey = await hkdfToAesKey(shared, ['encrypt']);

  const nonce = crypto.getRandomValues(new Uint8Array(12));
  const ephPub = new Uint8Array(await crypto.subtle.exportKey('raw', eph.publicKey));
  const ciphertext = new Uint8Array(
    await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv: nonce as BufferSource, additionalData: magicBytes() as BufferSource },
      aesKey,
      new TextEncoder().encode(plaintext),
    ),
  );

  const magic = magicBytes();
  const blob = new Uint8Array(magic.length + 32 + 12 + ciphertext.length);
  blob.set(magic, 0);
  blob.set(ephPub, magic.length);
  blob.set(nonce, magic.length + 32);
  blob.set(ciphertext, magic.length + 32 + 12);
  return toBase64(blob);
}

// Decrypt a config blob with the browser's stored private key.
export async function decryptConfig(blobB64: string, privateKeyB64: string): Promise<string> {
  const blob = fromBase64(blobB64);
  const magic = magicBytes();
  const minLen = magic.length + 32 + 12 + 16;
  if (blob.length < minLen) {
    throw new Error('Config blob is too small to be an AITK encrypted config');
  }
  for (let i = 0; i < magic.length; i++) {
    if (blob[i] !== magic[i]) throw new Error('Not an AITK encrypted config (bad magic)');
  }
  const ephPub = blob.slice(magic.length, magic.length + 32);
  const nonce = blob.slice(magic.length + 32, magic.length + 32 + 12);
  const ciphertext = blob.slice(magic.length + 32 + 12);

  const ephPubKey = (await crypto.subtle.importKey(
    'raw',
    ephPub as BufferSource,
    { name: 'X25519' },
    false,
    [],
  )) as CryptoKey;

  // The stored private key is either PKCS8 DER (current format) or the raw
  // 32-byte scalar (legacy format from before the WebCrypto spec dropped
  // raw private import/export - see X25519_PKCS8_PREFIX above). Both are
  // normalized to PKCS8 before importKey.
  const stored = fromBase64(privateKeyB64);
  let pkcs8: Uint8Array;
  if (stored.length === X25519_PKCS8_PREFIX.length + 32) {
    pkcs8 = stored;
  } else if (stored.length === 32) {
    pkcs8 = new Uint8Array(X25519_PKCS8_PREFIX.length + 32);
    pkcs8.set(X25519_PKCS8_PREFIX, 0);
    pkcs8.set(stored, X25519_PKCS8_PREFIX.length);
  } else {
    throw new Error('Stored config private key is not a valid X25519 key');
  }
  const privKey = (await crypto.subtle.importKey(
    'pkcs8',
    pkcs8 as BufferSource,
    { name: 'X25519' },
    true,
    ['deriveBits'],
  )) as CryptoKey;

  const shared = await crypto.subtle.deriveBits(
    { name: 'X25519', public: ephPubKey } as EcdhKeyDeriveParams,
    privKey,
    32,
  );
  const aesKey = await hkdfToAesKey(shared, ['decrypt']);

  const plain = await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv: nonce as BufferSource, additionalData: magicBytes() as BufferSource },
    aesKey,
    ciphertext as BufferSource,
  );
  return new TextDecoder().decode(plain);
}

// True if a string looks like one of our encrypted config blobs (magic check).
export function isEncryptedConfigBlob(b64: string): boolean {
  try {
    const blob = fromBase64(b64);
    const magic = magicBytes();
    if (blob.length < magic.length) return false;
    for (let i = 0; i < magic.length; i++) {
      if (blob[i] !== magic[i]) return false;
    }
    return true;
  } catch {
    return false;
  }
}
