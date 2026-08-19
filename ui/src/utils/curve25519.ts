// Pure-JS Curve25519 / Ed25519 primitives (RFC 7748 + RFC 8032).
//
// Why this module exists: the WebCrypto spec no longer supports importing a
// raw 32-byte OKP (X25519/Ed25519) PRIVATE key. `subtle.importKey('raw',
// seed, { name: 'X25519' | 'Ed25519' }, ...)` now only accepts PUBLIC keys
// (usages must be empty, or 'verify' for Ed25519) and returns a bare
// CryptoKey - there is no WebCrypto way left to get a keypair from a seed.
// So deriving the public key (and Ed25519 signatures) from a PBKDF2 seed is
// done here in userland, following the reference algorithms of
//   - RFC 7748 section 5 (X25519 Montgomery ladder, a24 = 121665)
//   - RFC 8032 section 5.1 (Ed25519, extended-coordinate Edwards arithmetic)
// WebCrypto is still used for PBKDF2 (seed derivation) and SHA-512.
//
// These implementations are verified against the RFC 7748 test vectors and
// against Node.js crypto / Python `cryptography` (see test_curve25519.mjs).
// They match, bit for bit:
//   - Python X25519PrivateKey.from_private_bytes(seed).public_key()
//   - Python Ed25519PrivateKey.from_private_bytes(seed).public_key() / .sign()

const P = 2n ** 255n - 19n; // curve25519 / edwards25519 prime
const A24 = 121665n; // (486662 - 2) / 4, RFC 7748
const D_EDWARDS =
  37095705934669439343138083508754565189542113879843219016388785533085940283555n; // edwards25519 d
const L = 2n ** 252n + 27742317777372353535851937790883648493n; // prime-order subgroup order (RFC 8032)

const mod = (a: bigint): bigint => {
  const r = a % P;
  return r < 0n ? r + P : r;
};

const modL = (a: bigint): bigint => {
  const r = a % L;
  return r < 0n ? r + L : r;
};

// Modular exponentiation (reduces after every step, so intermediates stay
// ~2*255 bits instead of growing with the exponent).
function powmod(base: bigint, exp: bigint, m: bigint): bigint {
  let result = 1n;
  base = mod(base);
  while (exp > 0n) {
    if (exp & 1n) result = mod(result * base);
    base = mod(base * base);
    exp >>= 1n;
  }
  return result;
}

// Modular inverse via Fermat's little theorem (P is prime).
const inv = (a: bigint): bigint => powmod(a, P - 2n, P);

function encode32LE(n: bigint): Uint8Array {
  const out = new Uint8Array(32);
  for (let i = 0; i < 32; i++) {
    out[i] = Number(n & 0xffn);
    n >>= 8n;
  }
  return out;
}

function toBigLE(bytes: Uint8Array): bigint {
  let n = 0n;
  for (let i = bytes.length - 1; i >= 0; i--) n = (n << 8n) | BigInt(bytes[i]);
  return n;
}

async function sha512(data: Uint8Array): Promise<Uint8Array> {
  return new Uint8Array(await crypto.subtle.digest('SHA-512', data as BufferSource));
}

function concatBytes(...parts: Uint8Array[]): Uint8Array {
  const total = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) {
    out.set(p, off);
    off += p.length;
  }
  return out;
}

// ---------------------------------------------------------------------------
// X25519 (RFC 7748)
// ---------------------------------------------------------------------------

/**
 * X25519 scalar multiplication of the u-coordinate `u` by the 32-byte scalar
 * (clamped per RFC 7748). Returns the 32-byte little-endian u-coordinate.
 */
export function x25519(scalar: Uint8Array, u: bigint): Uint8Array {
  const k = new Uint8Array(scalar);
  k[0] &= 248;
  k[31] &= 127;
  k[31] |= 64;
  const s = toBigLE(k);

  let x1 = mod(u);
  let x2 = 1n;
  let z2 = 0n;
  let x3 = mod(u);
  let z3 = 1n;
  let swap = 0;

  // RFC 7748 section 5 Montgomery ladder (t = 254 down to 0).
  for (let t = 254; t >= 0; t--) {
    const kt = Number((s >> BigInt(t)) & 1n);
    swap ^= kt;
    if (swap) {
      [x2, x3] = [x3, x2];
      [z2, z3] = [z3, z2];
    }
    swap = kt;
    const A = mod(x2 + z2);
    const AA = mod(A * A);
    const B = mod(x2 - z2);
    const BB = mod(B * B);
    const E = mod(AA - BB);
    const C = mod(x3 + z3);
    const D = mod(x3 - z3);
    const DA = mod(D * A);
    const CB = mod(C * B);
    x3 = mod((DA + CB) ** 2n);
    z3 = mod(x1 * (DA - CB) ** 2n);
    x2 = mod(AA * BB);
    z2 = mod(E * (AA + A24 * E));
  }
  if (swap) {
    [x2, x3] = [x3, x2];
    [z2, z3] = [z3, z2];
  }
  return encode32LE(mod(x2 * inv(z2)));
}

/**
 * Compute the X25519 public key (32 bytes, little-endian u-coordinate) for
 * the given 32-byte seed, i.e. seed * basepoint(9). This is exactly what
 * Python's `X25519PrivateKey.from_private_bytes(seed).public_key()` computes
 * (both sides clamp the scalar per RFC 7748).
 */
export function x25519PublicFromSeed(seed: Uint8Array): Uint8Array {
  return x25519(seed, 9n);
}

// ---------------------------------------------------------------------------
// Ed25519 (RFC 8032) - edwards25519 in extended coordinates (X, Y, Z, T)
// ---------------------------------------------------------------------------

interface EdPoint {
  x: bigint;
  y: bigint;
  z: bigint;
  t: bigint;
}

// RFC 7748 section 4.1 basepoint (edwards25519 X, Y; T = X*Y with Z = 1).
const BASEPOINT_X = 15112221349535400772501151409588531511454012693041857206046113283949847762202n;
const BASEPOINT_Y = 46316835694926478169428394003475163141307993866256225615783033603165251855960n;
const BASEPOINT: EdPoint = {
  x: BASEPOINT_X,
  y: BASEPOINT_Y,
  z: 1n,
  t: mod(BASEPOINT_X * BASEPOINT_Y),
};

const IDENTITY: EdPoint = { x: 0n, y: 1n, z: 1n, t: 0n };

// Point addition on -x^2 + y^2 = 1 + d*x^2*y^2 (twisted Edwards, a = -1),
// extended coordinates (X, Y, Z, T) with T = X*Y/Z.
//
//   A = (Y1-X1)(Y2-X2)   B = (Y1+X1)(Y2+X2)
//   C = 2*d*T1*T2        D = 2*Z1*Z2
//   E = B-A   F = D-C   G = B+A   H = D+C
//   X3 = E*F   Y3 = G*H   Z3 = H*F   T3 = E*G
function edwardsAdd(p: EdPoint, q: EdPoint): EdPoint {
  const A = mod((p.y - p.x) * (q.y - q.x));
  const B = mod((p.y + p.x) * (q.y + q.x));
  const C = mod(2n * D_EDWARDS * p.t * q.t);
  const D = mod(2n * p.z * q.z);
  const E = mod(B - A);
  const F = mod(D - C);
  const G = mod(B + A);
  const H = mod(D + C);
  return { x: mod(E * F), y: mod(G * H), z: mod(H * F), t: mod(E * G) };
}

// Point doubling: the same addition with q = p (A = (Y1-X1)^2, etc.).
function edwardsDouble(p: EdPoint): EdPoint {
  return edwardsAdd(p, p);
}

// Double-and-add scalar multiplication.
// Must scan from bit 255 down: the RFC 8032 clamped scalar `a` has bit 254
// set (a |= 2^254), so a 252-bit scan would silently drop the 2^254 term.
// Starts at the first set bit: doubling the identity point in extended
// coordinates degenerates to (0,0,0,0), so the accumulator is seeded with
// the base point instead.
function edwardsScalarMult(s: bigint, base: EdPoint): EdPoint {
  let r = IDENTITY;
  let started = false;
  for (let t = 255; t >= 0; t--) {
    if (started) r = edwardsDouble(r);
    if ((s >> BigInt(t)) & 1n) {
      r = started ? edwardsAdd(r, base) : base;
      started = true;
    }
  }
  return r;
}

// RFC 8032 section 5.1 point encoding: little-endian y, sign bit of x in the
// top bit of the last byte.
function encodePoint(p: EdPoint): Uint8Array {
  const zi = inv(p.z);
  const x = mod(p.x * zi);
  const y = mod(p.y * zi);
  const out = encode32LE(y);
  out[31] ^= Number(x & 1n) ? 0x80 : 0x00;
  return out;
}

// RFC 8032 section 5.1 scalar clamping of the first 32 bytes of SHA-512(seed).
function clampScalar32(bytes: Uint8Array): bigint {
  const b = new Uint8Array(32);
  b.set(bytes.subarray(0, 32));
  b[0] &= 248;
  b[31] &= 63;
  b[31] |= 64;
  return toBigLE(b);
}

/**
 * Derive the Ed25519 public key (32-byte RFC 8032 encoding) from the
 * 32-byte seed. Matches Python's
 * `Ed25519PrivateKey.from_private_bytes(seed).public_key()`.
 */
export async function ed25519PublicKeyFromSeed(seed: Uint8Array): Promise<Uint8Array> {
  const h = await sha512(seed);
  const a = clampScalar32(h.subarray(0, 32));
  return encodePoint(edwardsScalarMult(a, BASEPOINT));
}

/**
 * Produce the 64-byte RFC 8032 Ed25519 signature of `message` with the key
 * derived from `seed`. Deterministic; matches Python's
 * `Ed25519PrivateKey.from_private_bytes(seed).sign(message)`.
 */
export async function ed25519Sign(seed: Uint8Array, message: Uint8Array): Promise<Uint8Array> {
  const h = await sha512(seed);
  const prefix = h.subarray(32, 64);
  const a = clampScalar32(h.subarray(0, 32));
  const A = encodePoint(edwardsScalarMult(a, BASEPOINT));

  const r = modL(toBigLE(await sha512(concatBytes(new Uint8Array(prefix), message))));
  const R = encodePoint(edwardsScalarMult(r, BASEPOINT));
  const k = modL(toBigLE(await sha512(concatBytes(R, A, message))));
  const S = modL(r + k * a);

  const sig = new Uint8Array(64);
  sig.set(R, 0);
  sig.set(encode32LE(S), 32);
  return sig;
}
