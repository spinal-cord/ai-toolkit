#!/usr/bin/env python3
"""
Decrypt an AI Toolkit encrypted file, IN PLACE.

STANDALONE: this script does not import anything from the ai-toolkit
codebase. It only needs Python 3.8+ and the `cryptography` package:
    pip install cryptography

Supported blob types (auto-detected by magic header):
  * AITKSAMP - samples: X25519 public-key scheme, private key derived from
               the password (Settings -> Security -> Sample Encryption
               Password)
  * AITKENC1 - legacy / dataset password-encrypted blobs (scrypt -> AES-256-GCM)
  * AITKCFG1 - job configs (config.json): the SAME asymmetric logic as
               samples (X25519 ECDH -> HKDF-SHA256 -> AES-256-GCM), but the
               keypair is GENERATED in the browser (Settings -> Config
               Encryption) and the private key is kept in the browser's
               localStorage - it is NOT derived from a password. Export it
               from the browser devtools console:
                   localStorage.getItem('AITK_CONFIG_PRIVATE_KEY')
               and pass it with --config-key (or the CONFIG_PRIVATE_KEY env).

Behaviour:
  * The decrypted bytes overwrite the SAME file by default
    (use -o/--output to write somewhere else instead).
  * A file containing the base64 text of a blob (e.g. a config.json saved
    from the UI database) or a JSON document whose values are such blobs is
    detected and handled too.

Usage:
    python decrypt_sample.py <password> <path_to_encrypted_file>
    python decrypt_sample.py --password <password> <path>
    python decrypt_sample.py <path> --config-key <base64_private_key>
    python decrypt_sample.py <path> -o decrypted_out.png
"""

import argparse
import base64
import hashlib
import json
import os
import sys

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric.x25519 import (
        X25519PrivateKey,
        X25519PublicKey,
    )
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
except ImportError:
    print("Error: the 'cryptography' package is required (pip install cryptography).")
    sys.exit(1)


class DecryptError(Exception):
    pass


# --------------------------------------------------------------------------- #
# Blob formats (must stay in sync with toolkit/dataset_crypto.py and the
# browser code in ui/src/utils/*.ts)
# --------------------------------------------------------------------------- #

MAGIC = b"AITKENC1"          # dataset / legacy: scrypt(password) -> AES-256-GCM
SAMPLE_MAGIC = b"AITKSAMP"   # samples: X25519(password-derived key) + AES-256-GCM
CONFIG_MAGIC = b"AITKCFG1"   # configs: X25519(browser-generated key) + AES-256-GCM

_SALT_LEN = 16
_NONCE_LEN = 12
_HEADER = len(MAGIC) + _SALT_LEN + _NONCE_LEN           # 36 bytes
_PUB_HEADER = len(SAMPLE_MAGIC) + 32 + _NONCE_LEN       # 52 bytes (same for CONFIG)

# Sample key derivation - must match ui/src/utils/sampleKey.ts
SAMPLE_KEY_SALT = b"AITK-SAMPLE-KEY-SALT-v1"
SAMPLE_PBKDF2_ITERATIONS = 210_000
# HKDF info binding the sample key to its purpose
SAMPLE_HKDF_INFO = b"AITK-SAMPLE-V1"
# Config crypto parameters - must match ui/src/utils/configKey.ts
CONFIG_HKDF_SALT = b"AITK-CFG-SALT-v1"
CONFIG_HKDF_INFO = b"AITK-CFG-KEY-v1"


# --------------------------------------------------------------------------- #
# Crypto primitives
# --------------------------------------------------------------------------- #

def _derive_dataset_key(password: str, salt: bytes) -> bytes:
    return Scrypt(salt=salt, length=32, n=2 ** 14, r=8, p=1).derive(password.encode("utf-8"))


def _derive_sample_private_key(password: str) -> X25519PrivateKey:
    seed = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), SAMPLE_KEY_SALT,
        SAMPLE_PBKDF2_ITERATIONS, dklen=32,
    )
    return X25519PrivateKey.from_private_bytes(seed)


def _x25519_shared_secret(private_key: X25519PrivateKey, peer_public: X25519PublicKey) -> bytes:
    # cryptography >= 42: private_key.exchange(peer_public)
    # older releases:      peer_public.exchange(private_key)
    if hasattr(private_key, "exchange"):
        return private_key.exchange(peer_public)
    return peer_public.exchange(private_key)


# --------------------------------------------------------------------------- #
# Per-scheme decryption
# --------------------------------------------------------------------------- #

def decrypt_dataset_blob(blob: bytes, password: str) -> bytes:
    """AITKENC1: MAGIC | salt(16) | nonce(12) | AES-GCM ct(+16 tag)."""
    if len(blob) < _HEADER + 16:
        raise DecryptError("Blob is too small to be a valid AITKENC1 file.")
    salt = blob[len(MAGIC):len(MAGIC) + _SALT_LEN]
    nonce = blob[len(MAGIC) + _SALT_LEN:_HEADER]
    ct = blob[_HEADER:]
    key = _derive_dataset_key(password, salt)
    try:
        return AESGCM(key).decrypt(nonce, ct, MAGIC)
    except Exception as e:
        raise DecryptError(f"Decryption failed (wrong password or corrupt file): {e}")


def decrypt_sample_blob(blob: bytes, password: str) -> bytes:
    """AITKSAMP: SAMPLE_MAGIC | ephemeral_pub(32) | nonce(12) | AES-GCM ct(+16 tag)."""
    if len(blob) < _PUB_HEADER + 16:
        raise DecryptError("Blob is too small to be a valid AITKSAMP file.")
    eph_pub = X25519PublicKey.from_public_bytes(blob[len(SAMPLE_MAGIC):len(SAMPLE_MAGIC) + 32])
    nonce = blob[len(SAMPLE_MAGIC) + 32:_PUB_HEADER]
    ct = blob[_PUB_HEADER:]
    private_key = _derive_sample_private_key(password)
    shared = _x25519_shared_secret(private_key, eph_pub)
    aes_key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=SAMPLE_HKDF_INFO).derive(shared)
    try:
        return AESGCM(aes_key).decrypt(nonce, ct, SAMPLE_MAGIC)
    except Exception as e:
        raise DecryptError(f"Sample decryption failed (wrong password or corrupt file): {e}")


def decrypt_config_blob(blob: bytes, private_key_b64: str) -> bytes:
    """AITKCFG1: CONFIG_MAGIC | ephemeral_pub(32) | nonce(12) | AES-GCM ct(+16 tag).

    The private key (base64, 32 raw bytes) comes from the browser's
    localStorage - it is NOT derived from a password.
    """
    if len(blob) < _PUB_HEADER + 16:
        raise DecryptError("Blob is too small to be a valid AITKCFG1 file.")
    try:
        recipient_priv = X25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64))
        eph_pub = X25519PublicKey.from_public_bytes(blob[len(CONFIG_MAGIC):len(CONFIG_MAGIC) + 32])
    except Exception as e:
        raise DecryptError(f"Invalid config private key or blob: {e}")
    nonce = blob[len(CONFIG_MAGIC) + 32:_PUB_HEADER]
    ct = blob[_PUB_HEADER:]
    shared = _x25519_shared_secret(recipient_priv, eph_pub)
    aes_key = HKDF(
        algorithm=hashes.SHA256(), length=32,
        salt=CONFIG_HKDF_SALT, info=CONFIG_HKDF_INFO,
    ).derive(shared)
    try:
        return AESGCM(aes_key).decrypt(nonce, ct, CONFIG_MAGIC)
    except Exception as e:
        raise DecryptError(f"Config decryption failed (wrong private key or corrupt blob): {e}")


# --------------------------------------------------------------------------- #
# Blob detection / dispatch
# --------------------------------------------------------------------------- #

def is_known_blob(data: bytes) -> bool:
    return data.startswith(MAGIC) or data.startswith(SAMPLE_MAGIC) or data.startswith(CONFIG_MAGIC)


def try_decode_base64_blob(data: bytes):
    """If ``data`` is text that base64-encodes an AITK blob, return the decoded
    blob, else None.  Handles surrounding whitespace and quotes."""
    try:
        text = data.decode("utf-8").strip().strip('"').strip().strip("'").strip()
    except (UnicodeDecodeError, ValueError):
        return None
    if not text or len(text) < 40 or "\n" in text:
        return None
    try:
        decoded = base64.b64decode(text, validate=True)
    except Exception:
        return None
    if is_known_blob(decoded):
        return decoded
    return None


def decrypt_auto(blob: bytes, password: str = "", config_private_key: str = "") -> bytes:
    """Detect the AITK blob type and decrypt it."""
    if not is_known_blob(blob):
        decoded = try_decode_base64_blob(blob)
        if decoded is None:
            raise DecryptError(
                "File is not an AITK encrypted blob (no AITKENC1/AITKSAMP/AITKCFG1 magic, "
                "and not base64 of one)."
            )
        blob = decoded
    if blob.startswith(CONFIG_MAGIC):
        if not config_private_key:
            raise DecryptError(
                "Config blob (AITKCFG1) needs the X25519 private key. Export it from the "
                "browser (devtools console: localStorage.getItem('AITK_CONFIG_PRIVATE_KEY')) "
                "and pass it via --config-key / CONFIG_PRIVATE_KEY."
            )
        return decrypt_config_blob(blob, config_private_key)
    if not password:
        raise DecryptError(
            "This blob type (AITKENC1/AITKSAMP) is password-encrypted but no password was given "
            "(positional argument, --password, or DATASET_PASSWORD/SAMPLE_DECRYPT_PASSWORD env var)."
        )
    if blob.startswith(SAMPLE_MAGIC):
        return decrypt_sample_blob(blob, password)
    return decrypt_dataset_blob(blob, password)


# --------------------------------------------------------------------------- #
# File-level decryption (in place by default)
# --------------------------------------------------------------------------- #

def _decrypt_json_value(value, password, config_key, stats):
    """Recursively replace base64 AITK blobs inside a JSON document."""
    if isinstance(value, str):
        decoded = try_decode_base64_blob(value.encode("utf-8"))
        if decoded is not None:
            plain = decrypt_auto(decoded, password, config_key)
            stats["decrypted"] += 1
            try:
                return json.loads(plain.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                return plain.decode("utf-8", errors="replace")
        return value
    if isinstance(value, dict):
        return {k: _decrypt_json_value(v, password, config_key, stats) for k, v in value.items()}
    if isinstance(value, list):
        return [_decrypt_json_value(v, password, config_key, stats) for v in value]
    return value


def decrypt_any_file(path: str, password: str, config_key: str, output_path=None) -> str:
    with open(path, "rb") as f:
        data = f.read()

    # 1) Direct binary blob (AITKENC1 / AITKSAMP / AITKCFG1).
    if is_known_blob(data):
        plain = decrypt_auto(data, password, config_key)
    else:
        # 2) base64 text of a blob (possibly quoted), or a JSON document
        #    containing such blobs (e.g. a saved config.json).
        text = None
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            pass
        if text is not None:
            decoded = try_decode_base64_blob(data)
            if decoded is not None:
                plain = decrypt_auto(decoded, password, config_key)
            else:
                try:
                    doc = json.loads(text)
                except ValueError:
                    raise DecryptError(
                        f"{path} is not an AITK encrypted file (no known magic header, "
                        "not base64 of one, and not a JSON document containing one)."
                    )
                stats = {"decrypted": 0}
                new_doc = _decrypt_json_value(doc, password, config_key, stats)
                if stats["decrypted"] == 0:
                    raise DecryptError(
                        f"{path} is a JSON document but no encrypted AITK blob was found in it."
                    )
                plain = json.dumps(new_doc, indent=2, ensure_ascii=False).encode("utf-8")
        else:
            raise DecryptError(
                f"{path} is not an AITK encrypted file (no known magic header and not text)."
            )

    if output_path is None:
        output_path = path  # in place
    with open(output_path, "wb") as f:
        f.write(plain)
    return output_path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Decrypt an AI Toolkit encrypted file (sample, legacy blob or config) in place. "
                    "Standalone - requires only the 'cryptography' package."
    )
    parser.add_argument(
        "pos",
        nargs="*",
        metavar="arg",
        help="[password] path - the LAST positional argument is the file path; "
             "the first (when two are given) is the password. "
             "The password is not needed for config blobs (AITKCFG1).",
    )
    parser.add_argument(
        "--password",
        dest="password_opt",
        help="The encryption password (alternative to the positional argument).",
    )
    parser.add_argument(
        "--config-key",
        dest="config_key",
        default=None,
        help="X25519 private key (base64) for config blobs (AITKCFG1). Export from the "
             "browser: localStorage.getItem('AITK_CONFIG_PRIVATE_KEY').",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Optional output path (default: overwrite the input file in place).",
    )
    args = parser.parse_args()

    if not args.pos:
        parser.error("No input path given.")
    path = args.pos[-1]
    password_pos = args.pos[0] if len(args.pos) >= 2 else None
    if not os.path.exists(path):
        print(f"Error: file not found: {path}")
        sys.exit(1)

    password = password_pos or args.password_opt or os.environ.get(
        "SAMPLE_DECRYPT_PASSWORD") or os.environ.get("DATASET_PASSWORD") or ""
    config_key = args.config_key or os.environ.get("CONFIG_PRIVATE_KEY", "")

    try:
        output_path = decrypt_any_file(path, password, config_key, output_path=args.output)
    except DecryptError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if output_path == path:
        print(f"Decrypted in place: {path}")
    else:
        print(f"Decrypted: {path}")
        print(f"Output:    {output_path}")


if __name__ == "__main__":
    main()
