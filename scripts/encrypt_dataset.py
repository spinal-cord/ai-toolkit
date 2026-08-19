#!/usr/bin/env python3
"""
Encrypt / decrypt dataset items IN PLACE (txt, json, mp4, jpg, png, ...).

STANDALONE: this script does not import anything from the ai-toolkit
codebase. It only needs Python 3.8+ and the `cryptography` package:
    pip install cryptography

Dataset files use the PASSWORD-BASED (symmetric) AITKENC1 format:
    scrypt(password, per-file salt) -> AES-256-GCM
This is a different scheme from the sample/config public-key (X25519)
format, which is why it has its own script - the password is the only
secret and it is never stored.

When a dataset password is configured, the training dataloader
transparently decrypts these files into RAM, so encrypted datasets
train exactly like plain ones.

Usage:
    python encrypt_dataset.py <password> <file_or_dir> [more paths...]
    python encrypt_dataset.py <password> <dataset_dir> --decrypt

  * Default direction is ENCRYPT (pass --decrypt to decrypt instead).
  * The command is idempotent: files already in the target state are
    skipped.
  * Directories are scanned recursively (.git and hidden dirs are skipped).
  * Files are rewritten in place (decrypted/encrypted bytes replace the
    file).
  * The password can also come from the DATASET_PASSWORD env var.
"""

import argparse
import os
import sys

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
except ImportError:
    print("Error: the 'cryptography' package is required (pip install cryptography).")
    sys.exit(1)


class DatasetCryptoError(Exception):
    pass


# Blob format (must stay in sync with toolkit/dataset_crypto.py):
#   MAGIC(8) | salt(16) | nonce(12) | AES-GCM ciphertext(+16 byte tag)
# Key: scrypt(password, salt, n=2**14, r=8, p=1) -> 32 bytes.
# MAGIC is used as the GCM AAD so a corrupted/truncated blob fails auth.
MAGIC = b"AITKENC1"
_SALT_LEN = 16
_NONCE_LEN = 12
_HEADER = len(MAGIC) + _SALT_LEN + _NONCE_LEN  # 36 bytes


def is_encrypted_blob(data: bytes) -> bool:
    return data[:len(MAGIC)] == MAGIC


def _derive_key(password: str, salt: bytes) -> bytes:
    return Scrypt(salt=salt, length=32, n=2 ** 14, r=8, p=1).derive(password.encode("utf-8"))


def encrypt_bytes(plain: bytes, password: str) -> bytes:
    salt = os.urandom(_SALT_LEN)
    nonce = os.urandom(_NONCE_LEN)
    key = _derive_key(password, salt)
    ct = AESGCM(key).encrypt(nonce, plain, MAGIC)
    return MAGIC + salt + nonce + ct


def decrypt_bytes(blob: bytes, password: str) -> bytes:
    if not is_encrypted_blob(blob):
        raise DatasetCryptoError("Blob is not in the AITKENC1 encrypted format.")
    if len(blob) < _HEADER + 16:
        raise DatasetCryptoError("Blob is too small to be a valid AITKENC1 file.")
    salt = blob[len(MAGIC):len(MAGIC) + _SALT_LEN]
    nonce = blob[len(MAGIC) + _SALT_LEN:_HEADER]
    ct = blob[_HEADER:]
    key = _derive_key(password, salt)
    try:
        return AESGCM(key).decrypt(nonce, ct, MAGIC)
    except Exception as e:  # InvalidTag or corrupt data
        raise DatasetCryptoError(f"Decryption failed (wrong password or corrupt file): {e}")


# --------------------------------------------------------------------------- #
# File / directory handling
# --------------------------------------------------------------------------- #

def collect_files(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, dirs, names in os.walk(p):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for name in sorted(names):
                    if name.startswith("."):
                        continue
                    files.append(os.path.join(root, name))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"Warning: path not found, skipping: {p}")
    return files


def process_file(path: str, password: str, mode: str) -> str:
    """Encrypt or decrypt one file in place.

    Returns: 'encrypted' | 'decrypted' | 'skipped'
    Raises:  DatasetCryptoError on failure.
    """
    with open(path, "rb") as f:
        data = f.read()

    is_enc = is_encrypted_blob(data)

    if mode == "encrypt" and is_enc:
        return "skipped"
    if mode == "decrypt" and not is_enc:
        return "skipped"

    if mode == "decrypt":
        out = decrypt_bytes(data, password)
        result = "decrypted"
    else:
        out = encrypt_bytes(data, password)
        result = "encrypted"

    with open(path, "wb") as f:
        f.write(out)
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Encrypt/decrypt dataset files in place (symmetric password scheme). "
                    "Standalone - requires only the 'cryptography' package."
    )
    parser.add_argument(
        "pos",
        nargs="*",
        metavar="arg",
        help="[password] path [more paths...] - the LAST positional argument(s) are the "
             "files/directories to process; the first (when the password is given "
             "positionally) is the dataset password.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--encrypt", dest="mode_enc", action="store_true",
                       help="Only encrypt (already-encrypted files are skipped). Default.")
    group.add_argument("--decrypt", dest="mode_dec", action="store_true",
                       help="Only decrypt (plain files are skipped).")
    args = parser.parse_args()

    if not args.pos:
        parser.error("No input path given.")
    # [password] path [more paths...]:
    #   - a single positional is always the path (password from env var)
    #   - with several positionals, they are ALL paths when every one of them
    #     exists on disk (password then comes from the env var); otherwise
    #     the first is the password and the rest are paths.
    if len(args.pos) >= 2 and not all(os.path.exists(p) for p in args.pos):
        password_pos, paths = args.pos[0], args.pos[1:]
    else:
        password_pos, paths = None, args.pos

    password = password_pos or os.environ.get("DATASET_PASSWORD", "")
    if not password:
        parser.error("No password given (positional argument or DATASET_PASSWORD env var).")

    mode = "decrypt" if args.mode_dec else "encrypt"

    files = collect_files(paths)
    if not files:
        print("No files found.")
        sys.exit(1)

    counts = {"encrypted": 0, "decrypted": 0, "skipped": 0, "errors": 0}
    for path in files:
        try:
            result = process_file(path, password, mode)
            counts[result] += 1
            print(f"[{result:>9}] {path}")
        except DatasetCryptoError as e:
            counts["errors"] += 1
            print(f"[   error] {path}: {e}", file=sys.stderr)
        except OSError as e:
            counts["errors"] += 1
            print(f"[   error] {path}: {e}", file=sys.stderr)

    print(
        f"\nDone: {counts['encrypted']} encrypted, {counts['decrypted']} decrypted, "
        f"{counts['skipped']} skipped, {counts['errors']} error(s) - {len(files)} file(s) total."
    )
    sys.exit(1 if counts["errors"] else 0)


if __name__ == "__main__":
    main()
