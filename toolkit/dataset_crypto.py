"""
Dataset encryption / decryption for the AI Toolkit.

Dataset files (images, videos, audio, captions, and the on-disk caches:
latents, text embeddings, clip-vision and optical flow) can be stored
encrypted at rest.  When a dataset password is configured the training
dataloader transparently decrypts them **into RAM only** before decoding
and never writes the decrypted bytes back to disk.

Design goals (per requirements):
  * First launch / no password  -> datasets are read exactly as before
    (plain files work with zero behaviour change).
  * Encrypted dataset + no password -> the job is stopped with a clear error.
  * No temp directory is ever created.  Decrypted media is held in RAM for the
    duration of a single decode and then released.
  * Caches are encrypted with the same password when a password is set, and
    old (unencrypted) caches keep working (detected per-file by magic bytes).

On-disk format for an encrypted blob:
    MAGIC(8) | salt(16) | nonce(12) | AES-GCM ciphertext(+16 byte tag)
Key derivation: scrypt(password, salt) -> 32 byte key.  MAGIC is used as the
GCM AAD so a corrupted/truncated blob fails authentication.

Sample encryption (generated samples) uses a *public key* scheme so the
password itself is never sent to or stored on the server:
  * The web UI derives an X25519 keypair from the password in the browser
    (PBKDF2-SHA256, 210k iterations, fixed salt -> 32-byte seed) and sends
    only the 32-byte public key (base64) to the server (SAMPLE_PUBLIC_KEY).
  * The server encrypts each sample with an ephemeral X25519 key:
        ECDH(ephemeral_priv, stored_public) -> HKDF-SHA256 -> AES-256-GCM key
  * The user decrypts with `scripts/decrypt_sample.py <password> <path>`
    which re-derives the private key from the password in the same way.

On-disk format for an encrypted sample blob:
    SAMPLE_MAGIC(8) | ephemeral_pubkey(32) | nonce(12) | AES-GCM ct(+16 tag)
SAMPLE_MAGIC is used as the GCM AAD so a corrupted/truncated blob fails
authentication.

Job configs (config.json) use the same public-key pattern as samples, but
with a keypair GENERATED in the browser (Settings -> Config Encryption) whose
private key is kept in the browser's localStorage (AITK_CONFIG_PRIVATE_KEY) -
it is NOT derived from any password.  Only that browser (or a CLI holding an
export of the private key, see scripts/decrypt_sample.py) can decrypt a
stored config:
    CONFIG_MAGIC(8) | ephemeral_pubkey(32) | nonce(12) | AES-GCM ct(+16 tag)
Must stay in sync with ui/src/utils/configKey.ts (magic/salt/info).
"""

import base64
import hashlib
import io
import os
from typing import Optional, Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from safetensors.torch import load as _st_load, save as _st_save, load_file as _st_load_file

# --------------------------------------------------------------------------- #
# Constants / errors
# --------------------------------------------------------------------------- #

MAGIC = b"AITKENC1"
_SALT_LEN = 16
_NONCE_LEN = 12
_HEADER = len(MAGIC) + _SALT_LEN + _NONCE_LEN  # 36 bytes


class DatasetPasswordRequiredError(Exception):
    """Raised when an encrypted dataset/cache is found but no password is set."""


class DatasetDecryptError(Exception):
    """Raised when decryption fails (wrong password / corrupt file)."""


# --------------------------------------------------------------------------- #
# Password source
# --------------------------------------------------------------------------- #

def get_dataset_password() -> Optional[str]:
    """Return the configured dataset password (env var set by the UI worker)."""
    pw = os.environ.get("DATASET_PASSWORD")
    if pw is None:
        return None
    pw = pw.strip()
    return pw if pw else None


# --------------------------------------------------------------------------- #
# Sample encryption (public-key scheme - the password never reaches the server)
# --------------------------------------------------------------------------- #

SAMPLE_MAGIC = b"AITKSAMP"
# Must match ui/src/utils/sampleKey.ts - PBKDF2 parameters used by the browser
# to derive the X25519 private key from the password.
SAMPLE_KEY_SALT = b"AITK-SAMPLE-KEY-SALT-v1"
SAMPLE_PBKDF2_ITERATIONS = 210_000
# HKDF info string binding the derived key to its purpose.
SAMPLE_HKDF_INFO = b"AITK-SAMPLE-V1"
_SAMPLE_HEADER = len(SAMPLE_MAGIC) + 32 + _NONCE_LEN  # 52 bytes


def get_sample_public_key() -> Optional[str]:
    """Return the configured sample-encryption public key (base64, env var set by the UI worker)."""
    key = os.environ.get("SAMPLE_PUBLIC_KEY")
    if key is None:
        return None
    key = key.strip()
    return key if key else None


def is_sample_encryption_enabled() -> bool:
    """True when generated samples should be encrypted at rest."""
    return get_sample_public_key() is not None


def derive_sample_private_key(password: str) -> X25519PrivateKey:
    """Derive the X25519 private key from the sample-encryption password.

    Mirrors the browser-side derivation (ui/src/utils/sampleKey.ts):
    PBKDF2-HMAC-SHA256(password, SAMPLE_KEY_SALT, 210k) -> 32-byte seed ->
    X25519 private key (both sides clamp the scalar identically).
    """
    seed = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), SAMPLE_KEY_SALT,
        SAMPLE_PBKDF2_ITERATIONS, dklen=32,
    )
    return X25519PrivateKey.from_private_bytes(seed)


def _x25519_shared_secret(private_key: X25519PrivateKey, peer_public: X25519PublicKey) -> bytes:
    """X25519 ECDH shared secret, compatible with old and new cryptography APIs."""
    # cryptography >= 42: private_key.exchange(peer_public)
    # older releases:      peer_public.exchange(private_key)
    if hasattr(private_key, "exchange"):
        return private_key.exchange(peer_public)
    return peer_public.exchange(private_key)


def encrypt_sample_bytes(data: bytes, public_key_b64: str) -> bytes:
    """Encrypt bytes with the stored sample public key (X25519 ECDH + AES-256-GCM).

    Format: SAMPLE_MAGIC | ephemeral_pub(32) | nonce(12) | ciphertext(+16 tag).
    """
    try:
        recipient_pub = X25519PublicKey.from_public_bytes(base64.b64decode(public_key_b64))
    except Exception as e:
        raise DatasetDecryptError(f"Invalid sample public key: {e}")

    ephemeral_priv = X25519PrivateKey.generate()
    shared = _x25519_shared_secret(ephemeral_priv, recipient_pub)
    aes_key = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None, info=SAMPLE_HKDF_INFO
    ).derive(shared)
    nonce = os.urandom(_NONCE_LEN)
    ct = AESGCM(aes_key).encrypt(nonce, data, SAMPLE_MAGIC)
    return SAMPLE_MAGIC + ephemeral_priv.public_key().public_bytes_raw() + nonce + ct


def decrypt_sample_bytes(blob: bytes, password: str) -> bytes:
    """Decrypt a sample blob with the password (re-derives the private key)."""
    if not is_sample_encrypted_blob(blob):
        raise DatasetDecryptError("File is not an AITK encrypted sample (wrong file or already decrypted?).")
    ephemeral_pub = X25519PublicKey.from_public_bytes(blob[len(SAMPLE_MAGIC):len(SAMPLE_MAGIC) + 32])
    nonce = blob[len(SAMPLE_MAGIC) + 32:_SAMPLE_HEADER]
    ct = blob[_SAMPLE_HEADER:]
    private_key = derive_sample_private_key(password)
    shared = _x25519_shared_secret(private_key, ephemeral_pub)
    aes_key = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None, info=SAMPLE_HKDF_INFO
    ).derive(shared)
    try:
        return AESGCM(aes_key).decrypt(nonce, ct, SAMPLE_MAGIC)
    except Exception as e:
        raise DatasetDecryptError(f"Sample decryption failed (wrong password or corrupt file): {e}")


def is_sample_encrypted_blob(data: bytes) -> bool:
    return data[:len(SAMPLE_MAGIC)] == SAMPLE_MAGIC


def peek_sample_encrypted(path: str) -> bool:
    """Cheap header check of whether a file is an encrypted sample."""
    try:
        with open(path, "rb") as f:
            return f.read(len(SAMPLE_MAGIC)) == SAMPLE_MAGIC
    except (FileNotFoundError, OSError):
        return False


def encrypt_sample_file_in_place(path: str, public_key_b64: Optional[str] = None) -> bool:
    """Encrypt the file at ``path`` in place using the stored sample public key.

    The file keeps its name/extension; its contents are replaced with the
    AITK encrypted sample blob. No-op (returns False) when no sample public
    key is configured or the file does not exist.
    """
    key = public_key_b64 if public_key_b64 is not None else get_sample_public_key()
    if not key:
        return False
    if not os.path.exists(path):
        return False
    with open(path, "rb") as f:
        data = f.read()
    with open(path, "wb") as f:
        f.write(encrypt_sample_bytes(data, key))
    return True


def decrypt_sample_file(path: str, password: Optional[str] = None, output_path: Optional[str] = None,
                        config_private_key: Optional[str] = None) -> str:
    """Decrypt an AITK encrypted file, IN PLACE by default. Returns the output
    path (== ``path`` unless ``output_path`` is given).

    Auto-detects the blob type (see :func:`decrypt_auto`): sample blobs
    (``AITKSAMP``), legacy password-encrypted blobs (``AITKENC1``) and config
    blobs (``AITKCFG1``, requires ``config_private_key``).
    """
    with open(path, "rb") as f:
        blob = f.read()
    plain = decrypt_auto(blob, password, config_private_key)
    if output_path is None:
        output_path = path
    with open(output_path, "wb") as f:
        f.write(plain)
    return output_path


# --------------------------------------------------------------------------- #
# Config encryption (public-key scheme - mirrors ui/src/utils/configKey.ts)
# --------------------------------------------------------------------------- #

CONFIG_MAGIC = b"AITKCFG1"
# Must match ui/src/utils/configKey.ts (browser-side config crypto).
CONFIG_HKDF_SALT = b"AITK-CFG-SALT-v1"
CONFIG_HKDF_INFO = b"AITK-CFG-KEY-v1"
_CONFIG_HEADER = len(CONFIG_MAGIC) + 32 + _NONCE_LEN  # 52 bytes


def is_config_encrypted_blob(data: bytes) -> bool:
    return data[:len(CONFIG_MAGIC)] == CONFIG_MAGIC


def peek_config_encrypted(path: str) -> bool:
    """Cheap header check of whether a file is an encrypted config blob."""
    try:
        with open(path, "rb") as f:
            return f.read(len(CONFIG_MAGIC)) == CONFIG_MAGIC
    except (FileNotFoundError, OSError):
        return False


def decrypt_config_bytes(blob: bytes, private_key_b64: str) -> bytes:
    """Decrypt an AITK config blob with the X25519 private key (base64, 32 bytes).

    Mirrors ui/src/utils/configKey.ts: the config keypair is generated in the
    browser (Settings -> Config Encryption) and the private key is stored in
    the browser's localStorage - it is NOT derived from a password.  Key:
    ECDH(priv, ephemeral_pub) -> HKDF-SHA256(salt, info) -> AES-256-GCM key.
    Format: CONFIG_MAGIC(8) | ephemeral_pub(32) | nonce(12) | AES-GCM ct(+16 tag).
    """
    if not is_config_encrypted_blob(blob):
        raise DatasetDecryptError("Blob is not an AITK encrypted config (missing AITKCFG1 magic).")
    if len(blob) < _CONFIG_HEADER + 16:
        raise DatasetDecryptError("Config blob is too small to be valid.")
    try:
        recipient_priv = X25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64))
        eph_pub = X25519PublicKey.from_public_bytes(blob[len(CONFIG_MAGIC):len(CONFIG_MAGIC) + 32])
    except Exception as e:
        raise DatasetDecryptError(f"Invalid config private key or blob: {e}")
    nonce = blob[len(CONFIG_MAGIC) + 32:_CONFIG_HEADER]
    ct = blob[_CONFIG_HEADER:]
    shared = _x25519_shared_secret(recipient_priv, eph_pub)
    aes_key = HKDF(
        algorithm=hashes.SHA256(), length=32,
        salt=CONFIG_HKDF_SALT, info=CONFIG_HKDF_INFO,
    ).derive(shared)
    try:
        return AESGCM(aes_key).decrypt(nonce, ct, CONFIG_MAGIC)
    except Exception as e:
        raise DatasetDecryptError(f"Config decryption failed (wrong private key or corrupt blob): {e}")


# --------------------------------------------------------------------------- #
# Unified blob detection / decryption (used by the CLI scripts)
# --------------------------------------------------------------------------- #

KNOWN_MAGICS = (MAGIC, SAMPLE_MAGIC, CONFIG_MAGIC)


def is_any_known_blob(data: bytes) -> bool:
    return any(data.startswith(m) for m in KNOWN_MAGICS)


def try_decode_base64_blob(data: bytes) -> Optional[bytes]:
    """If ``data`` is text that base64-encodes an AITK blob, return the decoded
    blob, else None.  Handles surrounding whitespace and quotes (e.g. a config
    blob saved from the DB as a ``config.json`` text file).
    """
    try:
        text = data.decode("utf-8").strip().strip('"').strip().strip("'").strip()
    except (UnicodeDecodeError, ValueError):
        return None
    if not text or len(text) < 40 or "\n" in text.strip('"').strip():
        return None
    try:
        decoded = base64.b64decode(text, validate=True)
    except Exception:
        return None
    if is_any_known_blob(decoded):
        return decoded
    return None


def decrypt_auto(blob: bytes, password: Optional[str] = None,
                 config_private_key: Optional[str] = None) -> bytes:
    """Detect the AITK blob type and decrypt it.

    Handles dataset/legacy blobs (``AITKENC1``, password), sample blobs
    (``AITKSAMP``, password) and config blobs (``AITKCFG1``, X25519 private
    key).  A base64-encoded text representation of one of these blobs is
    unwrapped first.  Raises :class:`DatasetDecryptError` / :class:`DatasetPasswordRequiredError`
    when the blob is not recognised or the needed secret is missing.
    """
    if not is_any_known_blob(blob):
        decoded = try_decode_base64_blob(blob)
        if decoded is None:
            raise DatasetDecryptError(
                "File is not an AITK encrypted blob (no AITKENC1/AITKSAMP/AITKCFG1 magic, "
                "and not base64 of one)."
            )
        blob = decoded
    if is_config_encrypted_blob(blob):
        if not config_private_key:
            raise DatasetDecryptError(
                "Config blob (AITKCFG1) needs the X25519 private key. Export it from the "
                "browser (devtools console: localStorage.getItem('AITK_CONFIG_PRIVATE_KEY')) "
                "and pass it via --config-key / CONFIG_PRIVATE_KEY."
            )
        return decrypt_config_bytes(blob, config_private_key)
    if not password:
        raise DatasetPasswordRequiredError(
            "This blob type (AITKENC1/AITKSAMP) is password-encrypted but no password was given."
        )
    if is_sample_encrypted_blob(blob):
        return decrypt_sample_bytes(blob, password)
    return decrypt_bytes(blob, password)


# --------------------------------------------------------------------------- #
# Core crypto
# --------------------------------------------------------------------------- #

def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=32, n=2 ** 14, r=8, p=1)
    return kdf.derive(password.encode("utf-8"))


def encrypt_bytes(plain: bytes, password: str) -> bytes:
    salt = os.urandom(_SALT_LEN)
    nonce = os.urandom(_NONCE_LEN)
    key = _derive_key(password, salt)
    ct = AESGCM(key).encrypt(nonce, plain, MAGIC)
    return MAGIC + salt + nonce + ct


def decrypt_bytes(blob: bytes, password: str) -> bytes:
    if not is_encrypted_blob(blob):
        raise DatasetDecryptError("Blob is not in the AITK encrypted format.")
    salt = blob[len(MAGIC):len(MAGIC) + _SALT_LEN]
    nonce = blob[len(MAGIC) + _SALT_LEN:_HEADER]
    ct = blob[_HEADER:]
    key = _derive_key(password, salt)
    try:
        return AESGCM(key).decrypt(nonce, ct, MAGIC)
    except Exception as e:  # InvalidTag or corrupt data
        raise DatasetDecryptError(f"Decryption failed (wrong password or corrupt file): {e}")


def is_encrypted_blob(data: bytes) -> bool:
    return data[:len(MAGIC)] == MAGIC


def peek_encrypted(path: str) -> bool:
    """Cheap check (reads only the header) of whether a file is encrypted."""
    try:
        with open(path, "rb") as f:
            head = f.read(len(MAGIC))
        return head == MAGIC
    except (FileNotFoundError, OSError):
        return False


# --------------------------------------------------------------------------- #
# Media "looks valid" checks (used to verify a password / detect encryption)
# --------------------------------------------------------------------------- #

def looks_like_image(data: bytes) -> bool:
    if len(data) < 12:
        return False
    if data[:8] == b"\x89PNG\r\n\x1a\n":            # png
        return True
    if data[:3] == b"\xff\xd8\xff":                 # jpeg
        return True
    if data[:2] == b"RI" and data[8:12] == b"WEBP":  # webp
        return True
    if data[:4] == b"ftyp" and data[8:12] in (b"jxl ", b"jxlc"):  # jxl container
        return True
    if data[:2] == b"\xff\x0a":                     # jxl codestream
        return True
    if data[:6] in (b"GIF87a", b"GIF89a"):          # gif
        return True
    if data[:2] == b"BM":                            # bmp
        return True
    return False


def looks_like_video(data: bytes) -> bool:
    if len(data) < 12:
        return False
    if data[4:8] == b"ftyp":                        # mp4 / mov / m4v
        return True
    if data[:4] == b"\x1a\x45\xdf\xa3":             # webm / mkv (EBML)
        return True
    if data[:4] == b"RIFF" and data[8:12] == b"AVI ":  # avi
        return True
    return False


def looks_like_media(data: bytes) -> bool:
    if looks_like_image(data) or looks_like_video(data):
        return True
    # fall back to "decodes as utf-8 text" for captions (txt/json)
    try:
        data[:1024].decode("utf-8")
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Video source abstraction (cv2 for plain files, PyAV for encrypted ones)
# --------------------------------------------------------------------------- #

class VideoSource:
    width = 0
    height = 0
    total_frames = 0
    fps = 24.0

    def is_opened(self) -> bool:
        return True

    def read_frame(self, idx: int):  # -> np.ndarray RGB or None
        raise NotImplementedError

    def release(self):
        pass


class Cv2VideoSource(VideoSource):
    def __init__(self, path: str):
        import cv2
        self._cv2 = cv2
        self.cap = cv2.VideoCapture(path)
        if self.cap.isOpened():
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 24)

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def read_frame(self, idx: int):
        import cv2
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        self.cap.release()


class PyAVVideoSource(VideoSource):
    """Decodes an in-memory (already decrypted) video using PyAV.

    Frames are decoded sequentially and cached so that the increasing
    frame indices requested by the dataloader are produced without rework.
    """

    def __init__(self, data: bytes):
        import av
        self.container = av.open(io.BytesIO(data))
        self.stream = self.container.streams.video[0]
        self.width = self.stream.width
        self.height = self.stream.height
        self.total_frames = int(self.stream.frames or 0)
        self.fps = float(self.stream.average_rate or 24)
        self._frames = []
        self._gen = self.container.decode(video=0)
        self._exhausted = False

    def read_frame(self, idx: int):
        while len(self._frames) <= idx and not self._exhausted:
            try:
                frame = next(self._gen)
            except StopIteration:
                self._exhausted = True
                break
            self._frames.append(frame.to_ndarray(format="rgb24"))
        return self._frames[idx] if idx < len(self._frames) else None

    def release(self):
        self._frames = []
        try:
            self.container.close()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# The DatasetFile wrapper
# --------------------------------------------------------------------------- #

class DatasetFile:
    """Wraps a single dataset file.

    For plain files every accessor is a thin passthrough to the on-disk path
    (no extra I/O, no behaviour change).  For encrypted files the plaintext is
    decrypted into RAM on first use and released via ``cleanup()``.
    """

    def __init__(self, path: str):
        self.path = path
        self.is_encrypted = peek_encrypted(path)
        self._plain: Optional[bytes] = None

    def _ensure_plain(self) -> bytes:
        if self._plain is None:
            pw = get_dataset_password()
            if not pw:
                raise DatasetPasswordRequiredError(
                    "Encrypted dataset file found but no dataset password is set."
                )
            with open(self.path, "rb") as f:
                data = f.read()
            self._plain = decrypt_bytes(data, pw)
        return self._plain

    # -- raw bytes / text -------------------------------------------------- #
    def read_bytes(self) -> bytes:
        if self.is_encrypted:
            return self._ensure_plain()
        with open(self.path, "rb") as f:
            return f.read()

    def read_text(self) -> str:
        return self.read_bytes().decode("utf-8")

    # -- images ------------------------------------------------------------ #
    def open_image(self):
        from PIL import Image
        from PIL.ImageOps import exif_transpose
        if self.is_encrypted:
            img = Image.open(io.BytesIO(self._ensure_plain()))
        else:
            img = Image.open(self.path)
        return exif_transpose(img)

    def image_size(self) -> Tuple[int, int]:
        from toolkit import image_utils
        if self.is_encrypted:
            data = self._ensure_plain()
            return image_utils.get_image_size_from_bytesio(io.BytesIO(data), len(data))
        return image_utils.get_image_size(self.path)

    # -- video ------------------------------------------------------------- #
    def open_video(self) -> VideoSource:
        if self.is_encrypted:
            return PyAVVideoSource(self._ensure_plain())
        return Cv2VideoSource(self.path)

    # -- audio ------------------------------------------------------------- #
    def open_audio(self):
        import torchaudio
        if self.is_encrypted:
            return torchaudio.load(io.BytesIO(self._ensure_plain()))
        return torchaudio.load(self.path)

    # -- av container (duration probing etc.) ------------------------------ #
    def open_av(self):
        import av
        if self.is_encrypted:
            return av.open(io.BytesIO(self._ensure_plain()))
        return av.open(self.path)

    # -- safetensors caches ------------------------------------------------ #
    def safetensors_load(self, device="cpu"):
        if self.is_encrypted:
            tensors = _st_load(self._ensure_plain())  # loads on CPU
            if device and device != "cpu":
                tensors = {k: v.to(device) for k, v in tensors.items()}
            return tensors
        return _st_load_file(self.path, device=device)

    def cleanup(self):
        self._plain = None


def open_dataset(path: str) -> DatasetFile:
    return DatasetFile(path)


def read_text_file(path: str) -> str:
    """Read a (possibly encrypted) text file into a string, in RAM only."""
    df = open_dataset(path)
    try:
        return df.read_text()
    finally:
        df.cleanup()


def open_image(path: str):
    """Open an (optionally encrypted) image file, EXIF-transposed.

    Encrypted files are decrypted into RAM only; the returned image is safe
    to keep using afterwards (PIL holds its own copy of the decrypted bytes
    for lazy decoding, and plain files keep their own file handle).
    """
    df = open_dataset(path)
    try:
        return df.open_image()
    finally:
        df.cleanup()


# --------------------------------------------------------------------------- #
# Safetensors cache save / load (encrypt when a password is configured)
# --------------------------------------------------------------------------- #

def save_safetensors(state_dict, path: str, metadata: Optional[dict] = None):
    data = _st_save(state_dict, metadata=metadata)
    pw = get_dataset_password()
    if pw:
        data = encrypt_bytes(data, pw)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def load_safetensors(path: str, device="cpu"):
    if peek_encrypted(path):
        pw = get_dataset_password()
        if not pw:
            raise DatasetPasswordRequiredError(
                "Encrypted cache file found but no dataset password is set."
            )
        with open(path, "rb") as f:
            data = f.read()
        tensors = _st_load(decrypt_bytes(data, pw))  # loads on CPU
        if device and device != "cpu":
            tensors = {k: v.to(device) for k, v in tensors.items()}
        return tensors
    return _st_load_file(path, device=device)


def _parse_safetensors_metadata(data: bytes):
    """Read the `__metadata__` field from a (decrypted) safetensors blob.

    The safetensors header is: <u64 little-endian length N><N bytes of JSON>.
    Metadata is stored in the JSON header under the `__metadata__` key.
    """
    import json
    import struct
    try:
        if len(data) < 8:
            return None
        n = struct.unpack("<Q", data[:8])[0]
        if n == 0 or 8 + n > len(data):
            return None
        header = json.loads(data[8:8 + n].decode("utf-8"))
        return header.get("__metadata__")
    except Exception:
        return None


def load_safetensors_meta(path: str, device="cpu"):
    """Like :func:`load_safetensors` but also returns the file metadata.

    Returns ``(state_dict, metadata)`` where ``metadata`` is a dict or ``None``.
    """
    if peek_encrypted(path):
        pw = get_dataset_password()
        if not pw:
            raise DatasetPasswordRequiredError(
                "Encrypted cache file found but no dataset password is set."
            )
        with open(path, "rb") as f:
            data = f.read()
        plain = decrypt_bytes(data, pw)
        tensors = _st_load(plain)  # loads on CPU
        if device and device != "cpu":
            tensors = {k: v.to(device) for k, v in tensors.items()}
        return tensors, _parse_safetensors_metadata(plain)
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        meta = f.metadata()
    tensors = _st_load_file(path, device=device)
    return tensors, meta


# --------------------------------------------------------------------------- #
# Early dataset validation (fail fast before doing any work)
# --------------------------------------------------------------------------- #

def validate_dataset(file_paths, log=print) -> None:
    """Inspect a representative file and raise if the dataset is encrypted but
    the password is missing / wrong.  No-op for plain datasets.
    """
    sample = None
    candidates = list(file_paths)
    for p in candidates:
        if isinstance(p, str) and os.path.isfile(p):
            sample = p
            break
    if sample is None:
        return

    if not peek_encrypted(sample):
        return  # plain dataset -> nothing to do

    pw = get_dataset_password()
    if not pw:
        raise DatasetPasswordRequiredError(
            "The dataset files are encrypted but no dataset password is set. "
            "Set the 'Dataset Password' in Settings and re-run the job."
        )

    try:
        with open(sample, "rb") as f:
            data = f.read()
        plain = decrypt_bytes(data, pw)
    except DatasetDecryptError:
        raise DatasetDecryptError(
            "The dataset is encrypted but the configured password is incorrect."
        )

    if not looks_like_media(plain):
        raise DatasetDecryptError(
            "Decryption did not produce valid media; the dataset password is likely incorrect."
        )
    log("Detected encrypted dataset - decrypting items in RAM (nothing is written to disk).")
