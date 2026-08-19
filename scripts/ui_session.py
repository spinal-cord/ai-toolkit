#!/usr/bin/env python3
"""
Get an AI Toolkit UI session token without a browser.

Performs the same Ed25519 challenge-response login the web UI uses: the
password is only used locally to derive the private key and is NEVER sent
to the server (only a signature over a server-issued one-time challenge is
transmitted).

Usage:
    python scripts/ui_session.py <password> [base_url]
    # default base_url: http://localhost:8675

Prints the session token. Use it with curl either as a cookie or as a
Bearer token:
    curl --cookie "AITK_SESSION=<token>" http://localhost:8675/api/jobs
    curl -H "Authorization: Bearer <token>"  http://localhost:8675/api/jobs

The token is valid for 30 days (or until the server process restarts).
"""

import base64
import hashlib
import json
import sys
import urllib.error
import urllib.request

# Must match ui/src/utils/authKey.ts (browser-side derivation)
SALT = b"AITK-AUTH-KEY-SALT-v1"
ITERATIONS = 210_000
DEFAULT_BASE_URL = "http://localhost:8675"
SESSION_COOKIE = "AITK_SESSION"


def http_json(method: str, url: str, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        method=method,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8")), resp.headers.get("Set-Cookie", "")


def main():
    if len(sys.argv) < 2 or not sys.argv[1]:
        print(__doc__)
        sys.exit(1)

    password = sys.argv[1]
    base_url = (sys.argv[2] if len(sys.argv) > 2 else DEFAULT_BASE_URL).rstrip("/")

    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError:
        print("Error: the 'cryptography' package is required (pip install cryptography).")
        sys.exit(1)

    # 1) fetch a one-time challenge
    try:
        challenge_data, _ = http_json("GET", f"{base_url}/api/auth/challenge")
    except urllib.error.HTTPError as e:
        print(f"Error: could not reach {base_url}/api/auth/challenge (HTTP {e.code}).")
        sys.exit(1)
    except Exception as e:
        print(f"Error: could not reach {base_url}: {e}")
        sys.exit(1)

    challenge = challenge_data.get("challenge", "")
    if not challenge:
        print("Error: server did not return a challenge.")
        sys.exit(1)

    # 2) derive the private key locally and sign the challenge
    seed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), SALT, ITERATIONS, dklen=32)
    private_key = Ed25519PrivateKey.from_private_bytes(seed)
    signature = private_key.sign(bytes.fromhex(challenge))

    # 3) log in with the signature (the password itself is never sent)
    try:
        _, set_cookie = http_json(
            "POST",
            f"{base_url}/api/auth/login",
            {"challenge": challenge, "signature": base64.b64encode(signature).decode("ascii")},
        )
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        if e.code == 401:
            print("Error: invalid password.")
        else:
            print(f"Error: login failed (HTTP {e.code}) {detail}")
        sys.exit(1)

    token = ""
    for part in set_cookie.split(";"):
        part = part.strip()
        if part.startswith(f"{SESSION_COOKIE}="):
            token = part[len(f"{SESSION_COOKIE}="):]
            break

    if not token:
        print("Error: login succeeded but no session cookie was returned.")
        sys.exit(1)

    print(token)
    print(f"\nUse it with curl, e.g.:\n"
          f'  curl --cookie "{SESSION_COOKIE}={token}" {base_url}/api/jobs\n'
          f'  curl -H "Authorization: Bearer {token}" {base_url}/api/jobs', file=sys.stderr)


if __name__ == "__main__":
    main()
