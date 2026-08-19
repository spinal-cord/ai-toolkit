import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import {
  SESSION_COOKIE,
  SESSION_TTL_SECONDS,
  signSessionToken,
  verifyPassword,
  verifyEd25519Signature,
  consumeChallenge,
} from '@/server/auth';

const prisma = new PrismaClient();

export const dynamic = 'force-dynamic';

// POST /api/auth/login
//
// Public-key mode (AUTH_PUBLIC_KEY stored):
//   { "challenge": "<hex from /api/auth/challenge>", "signature": "<base64>" }
// Proves knowledge of the password WITHOUT transmitting it: the signature is
// verified against the stored Ed25519 public key.
//
// Legacy migration mode (only AUTH_PASSWORD_HASH stored, no public key):
//   { "password": "..." }
// Accepted so old installs can still log in; re-saving the password in
// Settings stores the public key and deletes the hash.
//
// On success an HttpOnly session cookie (AITK_SESSION) is issued.
export async function POST(request: Request) {
  try {
    const body = await request.json();
    const [pkRow, hashRow] = await Promise.all([
      prisma.settings.findFirst({ where: { key: 'AUTH_PUBLIC_KEY' } }),
      prisma.settings.findFirst({ where: { key: 'AUTH_PASSWORD_HASH' } }),
    ]);

    if (pkRow?.value) {
      const challenge = typeof body.challenge === 'string' ? body.challenge : '';
      const signature = typeof body.signature === 'string' ? body.signature : '';
      if (!challenge || !signature) {
        return NextResponse.json({ error: 'Missing challenge or signature' }, { status: 400 });
      }
      // Consume BEFORE verifying so a captured (challenge, signature) pair
      // can never be replayed to mint another session.
      if (!consumeChallenge(challenge)) {
        return NextResponse.json(
          { error: 'Challenge expired or already used - please try again' },
          { status: 400 },
        );
      }
      if (!verifyEd25519Signature(pkRow.value, challenge, signature)) {
        return NextResponse.json({ error: 'Invalid password' }, { status: 401 });
      }
      return issueSession();
    }

    if (hashRow?.value) {
      const password = typeof body.password === 'string' ? body.password : '';
      if (!verifyPassword(password, hashRow.value)) {
        return NextResponse.json({ error: 'Invalid password' }, { status: 401 });
      }
      return issueSession();
    }

    return NextResponse.json({ error: 'No password configured' }, { status: 400 });
  } catch (error) {
    console.error('Login failed:', error);
    return NextResponse.json({ error: 'Login failed' }, { status: 500 });
  }
}

function issueSession() {
  const token = signSessionToken();
  const res = NextResponse.json({ ok: true });
  res.cookies.set(SESSION_COOKIE, token, {
    httpOnly: true,
    path: '/',
    sameSite: 'lax',
    maxAge: SESSION_TTL_SECONDS,
  });
  return res;
}
