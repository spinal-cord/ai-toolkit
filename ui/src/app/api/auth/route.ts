import { NextRequest, NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { SESSION_COOKIE, verifySessionToken } from '@/server/auth';

const prisma = new PrismaClient();

export const dynamic = 'force-dynamic';

// GET /api/auth -> { required, authenticated, mode }
// required:      a password is configured in Settings (public key or legacy hash)
// authenticated: a valid session cookie is present
// mode:          'publickey' - login via Ed25519 challenge-response (password
//                        is never transmitted)
//                'legacy'    - old scrypt-hash install; plaintext password
//                        login accepted for migration (re-save the password
//                        in Settings to migrate to public-key mode)
//                'none'      - no password configured (first-launch mode)
export async function GET(request: NextRequest) {
  try {
    const [pkRow, hashRow] = await Promise.all([
      prisma.settings.findFirst({ where: { key: 'AUTH_PUBLIC_KEY' } }),
      prisma.settings.findFirst({ where: { key: 'AUTH_PASSWORD_HASH' } }),
    ]);
    const required = Boolean(pkRow?.value || hashRow?.value);
    const mode: 'publickey' | 'legacy' | 'none' = pkRow?.value
      ? 'publickey'
      : hashRow?.value
        ? 'legacy'
        : 'none';
    const cookie = request.cookies.get(SESSION_COOKIE)?.value;
    const authenticated = required && verifySessionToken(cookie);
    return NextResponse.json({ required, authenticated, mode });
  } catch (error) {
    console.error('Auth status failed:', error);
    return NextResponse.json({ required: false, authenticated: true, mode: 'none' });
  }
}
