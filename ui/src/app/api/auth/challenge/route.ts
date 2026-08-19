import { NextResponse } from 'next/server';
import { issueChallenge } from '@/server/auth';

export const dynamic = 'force-dynamic';

// GET /api/auth/challenge -> { challenge }
// Issues a one-time 32-byte challenge (hex, 60s TTL, single-use) that the
// client signs with the Ed25519 private key derived from the password.
// Public route (under the /api/auth prefix) - it reveals no secret.
export async function GET() {
  return NextResponse.json({ challenge: issueChallenge() });
}
