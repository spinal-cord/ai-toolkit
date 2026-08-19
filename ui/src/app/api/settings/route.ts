import { NextResponse } from 'next/server';
import { PrismaClient } from '@prisma/client';
import { defaultTrainFolder, defaultDatasetsFolder } from '@/paths';
import { flushCache } from '@/server/settings';
import { isValidEd25519PublicKey } from '@/server/auth';

const prisma = new PrismaClient();

export async function GET() {
  try {
    const settings = await prisma.settings.findMany();
    const settingsObject = settings.reduce((acc: any, setting) => {
      acc[setting.key] = setting.value;
      return acc;
    }, {});
    // if TRAINING_FOLDER is not set, use default
    if (!settingsObject.TRAINING_FOLDER || settingsObject.TRAINING_FOLDER === '') {
      settingsObject.TRAINING_FOLDER = defaultTrainFolder;
    }
    // if DATASETS_FOLDER is not set, use default
    if (!settingsObject.DATASETS_FOLDER || settingsObject.DATASETS_FOLDER === '') {
      settingsObject.DATASETS_FOLDER = defaultDatasetsFolder;
    }
    // never expose the auth public key or legacy hash to the client
    settingsObject.AUTH_PASSWORD_SET = Boolean(
      settingsObject.AUTH_PUBLIC_KEY || settingsObject.AUTH_PASSWORD_HASH,
    );
    delete settingsObject.AUTH_PUBLIC_KEY;
    delete settingsObject.AUTH_PASSWORD_HASH;
    // never expose the sample encryption public key to the client
    // (the password it was derived from is known only to the user's browser)
    settingsObject.SAMPLE_PASSWORD_SET = Boolean(settingsObject.SAMPLE_PUBLIC_KEY);
    delete settingsObject.SAMPLE_PUBLIC_KEY;
    // Dataset password is symmetric (the plaintext must reach the training
    // worker as the DATASET_PASSWORD env var), so it is stored in Settings -
    // but never returned to the client, only a "is set" flag.
    settingsObject.DATASET_PASSWORD_SET = Boolean(
      typeof settingsObject.DATASET_PASSWORD === 'string' && settingsObject.DATASET_PASSWORD.trim() !== '',
    );
    delete settingsObject.DATASET_PASSWORD;
    // Config-encryption public key IS safe to return - it is a public key and
    // the webui needs it to encrypt configs. The private key never leaves the
    // browser.
    settingsObject.CONFIG_ENCRYPTION_SET = Boolean(settingsObject.CONFIG_PUBLIC_KEY);
    settingsObject.CONFIG_PUBLIC_KEY = settingsObject.CONFIG_PUBLIC_KEY || '';
    return NextResponse.json(settingsObject);
  } catch (error) {
    return NextResponse.json({ error: 'Failed to fetch settings' }, { status: 500 });
  }
}

export async function POST(request: Request) {
  try {
    const body = await request.json();
    const { HF_TOKEN, TRAINING_FOLDER, DATASETS_FOLDER, AUTH_PUBLIC_KEY, TLS_CERT, TLS_KEY, CLEAR_AUTH, SAMPLE_PUBLIC_KEY, CLEAR_SAMPLE_PASSWORD, CONFIG_PUBLIC_KEY, CLEAR_CONFIG_ENCRYPTION, DATASET_PASSWORD, CLEAR_DATASET_PASSWORD } = body;

    const upserts: Promise<any>[] = [];
    const plainKeys: [string, string | undefined][] = [
      ['HF_TOKEN', HF_TOKEN],
      ['TRAINING_FOLDER', TRAINING_FOLDER],
      ['DATASETS_FOLDER', DATASETS_FOLDER],
      ['TLS_CERT', TLS_CERT],
      ['TLS_KEY', TLS_KEY],
    ];
    for (const [key, value] of plainKeys) {
      if (typeof value === 'string') {
        upserts.push(
          prisma.settings.upsert({
            where: { key },
            update: { value },
            create: { key, value },
          }),
        );
      }
    }

    // UI-password handling. The browser derives an Ed25519 keypair from the
    // user's password and sends ONLY the 32-byte public key (base64) - the
    // password itself is never transmitted or stored:
    //  - CLEAR_AUTH=true            -> remove the password (UI open again)
    //  - AUTH_PUBLIC_KEY valid      -> (re)set the public key
    //  - AUTH_PUBLIC_KEY empty      -> keep the current password
    if (CLEAR_AUTH === true) {
      upserts.push(prisma.settings.deleteMany({ where: { key: 'AUTH_PUBLIC_KEY' } }));
      upserts.push(prisma.settings.deleteMany({ where: { key: 'AUTH_PASSWORD_HASH' } }));
    } else if (typeof AUTH_PUBLIC_KEY === 'string' && AUTH_PUBLIC_KEY.length > 0) {
      if (!isValidEd25519PublicKey(AUTH_PUBLIC_KEY)) {
        return NextResponse.json(
          { error: 'Invalid UI password key (expected a 32-byte Ed25519 public key in base64)' },
          { status: 400 },
        );
      }
      upserts.push(
        prisma.settings.upsert({
          where: { key: 'AUTH_PUBLIC_KEY' },
          update: { value: AUTH_PUBLIC_KEY },
          create: { key: 'AUTH_PUBLIC_KEY', value: AUTH_PUBLIC_KEY },
        }),
      );
      // The public key fully replaces the legacy scrypt hash (migration).
      upserts.push(prisma.settings.deleteMany({ where: { key: 'AUTH_PASSWORD_HASH' } }));
    }

    // Sample-encryption handling. The browser derives an X25519 keypair from
    // the user's password and sends ONLY the 32-byte public key (base64) -
    // the password itself is never transmitted or stored:
    //  - CLEAR_SAMPLE_PASSWORD=true   -> remove the key (samples stored plain)
    //  - SAMPLE_PUBLIC_KEY valid      -> (re)set the public key
    //  - otherwise                    -> keep the current key
    if (CLEAR_SAMPLE_PASSWORD === true) {
      upserts.push(prisma.settings.deleteMany({ where: { key: 'SAMPLE_PUBLIC_KEY' } }));
    } else if (typeof SAMPLE_PUBLIC_KEY === 'string' && SAMPLE_PUBLIC_KEY.length > 0) {
      // validate: base64-decodable and exactly 32 bytes (X25519 public key)
      let keyBytes: Buffer;
      try {
        keyBytes = Buffer.from(SAMPLE_PUBLIC_KEY, 'base64');
      } catch {
        keyBytes = Buffer.alloc(0);
      }
      if (keyBytes.length !== 32) {
        return NextResponse.json(
          { error: 'Invalid sample encryption key (expected a 32-byte X25519 public key in base64)' },
          { status: 400 },
        );
      }
      upserts.push(
        prisma.settings.upsert({
          where: { key: 'SAMPLE_PUBLIC_KEY' },
          update: { value: SAMPLE_PUBLIC_KEY },
          create: { key: 'SAMPLE_PUBLIC_KEY', value: SAMPLE_PUBLIC_KEY },
        }),
      );
    }

    // Config-encryption handling. A X25519 keypair is generated in the browser
    // (Settings -> Config Encryption); only the 32-byte public key (base64) is
    // stored on the server. The private key stays in the browser (localStorage)
    // so only that browser can decrypt stored configs.
    //  - CLEAR_CONFIG_ENCRYPTION=true -> remove the public key
    //  - CONFIG_PUBLIC_KEY valid      -> (re)set the public key
    //  - otherwise                    -> keep the current key
    if (CLEAR_CONFIG_ENCRYPTION === true) {
      upserts.push(prisma.settings.deleteMany({ where: { key: 'CONFIG_PUBLIC_KEY' } }));
    } else if (typeof CONFIG_PUBLIC_KEY === 'string' && CONFIG_PUBLIC_KEY.length > 0) {
      let keyBytes: Buffer;
      try {
        keyBytes = Buffer.from(CONFIG_PUBLIC_KEY, 'base64');
      } catch {
        keyBytes = Buffer.alloc(0);
      }
      if (keyBytes.length !== 32) {
        return NextResponse.json(
          { error: 'Invalid config encryption key (expected a 32-byte X25519 public key in base64)' },
          { status: 400 },
        );
      }
      upserts.push(
        prisma.settings.upsert({
          where: { key: 'CONFIG_PUBLIC_KEY' },
          update: { value: CONFIG_PUBLIC_KEY },
          create: { key: 'CONFIG_PUBLIC_KEY', value: CONFIG_PUBLIC_KEY },
        }),
      );
    }

    // Dataset-password handling. Unlike the UI/sample passwords (public-key
    // schemes), this is the actual symmetric password the training worker
    // uses to decrypt datasets, so it is stored as-is:
    //  - CLEAR_DATASET_PASSWORD=true -> remove it (datasets must be plain)
    //  - non-empty string            -> (re)set it
    //  - empty / missing             -> keep the current one
    if (CLEAR_DATASET_PASSWORD === true) {
      upserts.push(prisma.settings.deleteMany({ where: { key: 'DATASET_PASSWORD' } }));
    } else if (typeof DATASET_PASSWORD === 'string' && DATASET_PASSWORD.trim() !== '') {
      upserts.push(
        prisma.settings.upsert({
          where: { key: 'DATASET_PASSWORD' },
          update: { value: DATASET_PASSWORD },
          create: { key: 'DATASET_PASSWORD', value: DATASET_PASSWORD },
        }),
      );
    }

    await Promise.all(upserts);

    flushCache();

    return NextResponse.json({ success: true });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to update settings' }, { status: 500 });
  }
}
