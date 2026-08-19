/*
 * Startup schema self-heal for the SQLite DB (aitk_db.db).
 *
 * The schema source of truth is prisma/schema.prisma, applied with
 * `prisma db push` (see `npm run update_db`).  This module covers the common
 * case of an EXISTING installation whose aitek_db.db predates a schema
 * addition, so that `npm run dev` / `npm start` work without a manual
 * `prisma db push`:
 *
 *   - Job.job_config_encrypted (TEXT, nullable) - encrypted config blob
 *
 * Missing nullable columns are added with plain `ALTER TABLE ... ADD COLUMN`
 * (safe on SQLite).  A fresh install (DB/tables not created yet) is left
 * alone - `prisma db push` creates the full schema there.
 *
 * Used by both processes that talk to the DB:
 *   - ui/server.js      (plain Node, required directly)
 *   - ui/cron/worker.ts (required at startup; `npm run build` copies
 *                        ui/db/ into dist/db/ so the same file resolves
 *                        from dist/cron/worker.js)
 */
const { PrismaClient } = require('@prisma/client');

// One entry per added column. Keep statements in sync with prisma/schema.prisma.
const COLUMN_CHECKS = [
  {
    table: 'Job',
    column: 'job_config_encrypted',
    statement: 'ALTER TABLE "Job" ADD COLUMN "job_config_encrypted" TEXT',
  },
];

async function ensureSchema() {
  const prisma = new PrismaClient();
  try {
    for (const check of COLUMN_CHECKS) {
      const tables = await prisma.$queryRawUnsafe(
        `SELECT name FROM sqlite_master WHERE type = 'table' AND name = '${check.table}'`
      );
      if (!tables || tables.length === 0) {
        // Fresh DB - do not touch it, `npm run update_db` creates the schema.
        console.warn(
          `[schema] table "${check.table}" not found - run "npm run update_db" (prisma db push) to create the schema`
        );
        return;
      }
      const cols = await prisma.$queryRawUnsafe(`PRAGMA table_info('${check.table}')`);
      const exists = Array.isArray(cols) && cols.some((c) => c.name === check.column);
      if (!exists) {
        await prisma.$executeUnsafe(check.statement);
        console.log(`[schema] added missing column ${check.table}.${check.column}`);
      }
    }
  } catch (e) {
    // Never block startup on a self-heal failure; the usual prisma error
    // will surface on the first real query if the DB is broken.
    console.error('[schema] ensureSchema failed:', e.message);
  } finally {
    try { await prisma.$disconnect(); } catch {}
  }
}

module.exports = { ensureSchema };
