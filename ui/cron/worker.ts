import processQueue from './actions/processQueue';

// Self-heal the SQLite schema on startup (adds e.g. Job.job_config_encrypted
// to pre-existing DBs so no manual `prisma db push` is needed; see
// db/ensureSchema.js). Plain JS so it can be shared with server.js.
/* eslint-disable @typescript-eslint/no-var-requires */
const { ensureSchema } = require('../db/ensureSchema');

class CronWorker {
  interval: number;
  is_running: boolean;
  intervalId: NodeJS.Timeout;
  constructor() {
    this.interval = 1000; // Default interval of 1 second
    this.is_running = false;
    ensureSchema().catch(() => {}); // fire-and-forget; never blocks the worker
    this.intervalId = setInterval(() => {
      this.run();
    }, this.interval);
  }
  async run() {
    if (this.is_running) {
      return;
    }
    this.is_running = true;
    try {
      // Loop logic here
      await this.loop();
    } catch (error) {
      console.error('Error in cron worker loop:', error);
    }
    this.is_running = false;
  }

  async loop() {
    await processQueue();
  }
}

// it automatically starts the loop
const cronWorker = new CronWorker();
console.log('Cron worker started with interval:', cronWorker.interval, 'ms');
