import { defineConfig } from "@playwright/test";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const repoRoot = resolve(process.cwd(), "..");
const venvPython = join(repoRoot, ".venv", "Scripts", "python.exe");
const python = process.env.PYTHON || (existsSync(venvPython) ? venvPython : "python");
const dataDir =
  process.env.LLMMAKER_DATA_DIR || mkdtempSync(join(tmpdir(), "llm-maker-e2e-"));

export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  fullyParallel: false,
  workers: 1,
  use: {
    baseURL: process.env.PLAYWRIGHT_BASE_URL || "http://127.0.0.1:8081",
    headless: true,
  },
  webServer: {
    // The API-only specs need the real Python backend. Starting it here keeps
    // `npm run test:e2e` a single command and prevents tests from silently
    // hitting an unrelated server.
    command: `"${python}" dashboard.py --host 127.0.0.1 --port 8081`,
    cwd: repoRoot,
    url: "http://127.0.0.1:8081/api/system/status",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
    env: {
      ...process.env,
      LLMMAKER_DATA_DIR: dataDir,
      PYTHONUNBUFFERED: "1",
    },
  },
});
