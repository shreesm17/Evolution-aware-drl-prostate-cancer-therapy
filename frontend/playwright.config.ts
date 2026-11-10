import { defineConfig } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  timeout: 90000,
  workers: 1,
  use: {
    baseURL: process.env.ONCOEVO_URL || "http://127.0.0.1:5173",
    viewport: { width: 1600, height: 1000 },
    trace: "retain-on-failure",
  },
  webServer: process.env.ONCOEVO_URL ? undefined : [
    {
      command: "python -m uvicorn src.api:app --host 127.0.0.1 --port 8000",
      cwd: "..",
      url: "http://127.0.0.1:8000/openapi.json",
      reuseExistingServer: true,
      timeout: 60000,
    },
    {
      command: "npm run dev",
      url: "http://127.0.0.1:5173",
      reuseExistingServer: true,
    },
  ],
});
