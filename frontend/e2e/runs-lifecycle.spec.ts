import { expect, test } from "@playwright/test";

test("start -> stop run lifecycle", async ({ request }) => {
  let runId: string | undefined;
  try {
    const started = await request.post("/api/runs", {
      data: { name: "e2e-start-stop" },
    });
    expect(started.ok()).toBeTruthy();
    const body = await started.json();
    expect(body.status).toBe("started");

    runId = body.payload?.run_id || body.payload?.id;
    expect(runId).toBeTruthy();

    const stop = await request.post(`/api/runs/${runId}/stop`);
    expect(stop.ok()).toBeTruthy();
    expect((await stop.json()).success).toBeTruthy();
  } finally {
    if (runId) {
      await request.post(`/api/runs/${runId}/stop`).catch(() => undefined);
    }
  }
});

