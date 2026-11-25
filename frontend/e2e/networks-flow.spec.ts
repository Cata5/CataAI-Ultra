import { expect, test } from "@playwright/test";

test("save network -> start run flow", async ({ request }) => {
  const netId = `e2e_net_${Date.now()}`;
  let runId: string | undefined;
  let saved = false;
  try {
    const response = await request.post("/api/networks", {
      data: {
        id: netId,
        name: "E2E Network",
        d_model: 32,
        n_layers: 1,
        n_q: 2,
        n_kv: 1,
        d_h: 16,
        d_ff: 64,
      },
    });
    expect(response.ok()).toBeTruthy();
    saved = true;

    const started = await request.post("/api/runs", {
      data: { name: "e2e-network-run", network_id: netId },
    });
    expect(started.ok()).toBeTruthy();
    const body = await started.json();
    expect(body.status).toBe("started");
    runId = body.payload?.run_id || body.payload?.id;
    expect(runId).toBeTruthy();
  } finally {
    if (runId) {
      await request.post(`/api/runs/${runId}/stop`).catch(() => undefined);
    }
    if (saved) {
      await request.delete(`/api/networks/${netId}`).catch(() => undefined);
    }
  }
});

