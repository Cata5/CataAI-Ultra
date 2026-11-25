import { expect, test } from "@playwright/test";

test("upload dataset -> tokenize flow", async ({ request }) => {
  let datasetId: string | undefined;
  try {
    const upload = await request.post("/api/datasets/upload", {
      multipart: {
        file: {
          name: "e2e_upload.jsonl",
          mimeType: "application/jsonl",
          buffer: Buffer.from(
            JSON.stringify({ instruction: "Say hello", input: "", output: "Hello from e2e" }) + "\n",
            "utf8"
          ),
        },
      },
    });
    expect(upload.ok()).toBeTruthy();
    const up = await upload.json();
    expect(up.success).toBeTruthy();

    datasetId = up.dataset_id || up.id;
    expect(datasetId).toBeTruthy();

    const tokenize = await request.post(`/api/datasets/${datasetId}/tokenize`);
    expect(tokenize.ok()).toBeTruthy();
    const result = await tokenize.json();
    expect(result.success).toBeTruthy();
  } finally {
    if (datasetId) {
      await request.delete(`/api/datasets/${datasetId}`).catch(() => undefined);
    }
  }
});

