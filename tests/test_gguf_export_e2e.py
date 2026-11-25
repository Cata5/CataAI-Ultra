"""End-to-end test for the GGUF export API handler (Dashboard v2)."""

import os
import shutil
import tempfile
from pathlib import Path

import torch
from aiohttp.test_utils import AioHTTPTestCase

from src.dashboard.server import create_app
from src.utils.paths import models_dir


class TestGgufExportEndToEnd(AioHTTPTestCase):
    """GET /api/models/{id}/export_gguf against a real saved checkpoint."""

    async def setUpAsync(self):
        # Isolate generated artifacts under a temp data dir (LLMMAKER_DATA_DIR).
        self._data_dir = tempfile.mkdtemp(prefix="llm-maker-gguf-e2e-")
        os.environ["LLMMAKER_DATA_DIR"] = self._data_dir
        await super().setUpAsync()

    async def tearDownAsync(self):
        os.environ.pop("LLMMAKER_DATA_DIR", None)
        await super().tearDownAsync()
        shutil.rmtree(self._data_dir, ignore_errors=True)

    async def get_application(self):
        return create_app()

    def _save_model(self, model_id: str) -> Path:
        from src.model.architecture import RoCanonConfig, RoCanonForCausalLM

        cfg = RoCanonConfig(
            # A runnable Llama GGUF needs the embedding rows and tokenizer vocab
            # to match; the project tokenizer has 12,288 tokens.
            vocab_size=12288,
            d_model=32,
            n_layers=1,
            n_q=2,
            n_kv=1,
            d_h=16,
            d_ff=64,
            max_seq_len=32,
            canon_enabled=False,
            qk_norm=False,
            residual_scale=False,
            architecture="llama",
        )
        model = RoCanonForCausalLM(cfg)
        folder = models_dir() / model_id
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint = folder / f"{model_id}_stripped.pt"
        torch.save({"model_state_dict": model.state_dict()}, checkpoint)
        return folder

    async def test_export_gguf_creates_file_and_manifest(self):
        self._save_model("gguf_e2e_model")
        resp = await self.client.get(
            "/api/models/gguf_e2e_model/export_gguf",
            params={"quant": "F16", "arch": "auto"},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["success"] is True
        assert data["architecture"] == "llama"
        assert data["quantization"] == "F16"
        assert data["tensor_count"] > 0
        gguf_path = Path(data["path"])
        assert gguf_path.is_file()
        assert gguf_path.stat().st_size > 0
        assert data["sha256"]
        sidecar = gguf_path.with_suffix(gguf_path.suffix + ".json")
        assert sidecar.is_file()

    async def test_export_gguf_missing_checkpoint_returns_404(self):
        resp = await self.client.get("/api/models/no_such_model/export_gguf")
        assert resp.status == 404
