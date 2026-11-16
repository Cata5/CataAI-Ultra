"""
tests/test_frontend_integration.py
Validates the full REST and WebSocket contract surface consumed by the Next.js Sci-Fi Workstation v2.
"""

import json
import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from src.dashboard.server import create_app


class TestFrontendBackendIntegration(AioHTTPTestCase):
    async def get_application(self):
        return create_app()

    async def test_system_status_contract(self):
        resp = await self.client.request("GET", "/api/system/status")
        assert resp.status == 200
        data = await resp.json()
        assert "state" in data
        assert "run_id" in data
        assert "alerts_count" in data

    async def test_hardware_telemetry_contract(self):
        resp = await self.client.request("GET", "/api/hardware")
        assert resp.status == 200
        data = await resp.json()
        assert "device_name" in data
        assert "cuda_available" in data
        assert "is_available" in data
        assert "total_vram_mb" in data or "pytorch_allocated_mb" in data

    async def test_networks_and_templates_contract(self):
        resp = await self.client.request("GET", "/api/networks/templates")
        assert resp.status == 200
        templates = await resp.json()
        assert isinstance(templates, list)
        assert len(templates) >= 6

        # Check RoCanon template structure
        rocanon = next((t for t in templates if t["id"] == "rocanon_48m"), None)
        assert rocanon is not None
        assert rocanon["vocab_size"] == 12288
        assert rocanon["d_model"] == 512
        assert rocanon["n_layers"] == 16

    async def test_network_estimation_and_validation(self):
        config = {
            "id": "rocanon_48m",
            "name": "RoCanon-48.5M Baseline",
            "architecture_family": "decoder_only",
            "vocab_size": 12288,
            "d_model": 512,
            "n_layers": 16,
            "n_q": 8,
            "n_kv": 2,
            "d_ff": 1280,
            "conv_kernel_size": 4,
            "has_conv1d": True,
            "tie_word_embeddings": True,
        }

        # Estimate
        resp_est = await self.client.request(
            "POST",
            "/api/networks/estimate",
            json={"config": config, "micro_batch_size": 4, "seq_len": 256},
        )
        assert resp_est.status == 200
        est_data = await resp_est.json()
        assert "parameters" in est_data
        assert "memory" in est_data
        assert est_data["parameters"]["total_parameters"] == 48531968
        assert est_data["memory"]["total_training_vram_mb"] > 0

        # Validate
        resp_val = await self.client.request("POST", "/api/networks/validate", json=config)
        assert resp_val.status == 200
        val_data = await resp_val.json()
        assert "valid" in val_data
        assert val_data["valid"] is True

    async def test_scheduler_curve_calculation(self):
        payload = {
            "schedule_type": "wsd",
            "total_steps": 1000,
            "peak_lr": 0.003,
            "min_lr": 0.0001,
            "warmup_steps": 100,
            "stable_steps": 700,
            "decay_style": "cosine",
        }
        resp = await self.client.request("POST", "/api/scheduler/curve", json=payload)
        assert resp.status == 200
        curve_list = await resp.json()
        assert isinstance(curve_list, list)
        assert len(curve_list) > 0
        assert "step" in curve_list[0]
        assert "lr" in curve_list[0]

    async def test_tokenizer_inspection(self):
        payload = {"text": "Modelul RoCanon procesează textul rapid."}
        resp = await self.client.request("POST", "/api/tokenizer/inspect", json=payload)
        assert resp.status == 200
        data = await resp.json()
        assert "token_count" in data
        assert "tokens" in data
        assert data["token_count"] > 0

    async def test_checkpoints_list_contract(self):
        resp = await self.client.request("GET", "/api/checkpoints")
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data, list)

    async def test_settings_get_and_post(self):
        resp = await self.client.request("GET", "/api/settings")
        assert resp.status == 200

        save_payload = {"vram_ceiling_mb": 4900, "temp_threshold_c": 78}
        resp_save = await self.client.request("POST", "/api/settings", json=save_payload)
        assert resp_save.status == 200
        saved = await resp_save.json()
        assert saved["status"] == "saved"

    async def test_dataset_stats_and_tokenize_contract(self):
        # Test preset stats
        resp = await self.client.request("GET", "/api/datasets/tech_wiki_ro/stats")
        assert resp.status == 200
        data = await resp.json()
        assert data["id"] == "tech_wiki_ro"
        assert "total_tokens" in data
        assert "file_path" in data

        # Test tokenize route
        resp_tok = await self.client.request("POST", "/api/datasets/synthetic/tokenize")
        assert resp_tok.status == 200
        tok_data = await resp_tok.json()
        assert tok_data["success"] is True
        assert tok_data["dataset_id"] == "synthetic"

    async def test_websocket_telemetry_handshake(self):
        ws = await self.client.ws_connect("/ws")
        msg = await ws.receive_json()
        assert "event_type" in msg
        assert msg["event_type"] == "STATE_CHANGED"
        await ws.close()
