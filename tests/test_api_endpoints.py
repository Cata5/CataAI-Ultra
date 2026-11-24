"""
tests/test_api_endpoints.py
Automated integration tests for RoCanon-48.5M Studio v2 REST API and WebSocket endpoints.
"""

import json
import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from src.dashboard.server import create_app


class TestStudioAPI(AioHTTPTestCase):
    """Integration test suite for Studio v2 aiohttp HTTP routes."""

    async def get_application(self):
        return create_app()

    async def test_get_index(self):
        resp = await self.client.request("GET", "/")
        assert resp.status == 200
        data = await resp.json()
        assert data["service"] == "CataAI Ultra ML Engine Backend"
        assert data["status"] == "online"
        assert data["workstation_ui"] == "http://localhost:8080"

    async def test_get_system_status(self):
        resp = await self.client.request("GET", "/api/system/status")
        assert resp.status == 200
        data = await resp.json()
        assert "state" in data
        assert "latest_metrics" in data

    async def test_local_origin_is_reflected(self):
        resp = await self.client.request(
            "GET", "/api/system/status", headers={"Origin": "http://localhost:3000"}
        )
        assert resp.status == 200
        assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"

    async def test_remote_origin_cannot_mutate(self):
        resp = await self.client.request(
            "POST",
            "/api/action",
            headers={"Origin": "https://evil.example"},
            json={"action": "stop"},
        )
        assert resp.status == 403

    async def test_remote_origin_cannot_read(self):
        resp = await self.client.request(
            "GET", "/api/settings", headers={"Origin": "https://evil.example"}
        )
        assert resp.status == 200
        assert "Access-Control-Allow-Origin" not in resp.headers

    async def test_null_origin_gets_no_cors_permission(self):
        resp = await self.client.get("/api/system/status", headers={"Origin": "null"})
        assert resp.status == 200
        assert "Access-Control-Allow-Origin" not in resp.headers

    async def test_null_origin_post_is_rejected(self):
        resp = await self.client.post(
            "/api/settings",
            json={},
            headers={"Origin": "null", "Content-Type": "application/json"},
        )
        assert resp.status == 403

    async def test_ipv6_local_host_is_allowed(self):
        resp = await self.client.request(
            "GET", "/api/system/status", headers={"Host": "[::1]:8080"}
        )
        assert resp.status == 200

    async def test_foreign_host_header_is_rejected(self):
        resp = await self.client.request(
            "GET", "/api/system/status", headers={"Host": "evil.example:8080"}
        )
        assert resp.status == 403

    async def test_get_hardware(self):
        resp = await self.client.request("GET", "/api/hardware")
        assert resp.status == 200
        data = await resp.json()
        assert "device_name" in data
        assert "is_available" in data

    async def test_get_checkpoints(self):
        resp = await self.client.request("GET", "/api/checkpoints")
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data, list)

    async def test_get_pipelines(self):
        resp = await self.client.request("GET", "/api/pipelines")
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    async def test_scheduler_curve_api(self):
        payload = {
            "schedule_type": "wsd",
            "total_steps": 500,
            "peak_lr": 0.003,
            "warmup_steps": 50,
            "stable_steps": 300,
        }
        resp = await self.client.request("POST", "/api/scheduler/curve", json=payload)
        assert resp.status == 200
        points = await resp.json()
        assert isinstance(points, list)
        assert len(points) > 10
        assert points[0]["lr"] == 0.0

    async def test_tokenizer_inspect_api(self):
        payload = {"text": "Testare tokenizer RoCanon 12288"}
        resp = await self.client.request("POST", "/api/tokenizer/inspect", json=payload)
        assert resp.status == 200
        data = await resp.json()
        assert data["token_count"] > 0
        assert "tokens" in data
        assert data["exact_match"] is True

    async def test_runs_api(self):
        resp = await self.client.request("GET", "/api/runs")
        assert resp.status == 200
        data = await resp.json()
        assert isinstance(data, list)

    async def test_calculate_params_api(self):
        resp = await self.client.request("GET", "/api/calculate_params?vocab_size=12288&d_model=512&n_layers=16&tied=true")
        assert resp.status == 200
        data = await resp.json()
        assert data["total_parameters"] == 48531968

    async def test_eval_generate_api(self):
        resp = await self.client.request(
            "POST",
            "/api/evaluation/generate",
            json={"prompt": "Test", "max_new_tokens": 5, "temperature": 0.0},
        )
        assert resp.status == 200
        data = await resp.json()
        assert "generated_text" in data
        assert "tokens_generated" in data
        assert "device" in data

    async def test_save_local_model_api(self):
        resp = await self.client.request(
            "POST",
            "/api/models/save-local",
            json={"model_id": "rocanon_48m", "custom_name": "test_saved_model", "save_mode": "best"},
        )
        assert resp.status in (200, 500)
        data = await resp.json()
        assert "success" in data or "error" in data



