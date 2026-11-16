"""
tests/unit/test_safe_deletion.py
Regression tests for safe deletion endpoints (datasets & network architectures).
Verifies:
1. Cannot delete dataset during active training (409 Conflict).
2. Cannot delete network architecture during active training (409 Conflict).
3. Cannot delete built-in baseline models (e.g. rocanon_48m) (400/403 status).
"""

import asyncio
from unittest.mock import MagicMock, patch
from aiohttp import web
from src.dashboard.server import handle_delete_dataset, handle_delete_network
from src.worker.worker_manager import RunState


def test_cannot_delete_dataset_during_active_training():
    """Attempting to delete a dataset during active training must fail with 409 Conflict."""
    async def _test():
        request = MagicMock(spec=web.Request)
        request.match_info = {"id": "tech_wiki_ro"}

        with patch("src.dashboard.server.WORKER_MGR") as mock_mgr:
            mock_mgr.current_state = RunState.RUNNING
            mock_mgr.current_payload = {"dataset_source": "tech_wiki_ro"}
            mock_mgr.current_run_id = None

            response = await handle_delete_dataset(request)
            assert response.status == 409
            assert "active training run is currently utilizing" in response.text

    asyncio.run(_test())


def test_cannot_delete_network_during_active_training():
    """Attempting to delete a network model during active training must fail with 409 Conflict."""
    async def _test():
        request = MagicMock(spec=web.Request)
        request.match_info = {"id": "custom_net_1"}

        with patch("src.dashboard.server.WORKER_MGR") as mock_mgr:
            mock_mgr.current_state = RunState.RUNNING
            mock_mgr.current_payload = {"network_id": "custom_net_1"}
            mock_mgr.current_run_id = None

            response = await handle_delete_network(request)
            assert response.status == 409
            assert "active training run is currently utilizing" in response.text

    asyncio.run(_test())


def test_cannot_delete_builtin_model():
    """Attempting to delete built-in templates (rocanon_48m) must fail."""
    async def _test():
        request = MagicMock(spec=web.Request)
        request.match_info = {"id": "rocanon_48m"}

        with patch("src.dashboard.server.WORKER_MGR") as mock_mgr:
            mock_mgr.current_state = RunState.IDLE
            mock_mgr.current_payload = {}
            mock_mgr.current_run_id = None

            response = await handle_delete_network(request)
            assert response.status in (400, 403)
            assert "protected" in response.text or "Could not delete" in response.text

    asyncio.run(_test())
