"""
tests/unit/test_v3_enhancements.py
Unit tests for the 4 core enhancements:
1. Dynamic Learning Rate Decay Floor (min_lr_floor in WSDScheduler & GeneralLRScheduler)
2. Discord & Telegram Webhook Notifications (non-blocking dispatch, embed/message formatting)
3. Dataset Quality Scorer & MinHash Deduplication (TextCurator integration in DatasetManager)
4. Streaming Token Generation (InferenceEngine.stream_generate)
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from src.datasets.manager import DatasetManager
from src.eval.inference import InferenceEngine
from src.notifications.webhook import WebhookNotifier
from src.training.scheduler import GeneralLRScheduler, WSDScheduler


class TestLearningRateFloor(unittest.TestCase):
    """Verifies that dynamic learning rate decay floor prevents learning rate from reaching absolute zero."""

    def test_wsd_scheduler_min_lr_floor(self):
        p = nn.Parameter(torch.zeros(2, 2))
        opt = torch.optim.AdamW([p], lr=1e-3)

        min_floor = 1e-4
        scheduler = WSDScheduler(
            optimizers=opt,
            warmup_steps=100,
            plateau_tokens=1_000,
            total_tokens=2_000,
            min_lr_ratio=0.0,  # Would normally decay to 0.0
            min_lr_floor=min_floor,
            decay_style="linear",
        )

        # Step into deep decay beyond total_tokens
        scheduler.step(step=5000, tokens=10_000)
        current_lr = opt.param_groups[0]["lr"]

        self.assertGreaterEqual(current_lr, min_floor)
        self.assertAlmostEqual(current_lr, min_floor, places=6)

    def test_general_lr_scheduler_min_lr_floor(self):
        p = nn.Parameter(torch.zeros(2, 2))
        opt = torch.optim.AdamW([p], lr=1e-3)

        min_floor = 5e-5
        scheduler = GeneralLRScheduler(
            optimizers=opt,
            total_steps=100,
            warmup_steps=10,
            schedule_type="linear",
            min_lr_ratio=0.0,
            min_lr_floor=min_floor,
        )

        # Step way past total_steps
        for s in range(150):
            scheduler.step()

        current_lr = opt.param_groups[0]["lr"]
        self.assertGreaterEqual(current_lr, min_floor)
        self.assertAlmostEqual(current_lr, min_floor, places=6)


class TestWebhookNotifications(unittest.TestCase):
    """Verifies Discord and Telegram webhook dispatchers and payload formatting."""

    def setUp(self):
        self.notifier = WebhookNotifier(
            discord_webhook_url="https://discord.com/api/webhooks/test/123",
            telegram_bot_token="123456:ABC-DEF",
            telegram_chat_id="987654321",
            enabled=True,
        )

    @patch("urllib.request.urlopen")
    def test_discord_sync_dispatch(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 204
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        fields = [{"name": "Step", "value": "100"}]
        success = self.notifier._send_discord_sync(
            webhook_url="https://discord.com/api/webhooks/test/123",
            title="Test Run Completed",
            description="All steps executed.",
            level="SUCCESS",
            fields=fields,
        )

        self.assertTrue(success)
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        data = json.loads(req.data.decode("utf-8"))
        self.assertEqual(len(data["embeds"]), 1)
        self.assertEqual(data["embeds"][0]["title"], "Test Run Completed")
        self.assertEqual(data["embeds"][0]["color"], 0x10B981)  # SUCCESS green

    @patch("urllib.request.urlopen")
    def test_telegram_sync_dispatch(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_urlopen.return_value.__enter__.return_value = mock_resp

        fields = [{"name": "Best Val Loss", "value": "1.2345"}]
        success = self.notifier._send_telegram_sync(
            bot_token="123456:ABC-DEF",
            chat_id="987654321",
            title="Best Checkpoint",
            message="New record loss.",
            level="BEST",
            fields=fields,
        )

        self.assertTrue(success)
        mock_urlopen.assert_called_once()
        req = mock_urlopen.call_args[0][0]
        self.assertIn("api.telegram.org/bot123456:ABC-DEF/sendMessage", req.full_url)
        data = json.loads(req.data.decode("utf-8"))
        self.assertEqual(data["chat_id"], "987654321")
        self.assertIn("Best Checkpoint", data["text"])

    @patch.object(WebhookNotifier, "_send_discord_sync", return_value=True)
    @patch.object(WebhookNotifier, "_send_telegram_sync", return_value=True)
    def test_test_notification_method(self, mock_tg, mock_dc):
        res = self.notifier.test_notification()
        self.assertTrue(res["discord"])
        self.assertTrue(res["telegram"])
        self.assertEqual(len(res["errors"]), 0)


class TestDatasetCuration(unittest.TestCase):
    """Verifies MinHash deduplication and dataset quality curation."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mgr = DatasetManager(upload_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_curate_dataset_deduplication(self):
        # Create a raw dataset JSONL with duplicates
        raw_path = os.path.join(self.temp_dir, "test_raw.jsonl")
        samples = [
            {"text": "Aceasta este o propoziție în limba română pentru testarea calității modelului CataAI."},
            {"text": "Aceasta este o propoziție în limba română pentru testarea calității modelului CataAI."},  # Exact dupe
            {"text": "Aceasta este o propozitie in limba romana pentru testarea calitatii modelului CataAI."},  # Near dupe
            {"text": "Un text complet diferit și unic despre antrenarea rețelelor neuronale pe GPU."},
            {"text": "Short"},  # Too short heuristic filter
        ]
        with open(raw_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        meta = self.mgr.register_uploaded_file(file_path=raw_path, name="Raw Test")
        ds_id = meta["dataset_id"]

        curate_stats = self.mgr.curate_dataset(
            dataset_id=ds_id,
            jaccard_threshold=0.75,
            lid_threshold=0.50,
            save_curated=True,
        )

        self.assertEqual(curate_stats["total_samples"], len(samples))
        self.assertGreater(curate_stats["curated_samples"], 0)
        self.assertGreaterEqual(curate_stats["exact_duplicates_dropped"], 1)
        self.assertIn("curated_dataset_id", curate_stats)
        self.assertIn("retention_rate", curate_stats)


class TestStreamingInference(unittest.TestCase):
    """Verifies token-by-token streaming inference in InferenceEngine."""

    def test_stream_generate_generator(self):
        # Mock model and tokenizer
        mock_model = MagicMock()
        mock_model.config.max_seq_len = 2048

        token_sequence = [1, 2, 3]
        step_counter = [0]

        def mock_forward(input_ids):
            # Return logits with 100 classes where step_counter token has highest logit
            logits = torch.zeros(1, input_ids.shape[1], 100)
            idx = step_counter[0]
            if idx < len(token_sequence):
                chosen_tok = token_sequence[idx]
            else:
                chosen_tok = 99  # EOS
            step_counter[0] += 1
            logits[0, -1, chosen_tok] = 100.0
            return logits

        mock_model.side_effect = mock_forward

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [10]
        mock_tokenizer.eos_token_id = 99
        mock_tokenizer._tokenizer.token_to_id.return_value = None

        token_map = {10: "", 1: "Salut", 2: " lume", 3: "!"}

        def mock_decode(ids):
            return "".join([token_map.get(i, "") for i in ids])

        mock_tokenizer.decode.side_effect = mock_decode

        engine = InferenceEngine(model=mock_model, tokenizer=mock_tokenizer, device="cpu")

        chunks = []
        for chunk in engine.stream_generate(prompt="Salut", max_new_tokens=3, temperature=0.0):
            chunks.append(chunk)

        # Verify stream yielded tokens chunk by chunk
        self.assertGreater(len(chunks), 0)
        full_text = "".join(chunks)
        self.assertEqual(full_text, "Salut lume!")


class TestLocalModelSaver(unittest.TestCase):
    """Verifies local model package saving with keep-training .pt, stripped .pt, and charts."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.chk_dir = os.path.join(self.test_dir, "checkpoints")
        self.models_dir = os.path.join(self.test_dir, "models")
        os.makedirs(self.chk_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        from src.checkpoints.manager import CheckpointManager
        self.mgr = CheckpointManager(checkpoint_dir=self.chk_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_save_local_model_package_dual_weights_and_charts(self):
        # Create a simple dummy model with linear layers and optimizer
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # Populate optimizer state
        dummy_in = torch.randn(2, 16)
        loss = model(dummy_in).sum()
        loss.backward()
        opt.step()

        # Dummy metrics history
        metrics = [
            {"step": i, "train_loss": 3.0 - 0.01 * i, "val_loss": 3.2 - 0.008 * i if i % 10 == 0 else None,
             "muon_lr": 0.02, "adamw_lr": 0.002, "tokens_per_second": 5000, "step_time_ms": 150}
            for i in range(1, 31)
        ]

        result = self.mgr.save_local_model_package(
            model_id="test_model_48m",
            custom_name="my_custom_model",
            save_mode="current",
            active_model=model,
            active_optimizers=(None, opt),
            metrics_history=metrics,
            output_base_dir=self.models_dir,
        )

        self.assertIn("files", result)
        full_info = result["files"]["full_weights"]
        stripped_info = result["files"]["stripped_weights"]

        # Check full .pt exists and loads
        self.assertTrue(os.path.exists(full_info["path"]))
        full_sd = torch.load(full_info["path"], map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", full_sd)
        self.assertIn("adamw_state_dict", full_sd)
        self.assertEqual(full_sd["type"], "keep_training")

        # Check stripped .pt exists and loads without optimizer states
        self.assertTrue(os.path.exists(stripped_info["path"]))
        stripped_sd = torch.load(stripped_info["path"], map_location="cpu", weights_only=False)
        self.assertIn("model_state_dict", stripped_sd)
        self.assertNotIn("adamw_state_dict", stripped_sd)
        self.assertEqual(stripped_sd["type"], "stripped_inference")

        # Check file size reduction: stripped weights should be smaller or equal
        self.assertLessEqual(stripped_info["size_bytes"], full_info["size_bytes"])

        # Check metadata.json exists
        meta_path = os.path.join(self.models_dir, "my_custom_model", "metadata.json")
        self.assertTrue(os.path.exists(meta_path))

        # Check generated charts
        charts = result["files"]["charts"]
        self.assertIn("val_loss_curve", charts)
        self.assertIn("training_dynamics", charts)
        val_chart_file = os.path.join(self.models_dir, "my_custom_model", charts["val_loss_curve"])
        dynamics_chart_file = os.path.join(self.models_dir, "my_custom_model", charts["training_dynamics"])
        self.assertTrue(os.path.exists(val_chart_file))
        self.assertTrue(os.path.exists(dynamics_chart_file))


if __name__ == "__main__":
    unittest.main()
