"""
tests/unit/test_remediation_m2.py
Automated acceptance tests for Milestone 2:
- BUG-04: Uploaded dataset integration & strict resolution (no silent fallback)
- BUG-06: CNN normalization shape fix (SequenceBatchNorm1d forward/backward)
"""

from pathlib import Path
import pytest
import torch

from src.datasets.manager import DatasetManager
from src.data.custom_loader import CustomDatasetEngine
from src.db.database import StudioDatabase
from src.networks.factory import ModelFactory
from src.networks.registry import NetworkRegistry
from src.tokenizer.tokenizer import RoCanonTokenizer


class TestMilestone2Remediation:
    def test_cnn_batchnorm_shape_and_gradients(self):
        """BUG-06: conv1d_network_4m forward and backward pass with SequenceBatchNorm1d."""
        reg = NetworkRegistry()
        cfg = reg.get_network("conv1d_network_4m")
        assert cfg is not None, "conv1d_network_4m must exist in registry"

        model = ModelFactory.create(dict(cfg))
        input_dim = int(cfg.get("input_dim", 64))
        x = torch.randn(2, 16, input_dim)

        out = model(x)
        assert out.shape == torch.Size([2, 2]), f"Expected shape [2, 2], got {out.shape}"

        loss = out.sum()
        loss.backward()

        # Verify gradient reached the conv layer
        assert model.blocks[0].conv.weight.grad is not None
        assert not torch.all(model.blocks[0].conv.weight.grad == 0)

    def test_uploaded_dataset_resolution_and_content_integrity(self, tmp_path):
        """BUG-04: Uploaded dataset is resolved by ID from DB and distinctive content is loaded."""
        db = StudioDatabase(tmp_path / "studio.db")
        dm = DatasetManager(upload_dir=tmp_path / "uploads", db=db)
        tokenizer = RoCanonTokenizer()

        # Create a distinctive dataset text
        distinctive_text = (
            "Acesta este un document de testare extrem de specific pentru CataAI Ultra.\n"
            "Verificam ca tokenii din acest text unic ajung in mod real in tensorii de antrenare.\n"
            "Niciun fel de text sintetic generic nu trebuie sa inlocuiasca aceste date reale.\n"
        )
        test_file = tmp_path / "distinctive_rocanon_test.txt"
        test_file.write_text(distinctive_text, encoding="utf-8")

        reg_res = dm.register_uploaded_file(test_file, name="distinctive_rocanon_test.txt")
        assert reg_res["success"] is True
        dataset_id = reg_res["dataset_id"]

        # Instantiate CustomDatasetEngine wired to the DB
        engine = CustomDatasetEngine(tokenizer=tokenizer, cache_dir=str(tmp_path / "cache"), db=db)

        # 1. Resolve by registered dataset ID
        train_in, train_lbl, val_in, val_lbl = engine.load_dataset(
            dataset_source=dataset_id,
            phase="pretrain",
            device=torch.device("cpu"),
        )

        assert len(train_in) > 0, "Train inputs must contain tokens from uploaded dataset"

        # Decode tokens to confirm actual uploaded text reached the tensors
        decoded_text = tokenizer.decode(train_in[:50].tolist())
        assert "document de testare" in decoded_text or "CataAI" in decoded_text, (
            f"Decoded text must contain distinctive uploaded prose, got: {decoded_text}"
        )

    def test_unknown_dataset_id_raises_explicit_error(self, tmp_path):
        """BUG-04: Unknown dataset ID must raise ValueError, never silently fall back to synthetic data."""
        db = StudioDatabase(tmp_path / "studio.db")
        engine = CustomDatasetEngine(cache_dir=str(tmp_path / "cache"), db=db)

        with pytest.raises(ValueError) as excinfo:
            engine.load_dataset(dataset_source="non_existent_dataset_id_12345", phase="pretrain")

        assert "Unknown or missing dataset source" in str(excinfo.value)
        assert "Synthetic data is only used when explicitly requested" in str(excinfo.value)

    def test_explicit_synthetic_mode(self, tmp_path):
        """Synthetic data is generated only when explicitly set to 'synthetic'."""
        engine = CustomDatasetEngine(cache_dir=str(tmp_path / "cache"))
        train_in, _, val_in, _ = engine.load_dataset(dataset_source="synthetic", phase="pretrain")
        assert len(train_in) > 0
