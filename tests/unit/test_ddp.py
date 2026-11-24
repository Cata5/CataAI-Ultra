"""CPU/gloo proof that the DDP wrapper all-reduces gradients."""

import os
import socket

import torch

from src.model.architecture import RoCanonConfig, RoCanonForCausalLM
from src.training.distributed import wrap_model_for_ddp


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _ddp_worker(rank: int, world_size: int, port: int, out_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.distributed.init_process_group("gloo", init_method="env://")
    cfg = RoCanonConfig()
    cfg.vocab_size = 32
    cfg.d_model = 16
    cfg.n_layers = 1
    cfg.n_q = 2
    cfg.n_kv = 1
    cfg.d_h = 8
    cfg.d_ff = 32
    cfg.max_seq_len = 16
    model = RoCanonForCausalLM(cfg)
    ddp = wrap_model_for_ddp(model, torch.device("cpu"), local_rank=rank)
    optimizer = torch.optim.SGD(ddp.parameters(), lr=0.1)
    optimizer.zero_grad()
    tokens = torch.full((1, 8), 5 + rank, dtype=torch.long)
    logits = ddp(tokens)
    loss = logits.float().pow(2).mean()
    loss.backward()
    optimizer.step()
    grad = ddp.module.embed_tokens.weight.grad.detach().clone()
    torch.distributed.barrier()
    torch.save(grad, os.path.join(out_dir, f"grad_{rank}.pt"))
    torch.distributed.destroy_process_group()


def test_ddp_all_reduces_gradients(tmp_path) -> None:
    world_size = 2
    port = _free_port()
    torch.multiprocessing.spawn(
        _ddp_worker,
        args=(world_size, port, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    rank0 = torch.load(tmp_path / "grad_0.pt", weights_only=True)
    rank1 = torch.load(tmp_path / "grad_1.pt", weights_only=True)
    assert torch.allclose(rank0, rank1)
