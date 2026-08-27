import torch

from prismatic.models.ditx_vla_adapter import DiTXVLAdapter


def run_one(mode: str) -> None:
    torch.manual_seed(0)
    model = DiTXVLAdapter(
        input_dim=7,
        output_dim=7,
        horizon=8,
        hidden_dim=32,
        n_layer=2,
        n_head=4,
        n_emb=32,
        num_task_tokens=5,
        zero_init_adaln=False,
        zero_init_output=False,
        condition_injection_mode=mode,
    )
    sample = torch.randn(2, 8, 7)
    context = torch.randn(2, 6, 32)
    mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]],
        dtype=torch.bool,
    )
    out = model(sample, torch.rand(2), torch.rand(2), context, mask)
    assert out.shape == (2, 8, 7), (mode, out.shape)
    assert torch.isfinite(out).all(), mode
    print(f"[smoke] {mode} output_shape={tuple(out.shape)} abs_mean={float(out.abs().mean()):.6f}")


if __name__ == "__main__":
    for injection_mode in ("cross_attn", "joint_prefix", "action_expert_prefix"):
        run_one(injection_mode)
