import torch

from prismatic.models.action_heads import VelocityNetwork


def main():
    net = VelocityNetwork(
        input_dim=32,
        hidden_dim=32,
        action_dim=7,
        num_blocks=1,
        num_task_tokens=3,
        num_heads=4,
        mlp_ratio=2.0,
        use_adaptive_bridge=False,
        fixed_layer_index=24,
        balanced_train_layers="1,5,9,13,24",
    )
    net.train()
    counts = torch.zeros(25, dtype=torch.long)
    for _ in range(5):
        z = torch.randn(8, 8, 7)
        time = torch.randn(8, 32)
        task = torch.randn(8, 25, 3, 32)
        action = torch.randn(8, 25, 8, 32)
        proprio = torch.randn(8, 1, 32)
        output = net(z, time, time, task, action, proprio)
        assert output.shape == (8, 8, 7)
        counts.scatter_add_(0, net._last_selected_layers.cpu(), torch.ones(8, dtype=torch.long))
    assert counts[[1, 5, 9, 13, 24]].tolist() == [8, 8, 8, 8, 8], counts.tolist()
    net.eval()
    with torch.no_grad():
        output = net(z, time, time, task, action, proprio)
    assert output.shape == (8, 8, 7)
    assert net._last_selected_layers is None
    print({"candidate_counts": counts[[1, 5, 9, 13, 24]].tolist(), "eval_fixed_layer": 24})


if __name__ == "__main__":
    main()
