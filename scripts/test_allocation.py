import math
import pytest

from sadra.rank_allocator import allocate_ranks, RankAllocConfig

def lora_param_cost(in_dim: int, out_dim: int, r: int) -> int:
    """
    LoRA param count for a Linear layer (A: r x in, B: out x r)
    total = r*in + out*r
    """
    return r * in_dim + out_dim * r

def test_allocate_ranks_respects_budget_and_pruning():
    # Fake shapes: {module_name: (in_dim, out_dim)}
    shapes = {
        "m0": (768, 768),
        "m1": (768, 3072),
        "m2": (3072, 768),
        "m3": (768, 768),
    }

    # Fake sensitivity (higher => more important)
    sens = {
        "m0.weight": 0.01,   # should be pruned (maybe)
        "m1.weight": 1.20,   # important
        "m2.weight": 0.90,   # important
        "m3.weight": 0.05,   # less important
    }

    cfg = RankAllocConfig(
        lora_budget=200_000,
        rank_min=0,                 # allow pruning
        rank_max=16,
        allowed_ranks=(0, 2, 4, 8, 16),  # discrete + zero
        use_discrete_ranks=True,
    )

    rank_config = allocate_ranks(sens, shapes, cfg)

    # ---- A) r must be >= 0 and in allowed set if discrete mode
    for name, r in rank_config.items():
        assert r >= 0
        assert r in cfg.allowed_ranks

    # ---- B) Budget check ONLY for r>0 modules
    total = 0
    for name, r in rank_config.items():
        if r <= 0:
            continue
        mod = name.replace(".weight", "")
        assert mod in shapes
        in_dim, out_dim = shapes[mod]
        total += lora_param_cost(in_dim, out_dim, r)

    assert total <= cfg.lora_budget

    # ---- C) Ensure at least one module is pruned when budget tight
    assert any(r == 0 for r in rank_config.values())

def test_allocate_ranks_returns_weight_keys():
    shapes = {"x": (128, 128)}
    sens = {"x.weight": 1.0}

    cfg = RankAllocConfig(
        lora_budget=10_000,
        rank_min=0,
        rank_max=8,
        allowed_ranks=(0, 2, 4, 8),
        use_discrete_ranks=True,
    )

    rank_config = allocate_ranks(sens, shapes, cfg)

    # SADRA pipeline expects sensitivity keys like "*.weight"
    assert "x.weight" in rank_config
