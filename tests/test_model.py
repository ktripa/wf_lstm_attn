import torch

from fwi_attn.config import Config
from fwi_attn.models.lstm_attention import FWIAttnModel

CFG = Config(
    {
        "branch_a": {"hidden_size": 4},
        "branch_b": {"hidden_size": 8, "num_layers": 1, "attn_size": 8, "dropout": 0.0, "pooling": "attention"},
        "branch_c": {"hidden_size": 3},
        "head": {"hidden_sizes": [6], "dropout": 0.0},
    }
)


def _rand_inputs(batch=5):
    return torch.randn(batch, 7), torch.randn(batch, 12, 3), torch.randn(batch, 9)


def test_full_model_forward_shape_and_attention_weights():
    model = FWIAttnModel(CFG, branch_a_in=7, branch_b_in=3, branch_c_in=9, branches=("A", "B", "C"))
    xa, xb, xc = _rand_inputs()
    out, attn = model(xa, xb, xc)
    assert out.shape == (5,)
    assert attn.shape == (5, 12)
    assert torch.allclose(attn.sum(dim=1), torch.ones(5), atol=1e-5)


def test_branch_subset_b_only():
    model = FWIAttnModel(CFG, branch_a_in=7, branch_b_in=3, branch_c_in=9, branches=("B",))
    _, xb, _ = _rand_inputs()
    out, attn = model(None, xb, None)
    assert out.shape == (5,)
    assert attn.shape == (5, 12)


def test_final_state_pooling_baseline_variant():
    cfg = Config({**CFG.to_dict(), "branch_b": {**CFG.branch_b.to_dict(), "pooling": "final_state"}})
    model = FWIAttnModel(cfg, branch_a_in=7, branch_b_in=3, branch_c_in=9, branches=("A", "B", "C"))
    xa, xb, xc = _rand_inputs()
    out, attn = model(xa, xb, xc)
    assert out.shape == (5,)
    assert attn is None
