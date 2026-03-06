"""Tests for mixed quantization module."""

import pytest

from kuantala.mixed import (
    compute_heuristic_overrides,
    expand_keep_rules,
    parse_keep_rules,
)


def test_heuristic_overrides_norm_layers():
    names = ["layer1.weight", "layer1.norm.weight", "attn.to_q.weight"]
    overrides = compute_heuristic_overrides(names)
    assert "layer1.norm.weight" in overrides
    assert overrides["layer1.norm.weight"] == "F16"


def test_heuristic_overrides_attention():
    names = ["block.attn.to_q.weight", "block.attn.to_k.weight", "block.ff.weight"]
    overrides = compute_heuristic_overrides(names)
    assert "block.attn.to_q.weight" in overrides
    assert "block.attn.to_k.weight" in overrides
    assert "block.ff.weight" not in overrides


def test_heuristic_overrides_timestep():
    names = ["time_embed.linear.weight", "regular.weight"]
    overrides = compute_heuristic_overrides(names)
    assert "time_embed.linear.weight" in overrides
    assert "regular.weight" not in overrides


def test_parse_keep_rules():
    rules = parse_keep_rules(["norm_*:F16", "attn*:Q8_0"])
    assert rules == {"norm_*": "F16", "attn*": "Q8_0"}


def test_parse_keep_rules_invalid():
    with pytest.raises(ValueError, match="Invalid --keep spec"):
        parse_keep_rules(["bad_spec_no_colon"])


def test_expand_keep_rules():
    rules = {"norm_*": "F16", "attn*": "Q8_0"}
    names = ["norm_1.weight", "norm_2.weight", "attn_q.weight", "ff.weight"]
    overrides = expand_keep_rules(rules, names)
    assert overrides == {
        "norm_1.weight": "F16",
        "norm_2.weight": "F16",
        "attn_q.weight": "Q8_0",
    }
