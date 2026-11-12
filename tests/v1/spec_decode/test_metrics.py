# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import partial
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
from prometheus_client import CollectorRegistry, Counter

from vllm.v1.spec_decode.metrics import (
    AdaptiveVerificationStats,
    SpecDecodingLogging,
    SpecDecodingProm,
    SpecDecodingStats,
)

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("verified", [0, 12, 20])
def test_adaptive_counts_use_untrimmed_widths_and_admitted_batch_total(verified):
    # The zero-width row represents a prefill request mixed with verification.
    batch = SimpleNamespace(
        num_draft_tokens_per_req=np.array([7, 6, 0, 7]),
        num_draft_tokens=verified,
    )
    stats = AdaptiveVerificationStats.from_batch(batch)
    assert stats.proposed_draft_tokens == 20
    assert stats.verified_draft_tokens == verified
    assert stats.trimmed_draft_tokens == 20 - verified


@pytest.mark.parametrize("widths", [None, np.zeros(3, dtype=np.int32)])
def test_batches_without_draft_slots_do_not_contribute_to_trim_ratio(widths):
    batch = SimpleNamespace(num_draft_tokens_per_req=widths, num_draft_tokens=0)
    assert AdaptiveVerificationStats.from_batch(batch) is None


@pytest.mark.parametrize("verified", [0, 12, 20])
def test_adaptive_stats_survive_frontend_msgpack_serialization(verified):
    stats = SpecDecodingStats.new(7)
    stats.adaptive_verification_stats = AdaptiveVerificationStats(20, verified)
    decoded = msgspec.msgpack.decode(
        msgspec.msgpack.encode(stats), type=SpecDecodingStats
    )
    assert decoded.adaptive_verification_stats == stats.adaptive_verification_stats


def test_adaptive_log_weights_by_tokens_and_resets_with_no_live_draft_results():
    logging = SpecDecodingLogging()
    messages = []

    def log_fn(fmt, *args):
        messages.append(fmt % args)

    for proposed, verified in [(20, 12), (2, 0), (8, 8)]:
        stats = SpecDecodingStats.new(7)
        stats.adaptive_verification_stats = AdaptiveVerificationStats(
            proposed, verified
        )
        logging.observe(stats)
    logging.log(log_fn=log_fn)
    assert messages == [
        (
            "Adaptive verification metrics: Proposed: 30 tokens, Verified: 20 tokens, "
            "Trimmed: 10 tokens, Trim ratio: 33.3%"
        )
    ]
    logging.log(log_fn=log_fn)
    assert len(messages) == 1


def test_prometheus_counts_zero_budget_once_and_keeps_engines_separate(monkeypatch):
    registry = CollectorRegistry()
    monkeypatch.setattr(
        SpecDecodingProm, "_counter_cls", partial(Counter, registry=registry)
    )
    prom = SpecDecodingProm(
        SimpleNamespace(num_speculative_tokens=7, enable_adaptive_verification=True),
        ["model_name", "engine"],
        {0: ["test", "0"], 1: ["test", "1"]},
    )
    for proposed, verified in [(20, 12), (2, 0), (8, 8)]:
        stats = SpecDecodingStats.new(7)
        stats.adaptive_verification_stats = AdaptiveVerificationStats(
            proposed, verified
        )
        prom.observe(stats, engine_idx=0)
    for suffix, expected in [("proposed", 30), ("trimmed", 10)]:
        name = f"vllm:adaptive_{suffix}_draft_tokens_total"
        assert (
            registry.get_sample_value(name, {"model_name": "test", "engine": "0"})
            == expected
        )
        assert (
            registry.get_sample_value(name, {"model_name": "test", "engine": "1"}) == 0
        )


@pytest.mark.parametrize("mode", ["fixed", "disabled", "diffusion"])
def test_adaptive_counters_are_absent_outside_adaptive_verification(monkeypatch, mode):
    registry = CollectorRegistry()
    monkeypatch.setattr(
        SpecDecodingProm, "_counter_cls", partial(Counter, registry=registry)
    )
    config = (
        SimpleNamespace(num_speculative_tokens=7, enable_adaptive_verification=False)
        if mode == "fixed"
        else None
    )
    prom = SpecDecodingProm(
        config, ["engine"], {0: ["0"]}, is_diffusion=mode == "diffusion"
    )
    prom.observe(SpecDecodingStats.new(7))
    assert not any(
        metric.name.startswith("vllm:adaptive_") for metric in registry.collect()
    )
