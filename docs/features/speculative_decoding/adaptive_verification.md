# Adaptive Verification

Speculative decoding buys fewer decode steps with more compute. At batch size 1 that is a good trade: the GPU is memory-bound with spare compute, so the extra draft tokens are close to free. At batch size 256 it is a much more delicate one. Draft tokens now compete with real tokens for the same compute, and every rejected token is compute wasted; with enough of them, throughput drops.

That matters because per-position acceptance decays fast. While the GPU is memory-bound that slot is effectively free and worth the gamble; once it saturates the gamble has a real throughput cost. The crossover moves with load and with workload-dependent acceptance rates, so no static `num_speculative_tokens` is right across concurrencies.

Adaptive verification decides per step how much of the draft to verify instead. Every (request, position) draft slot is scored by its *survival probability*, the running product of that request's per-position confidences, and the highest-scoring slots are admitted until a global budget is spent. Slots compete across requests: position 5 of a confident request can outrank position 1 of a doubtful one, so one request keeps its full block while another could be trimmed after a token or two.

The budget itself comes from a cost model profiled at startup. vLLM measures what a step costs at each shape, then picks the token count that maximizes expected accepted tokens per second.

The practical effect is that one configuration holds up across the whole load range, which removes most of the need to tune `num_speculative_tokens` per deployment.

## Support

Adaptive verification needs per-position acceptance estimates, so today it is only supported for DSpark with a **confidence head**.

## Usage

It is off by default. Enable it in the speculative config:

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash-DSpark \
  --tokenizer-mode deepseek_v4 --trust-remote-code \
  --speculative-config '{
    "method": "dspark",
    "model": "deepseek-ai/DeepSeek-V4-Flash-DSpark",
    "num_speculative_tokens": 7,
    "draft_sample_method": "probabilistic",
    "enable_adaptive_verification": true
  }'
```

Set `enable_adaptive_verification: false` to verify the full block for every request.

## Requirements and limitations

- The attention backend must tolerate device-decided query lengths, since the CPU lengths only bound them from above. Backends that plan off the CPU lengths are excluded by the attention selector, and rejected at startup for models that hard-wire their backend.
- Full cudagraphs are required: step costs are profiled from captured graphs, so `--enforce-eager` is rejected at startup.
- Not supported with LoRA (the per-token LoRA mapping is built from CPU-side boundaries) or pipeline parallelism (cost curves and confidences exist only on the last rank).

## Tuning the cost profile

Step costs are profiled against a synthetic KV context, 8192 tokens by default. Deployments serving much longer contexts may want to raise it so the profiled step reads a more realistic amount of cache (this matters a bit less for sparse attention models like DeepSeek-v4 since the cheap indexer is the main cost that scales with context length).

```bash
export VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN=131072
```

## Measuring draft trimming

With adaptive verification enabled, `/metrics` exposes two counters with the
existing `model_name` and `engine` labels:

- `vllm:adaptive_proposed_draft_tokens_total`: scheduled draft slots before
  adaptive trimming.
- `vllm:adaptive_trimmed_draft_tokens_total`: scheduled draft slots removed by
  the adaptive budget before the target forward pass.

The token-weighted trim ratio across engines is:

```promql
sum by (model_name) (rate(vllm:adaptive_trimmed_draft_tokens_total[5m]))
/
sum by (model_name) (rate(vllm:adaptive_proposed_draft_tokens_total[5m]))
```

For example, 20 scheduled drafts and a budget of 12 give a trim ratio of
`(20 - 12) / 20 = 40%`. A zero budget counts all scheduled slots as trimmed;
a full budget counts none. Batches without scheduled drafts do not contribute.
When there are no draft observations in the query window, the ratio is undefined.
The counters are only registered for adaptive verification, so a fixed-K run
does not expose these time series.

The regular statistics log also reports the totals and ratio for its interval:

```text
Adaptive verification metrics: Proposed: 20 tokens, Verified: 12 tokens, Trimmed: 8 tokens, Trim ratio: 40.0%
```

Counts exclude bonus tokens and graph padding. They describe scheduled draft
slots, including placeholder slots, rather than all tokens produced by the draft
model. Trimming includes the budget's logits-capacity limit. The measurements use
CPU counts already available in the V2 GPU runner and add no device synchronization.
They are carried with each model-runner output, including zero-budget steps and
steps whose requests no longer produce live outputs.

A high trim ratio shows that less draft verification work was admitted. It does
not measure savings in draft generation or prove a throughput improvement.
Compare output tokens/s and p95/p99 latency against a fixed-K run with the same
model, workload, concurrency, sampling settings, hardware, and warmup. Existing
speculative-decoding acceptance metrics retain their scheduler-based semantics;
use the new counters for the trim ratio.
