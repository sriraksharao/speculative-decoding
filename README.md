
# Speculative Decoding Simulator

Empirical measurement of speculative decoding inference speedup on a T4 GPU.
Implements the algorithm from [Leviathan et al., 2023](https://proceedings.mlr.press/v202/leviathan23a.html)
with KV-cache support and proper nucleus sampling.

## Results

### Gamma Sweep (AR baseline: 37.4 tok/s)

| γ | Throughput (tok/s) | Acceptance Rate | Speedup |
|---|---|---|---|
| 1 | 32.2 | 72.8% | 0.86x |
| 2 | 39.8 | 69.1% | 1.07x |
| 3 | 47.6 | 68.1% | 1.27x |
| **4** | **49.8** | **67.8%** | **1.33x** |
| 5 | 46.9 | 69.4% | 1.25x |
| 6 | 47.9 | 71.4% | 1.28x |
| 8 | 44.7 | 71.0% | 1.20x |

**Best: 49.8 tok/s at γ=4 (1.33x speedup)**

### Temperature Sweep (γ=4)

| Temperature | Acceptance Rate | Speedup |
|---|---|---|
| 0.0 (greedy) | 80.1% | 1.23x |
| 0.2 | 76.5% | 1.13x |
| 0.4 | 79.5% | 1.14x |
| 0.6 | 67.2% | 0.98x |
| 0.8 | 67.9% | 1.00x |
| 1.0 | 63.4% | 0.91x |
| 1.2 | 62.7% | 0.93x |

Greedy decoding (temp=0.0) yields the highest acceptance rate (80.1%) due to the deterministic distribution alignment between draft and verifier.

## Setup

| | |
|---|---|
| Draft | distilgpt2 (82M params) |
| Verifier | gpt2-medium (345M params, 4.3× larger) |
| Sampling | temperature=0.6, top-k=50, top-p=0.9 |
| Hardware | Google Colab T4 GPU (15.6 GB VRAM) |
| VRAM used | ~0.92 GB |
| Tokens generated | 100 per sequence |
| Prompts | 8 diverse prompts × 2 repeats |

Both models share the GPT-2 tokenizer (vocab=50,257). distilgpt2 was distilled
from the GPT-2 family, giving high token-level agreement with gpt2-medium.

## What's measured

- **Gamma sweep** (γ=1–8) — throughput, acceptance rate, and speedup vs. draft length
- **Temperature sweep** (0.0–1.2) — effect of sampling temperature on acceptance rate and speedup
- **Latency distribution** — p50/p95/p99 per-step latency for AR vs. speculative decoding

## Algorithm

The KV-cache aware engine works as follows:

1. **Prime** — run verifier on the prompt once to build the KV cache
2. **Draft** — distilgpt2 autoregressively proposes γ tokens (KV-cache assisted, only new token processed each step)
3. **Verify** — gpt2-medium scores the full draft in one parallel forward pass
4. **Accept/Reject** — each token accepted with probability `min(1, p_verifier / p_draft)`; on rejection, resample from corrected distribution `max(0, p_v − p_d) / Z`
5. **Re-prime** — draft KV cache rebuilt from the accepted prefix for the next round

This guarantees the output distribution is **mathematically identical** to autoregressive sampling from the verifier alone.

## Key findings

- Speedup peaks at γ=4 before declining as rejection probability accumulates over longer draft chains
- ~67–73% acceptance rate at temperature=0.6 across γ=1–8, consistent with distilgpt2/gpt2-medium distribution alignment
- Greedy decoding gives highest acceptance (80.1%) since both models agree more on argmax tokens
- KV-cache is essential — without it, draft forward passes are not cheap enough to achieve any speedup

## How to run

Open `speculative_decoding.ipynb` in Google Colab with a **T4 GPU** runtime.
No API keys required. All dependencies install in Cell 1.
```bash
# Dependencies (auto-installed in Cell 1)
pip install transformers accelerate
```

Run all cells in order. Results are saved as:
- `spec_v2_gamma_sweep.csv`
- `spec_v2_temp_sweep.csv`
- `spec_decoding_v2_dashboard.png`

## Why this matters

Production inference engines (vLLM, TensorRT-LLM) use speculative decoding
to accelerate large model serving. This project validates the core algorithm
and measures where speedup actually comes from — acceptance rate, draft cost,
and the gamma tradeoff — rather than just implementing it.


