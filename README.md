# AnyJev L0 · Gemma-4-31B

> [!NOTE]
> This repository is the engine as it was submitted to the Decision Index 0.3 board (commit `50476e9`, which the
> board links to); it is kept as it is. The engine is maintained in
> [nokia-applied-research/AnyJev, `examples/decision_index/`](https://github.com/nokia-applied-research/AnyJev/tree/main/examples/decision_index).

A training-free decision engine for the [Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index):
the stock [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) read through
[AnyJev](https://github.com/nokia-applied-research/AnyJev)'s L0 readout. No weights are trained or published here; the
model is Google's checkpoint, quantized to FP8 when vLLM loads it. Model card:
[morriszjm/AnyJev-L0-Gemma-4-31B](https://huggingface.co/morriszjm/AnyJev-L0-Gemma-4-31B).

## On the board

| Full score | public <sub>20%</sub> | same skills, private <sub>50%</sub> | new domains, private <sub>30%</sub> | latency median / mean / p95 |
|:--:|:--:|:--:|:--:|:--:|
| **60.22** | 59.79 | 59.13 | 52.98 | 98 / 260 / 1,080 ms |

5th of 149 entries by Full score on the Decision Index 0.3 board, 2026-10-12. The maintainers re-scored the public
run (59.79, as submitted), ran the two private parts and measured the latency themselves (single process, one RTX PRO
6000); their rerun agreed with the submitted answers on 98.18% of 827 rows.

## The readout

Each question is rendered with AnyJev's prompt and read in one forward pass per rotation; nothing is generated.

- **Restricted softmax.** The answer is the next-token distribution restricted to the option labels (letters A..Z),
  read from vLLM with `--logprobs-mode processed_logprobs`.
- **L0.** The options are shown in 2 cyclic rotations; the per-option log-probabilities are averaged across them, and
  a label-free running prior (AnyJev's batch prior, once a question has been seen 8 times) corrects label bias. No
  labelled data is used anywhere. The prior is kept per process, so the distribution for a question depends slightly
  on the requests that came before it.
- **More than 26 options** (up to 255 in the suite): each option gets a letter and a digit (A0..A9, B0..B9, ...), and
  p(option) = p(letter) · p(digit | letter), from one read restricted to the letters in use and one per letter
  restricted to the digits.
- **Yes/no questions** are shown with their criteria ("Yes means: … No means: …").

## Run it

```bash
pip install -r requirements.txt                # anyjev 0.3.0, vLLM 0.29.0; plus the Decision Index kit
bash serve.sh                                  # one GPU; TP=2 MAXLEN=131072 for our full-window setting
bash run.sh                                    # the public 0.3 suite, then the kit's scorer
```

The engine is `anyjev_engine:AnyJevEngine` with `level=L0 max_rotations=2 big_rotations=2` (the defaults).

## Settings of the submitted run

| | |
|---|---|
| model | `google/gemma-4-31B-it`, FP8 (`--quantization fp8`) |
| server | vLLM 0.29.0, torch 2.13.0+cu129, prefix caching on, `processed_logprobs`, max-model-len 131,072 |
| engine | `anyjev_engine.py` in this repository, anyjev 0.3.0 (see the note below) |
| readout | L0, 2 rotations (≤ 26 options) and 2 rotations (> 26 options), yes/no with criteria |
| kit | apolinario/decision-index @ `9eb2dbe2a358004c8782c66e40a83ac07b953fec`, edition 0.3 |
| hardware | NVIDIA H100 NVL (94 GB) |
| declared limits | a prompt over the server window (131,072 tokens) or a choice over 260 options is refused as unsupported; none occurred in the public suite |

## Results (public suite, edition 0.3)

| public index | knowledge | language | retrieval | tools | arts |
|---|---|---|---|---|---|
| **59.79** | 47.8 | 61.7 | 63.5 | 76.4 | 48.0 |

From the kit's `scores.json` of the complete run (140,178 requests, all answered, `"complete": true`); raw index
69.37, breadth 59.06.

Latency, single process, one request at a time, on a stratified 750-row sample, one H100 NVL, FP8:
median 123 ms, mean 460 ms, 80th percentile 801 ms, 95th percentile 1,854 ms. The board's own measurement, on one
RTX PRO 6000, is in the table above.

**Note on the engine.** The public run was made with the development version of `anyjev_engine.py`, which has three
more options (`noul_meanings`, `prompt_style`, `adaptive`) and records them in `environment.json`. At the values the run
used (`noul_meanings=1`, `prompt_style=anyjev`, `adaptive=false`) they are the behaviour of the engine in this
repository, which drops them. Rerun with this engine and anyjev 0.3.0 from PyPI on 5,192 rows of the suite, 4 answers
differed, all near-ties.

## Licence

Code in this repository: Apache-2.0. The model weights are Google's `google/gemma-4-31B-it`, released under Apache-2.0.
