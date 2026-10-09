#!/bin/bash
# The model server the engine reads: stock google/gemma-4-31B-it, quantized to FP8 at load, on one GPU.
#   processed_logprobs  the label restriction applies before the logprobs are taken (required by the readout)
#   max-logprobs 32     up to 26 letter labels per read
#   prefix caching      the questions and rotations of one state share its prefill
# Gemma 4 needs vLLM >= 0.29. Our runs: vLLM 0.29.0, torch 2.13.0+cu129.
PORT=${PORT:-8000}
MAXLEN=${MAXLEN:-131072}     # our run used 131072 on two GPUs (tensor parallel 2); no public request needed more
TP=${TP:-1}
vllm serve google/gemma-4-31B-it --quantization fp8 --logprobs-mode processed_logprobs --max-logprobs 32 \
  --enable-prefix-caching --max-model-len $MAXLEN --tensor-parallel-size $TP --gpu-memory-utilization 0.90 \
  --max-num-seqs 256 --limit-mm-per-prompt '{"image": 0, "video": 0}' --host 127.0.0.1 --port $PORT
