"""AnyJev L0 as a Decision Index engine: a training-free readout of a stock instruct model.

The harness (github.com/apolinario/decision-index) calls `engine(state, questions)` once per request. Every
question is read with AnyJev's prompt (`anyjev.readout.build_prompt`) and the restricted softmax over its answer
labels, from a vLLM server started with `--logprobs-mode processed_logprobs` (see serve.sh). No tokens are
generated. A prompt longer than the server's window is refused as Unsupported, never cut.

  L0       the options' cyclic rotations (`max_rotations` of them) averaged in log space, with the label-free
           running prior `anyjev.Decider` keeps for a question it has seen at least 8 times
  labels   letters A..Z up to 26 options. Above 26 (the harness allows 255) each option gets a letter and a
           digit, A0..A9, B0..B9, ..., and p(option) = p(letter) * p(digit | letter): one read restricted to
           the letters in use, one per letter restricted to the digits; `big_rotations` rotations, no prior
  yes/no   the question with its criteria ("Yes means: ... No means: ...")

The model sees each option's description (its key when the description is empty); answers are keyed back to
the harness's option keys.

    python -m decision_index run --engine anyjev_engine:AnyJevEngine \\
        --option base_url=http://127.0.0.1:8000 --option model=google/gemma-4-31B-it
"""
from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import json
import math
import urllib.error
import urllib.request

import numpy as np

from anyjev import Decider
from anyjev.backends.vllm import VLLMBackend
from anyjev.calibrate.permute import cyclic_shifts, marginalize, spread_order
from anyjev.question import MAX_OPTIONS, Question
from anyjev.readout import DEFAULT_SYSTEM, LETTERS, LabelTokenError, build_prompt, map_label_tokens, render_chat
from anyjev.state import render_state
from decision_index.engines.base import Engine, Unsupported, text

DIGITS = "0123456789"
MAX_TWO_LEVEL = len(LETTERS) * len(DIGITS)


class _Backend(VLLMBackend):
    """VLLMBackend reading labels by token id, a label missing from the reply taken as probability 0.

    With --logprobs-mode processed_logprobs the top-K entries come after the restriction to the K label tokens, so
    a label can be missing only when its log-probability is -inf and a masked token ties with it. Only a reply
    with no label at all refuses."""

    def _one(self, prompt, ids):
        body = {"model": self.name, "prompt": prompt, "max_tokens": 1, "temperature": 0.0,
                "logprobs": len(ids), "allowed_token_ids": list(ids), "return_tokens_as_token_ids": True}
        req = urllib.request.Request(self.base_url + "/v1/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            top = json.load(r)["choices"][0]["logprobs"]["top_logprobs"][0] or {}
        lp = {int(k.split(":", 1)[1]): v for k, v in top.items() if k.startswith("token_id:")}
        out = np.array([lp.get(t, -math.inf) for t in ids], dtype=np.float64)
        if not np.isfinite(out).any():
            raise LabelTokenError("the server returned no label among %s" % sorted(top))
        return out


def _restricted(values):
    v = np.asarray(values, dtype=np.float64)
    m = np.max(v)
    if not math.isfinite(m):
        raise RuntimeError("no finite label log-probability came back")
    return v - (m + np.log(np.exp(v - m).sum()))


class AnyJevEngine(Engine):
    name = "anyjev"
    latency = ("In-process request wall time including prompt construction and every vLLM read the readout makes "
               "(one per rotation, plus one per letter in use above 26 options); excludes server startup.")

    def __init__(self, base_url, model, tokenizer=None, level="L0", max_rotations=2, big_rotations=2,
                 max_model_len=None, workers=32, timeout=600, **options):
        super().__init__(**options)
        if level not in ("raw", "L0"):
            raise ValueError("level must be raw or L0")
        self.level = level
        self.max_rotations = int(max_rotations) if max_rotations else None
        self.big_rotations = int(big_rotations)
        self.backend = _Backend(base_url, model, tokenizer_name=tokenizer, workers=int(workers), timeout=float(timeout))
        self.tok = self.backend.tokenizer
        self.decider = Decider(self.backend, level=level, max_permutations=self.max_rotations)
        self.url = base_url.rstrip("/") + "/v1/completions"
        self.model = model
        self.timeout = float(timeout)
        self.max_model_len = int(max_model_len) if max_model_len else self._server_window()
        self.pool = cf.ThreadPoolExecutor(int(workers))
        self.letter_ids = map_label_tokens(self.tok, list(LETTERS))
        self.digit_ids = map_label_tokens(self.tok, list(DIGITS))
        self.provenance = {"kind": "inference technique", "package": "anyjev==0.3.0", "model": model, "level": level,
                           "max_rotations": self.max_rotations, "big_rotations": self.big_rotations,
                           "max_model_len": self.max_model_len,
                           "policy": "AnyJev prompt and restricted label readout; letters up to 26 options, "
                                     "letter+digit above; refuses a prompt longer than the server window"}

    # ------------------------------------------------------------------ harness entry point
    def __call__(self, state, questions):
        st = render_state(state)
        keys = list(questions)
        answers, small = {}, []
        for k in keys:
            q = questions[k]
            instr = render_state(q.get("instructions", ""))
            if q["type"] == "noul":
                crit = q.get("criteria") or {}
                if isinstance(crit, dict) and ("true" in crit or "false" in crit):
                    instr = instr + "\nYes means: %s\nNo means: %s" % (render_state(crit.get("true", "")),
                                                                     render_state(crit.get("false", "")))
                small.append((k, Question.noul(instr), None))
                continue
            if q["type"] != "choice":
                raise Unsupported("question type %r" % q["type"])
            opt_keys = list(q["criteria"])
            if len(opt_keys) > MAX_TWO_LEVEL:
                raise Unsupported("options per choice: %d above %d" % (len(opt_keys), MAX_TWO_LEVEL))
            shown = self._shown(opt_keys, q["criteria"])
            if len(opt_keys) <= MAX_OPTIONS:
                small.append((k, Question.choice(instr, shown), opt_keys))
            else:
                answers[k] = self._choice(opt_keys, self._two_level(st, Question("choice", instr, tuple(shown))))
        if small:
            for _, question, _ in small:
                self._check_len(st, question)
            # one Decider call for every question of the request: its prompts go to the server together
            decs = self.decider.decide(state, [question for _, question, _ in small])
            for (k, question, opt_keys), d in zip(small, decs):
                if question.kind == "noul":
                    answers[k] = {"type": "noul", "noul": float(d.p_true)}
                else:
                    answers[k] = self._choice(opt_keys, np.asarray(d.probs, dtype=np.float64))
        return {"model": self.model, "answers": {k: answers[k] for k in keys}}, None

    @staticmethod
    def _choice(opt_keys, probs):
        probs = probs / probs.sum()
        j = int(np.argmax(probs))
        return {"type": "choice", "choice": opt_keys[j], "probabilities": {k: float(p) for k, p in zip(opt_keys, probs)}}

    @staticmethod
    def _shown(keys, criteria):
        shown = [text(criteria[k]).strip() or str(k) for k in keys]
        if len(set(shown)) < len(shown):                # equal descriptions: tell them apart by key
            seen = {}
            for s in shown:
                seen[s] = seen.get(s, 0) + 1
            shown = ["%s [%s]" % (s, k) if seen[s] > 1 else s for s, k in zip(shown, keys)]
        return shown

    # ------------------------------------------------------------------ window
    def _server_window(self):
        with urllib.request.urlopen(urllib.request.Request(self.url.replace("/completions", "/models")), timeout=60) as r:
            return int(json.load(r)["data"][0]["max_model_len"])

    def _check_len(self, st, question):
        # every rotation of a question has the same length; one rendering is enough
        n = len(self.tok(render_chat(self.tok, build_prompt(st, question, list(range(question.k)))),
                         add_special_tokens=False).input_ids)
        if n + 1 > self.max_model_len:
            raise Unsupported("maximum context length: prompt of %d tokens, window %d" % (n, self.max_model_len))

    # ------------------------------------------------------------------ above 26 options
    def _prompt_ids(self, st, q, perm, codes):
        spec = build_prompt(st, q, perm, DEFAULT_SYSTEM, codes)
        spec = dataclasses.replace(spec, user=spec.user.replace(
            "Answer with the letter only.", "Answer with the label only: its letter and its digit."))
        ids = self.tok(render_chat(self.tok, spec), add_special_tokens=False).input_ids
        if len(ids) + 2 > self.max_model_len:
            raise Unsupported("maximum context length: prompt of %d tokens, window %d" % (len(ids), self.max_model_len))
        return ids

    def _complete(self, prompts, allowed):
        body = dict(model=self.model, prompt=prompts, max_tokens=1, temperature=0.0, logprobs=len(allowed),
                    allowed_token_ids=list(allowed), return_tokens_as_token_ids=True)
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                choices = sorted(json.load(r)["choices"], key=lambda c: c["index"])
        except urllib.error.HTTPError as e:
            msg = e.read()[:500].decode(errors="replace")
            if e.code in (400, 413) and "context" in msg.lower():
                raise Unsupported(msg) from None
            raise RuntimeError("vLLM %s: %s" % (e.code, msg)) from None
        out = []
        for c in choices:
            top = c["logprobs"]["top_logprobs"][0] or {}
            lp = {int(k.split(":", 1)[1]): v for k, v in top.items() if k.startswith("token_id:")}
            out.append(np.array([lp.get(t, -math.inf) for t in allowed], dtype=np.float64))
        return out

    def _read_two_level(self, ids, K):
        """log p(position j) for the K positions of one prompt."""
        n_letters = (K + len(DIGITS) - 1) // len(DIGITS)
        letters = self.letter_ids[:n_letters]
        f_letter = self.pool.submit(self._complete, [ids], letters)
        f_digit = self.pool.submit(self._complete, [ids + [t] for t in letters], self.digit_ids)
        lp_letter = _restricted(f_letter.result()[0])
        lp = np.empty(K)
        for a, row in enumerate(f_digit.result()):
            used = min(len(DIGITS), K - a * len(DIGITS))
            # a partly used last group whose digits all came back at zero probability keeps its letter's mass,
            # spread evenly over its options
            digits = row[:used] if np.isfinite(row[:used]).any() else np.zeros(used)
            lp[a * len(DIGITS):a * len(DIGITS) + used] = lp_letter[a] + _restricted(digits)
        return lp

    def _two_level(self, st, q):
        K = q.k
        codes = [LETTERS[j // len(DIGITS)] + DIGITS[j % len(DIGITS)] for j in range(K)]
        if self.level == "raw":
            perms = [list(range(K))]
        else:
            shifts = cyclic_shifts(K)
            perms = [shifts[t] for t in spread_order(K)[: self.big_rotations]]
        reads = [self._read_two_level(self._prompt_ids(st, q, perm, codes), K) for perm in perms]
        p_pos = np.exp(np.stack(reads))                                  # [P, K] by position
        return np.asarray(marginalize(p_pos, perms, "logmean"), dtype=np.float64)

    def close(self):
        self.pool.shutdown(wait=False)
