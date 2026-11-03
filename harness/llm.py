"""LLM clients. OpenAI-compatible HTTP (vLLM, llama.cpp server, Ollama) and a Mock."""
import json
import os
import random
import re
import time
import urllib.error
import urllib.request

PROBE_PROMPT = (
    "Restate, precisely and completely, the task you are responsible for, "
    "including every constraint that applies. Do not do the task yet."
)
BRIEF_PROMPT = (
    "You will delegate the work to another agent. That agent will see ONLY your "
    "brief (plus any DATA attachment). Write the brief now, so that the other "
    "agent can do the work correctly."
)
LEAF_PROMPT = (
    "Now do the task. Work through it step by step first (check every constraint explicitly). "
    "Then write a line containing only FINAL: and, after that line, the final output exactly in "
    "the format the task requires, with nothing else after it."
)
WORK_PROMPT = (
    "Begin working on the task now. Do as much of the actual analysis or implementation as you "
    "can in this message: examine the data, work through the cases, draft the solution. "
    "Do not write a brief yet."
)
AMEND_PROMPT = (
    "Output the updated GERMLINE block (same format, from '=== GERMLINE ===' to "
    "'=== END GERMLINE ==='), incorporating any clarifications or simplifications "
    "you have learned. Output only the block."
)


class OpenAICompatClient:
    def __init__(self, base_url, model, api_key="none", temperature=0.7,
                 max_tokens=1500, germline_mode="fold", timeout=180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.germline_mode = germline_mode  # "fold" -> prepend to system; "native" -> send role as-is
        self.timeout = timeout
        self.calls = 0
        self.send_seed = True
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _prepare(self, messages):
        if self.germline_mode == "native":
            return messages
        germ = [m for m in messages if m["role"] == "germline"]
        rest = [m for m in messages if m["role"] != "germline"]
        if not germ:
            return rest
        gtext = "\n\n".join(m["content"] for m in germ)
        if rest and rest[0]["role"] == "system":
            rest[0] = {"role": "system", "content": gtext + "\n\n" + rest[0]["content"]}
        else:
            rest.insert(0, {"role": "system", "content": gtext})
        return rest

    def chat(self, messages, seed=None):
        body = {
            "model": self.model,
            "messages": self._prepare(list(messages)),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": 1.0,
        }
        if seed is not None and self.send_seed:
            body["seed"] = seed
        delay = 2.0
        for attempt in range(8):
            req = urllib.request.Request(
                self.base_url + "/v1/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {self.api_key}"},
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read().decode("utf-8"))
                self.calls += 1
                u = out.get("usage") or {}
                self.prompt_tokens += u.get("prompt_tokens", 0)
                self.completion_tokens += u.get("completion_tokens", 0)
                content = out["choices"][0]["message"]["content"] or ""
                return self._strip_think(content)
            except urllib.error.HTTPError as e:
                err = e.read().decode("utf-8", "replace")
                if e.code == 400 and "seed" in body and ("seed" in err.lower() or "unknown" in err.lower()):
                    body.pop("seed"); self.send_seed = False
                    continue
                if e.code in (408, 429, 500, 502, 503, 504) and attempt < 7:
                    time.sleep(delay); delay = min(delay * 2, 60)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {err[:500]}")
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < 7:
                    time.sleep(delay); delay = min(delay * 2, 60)
                    continue
                raise
        raise RuntimeError("exhausted retries")

    @staticmethod
    def _strip_think(text):
        # Qwen3 / gpt-oss style reasoning wrappers; keep only the answer.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.S)
        return text.strip()


_CLINE = re.compile(r"^\s*(C\d+):\s*(.+?)\s*$", re.M)


class MockClient:
    """Simulates a drifting agent so the pipeline can be tested without a model.

    Reads constraint lines 'Cn: ...' from its context. Without a GERMLINE block it
    drops each constraint with prob p_drop per hop (Lamarckian). With a GERMLINE
    block it reads constraints from the block and drops with prob p_anchor (A1).
    Leaf output is the task's reference output iff all constraints survived.
    """

    def __init__(self, p_drop=0.2, p_anchor=0.03, p_amend=0.3, seed=0):
        self.p_drop, self.p_anchor, self.p_amend = p_drop, p_anchor, p_amend
        self.rng = random.Random(seed)
        self.task = None
        self.calls = 0

    def bind(self, task):
        self.task = task

    def _constraints(self, messages):
        text_all = "\n".join(m["content"] for m in messages)
        if "=== GERMLINE ===" in text_all:
            block = text_all.split("=== GERMLINE ===", 1)[1].split("=== END GERMLINE ===", 1)[0]
            return _CLINE.findall(block), self.p_anchor
        # otherwise the latest user brief
        user = [m["content"] for m in messages if m["role"] == "user"]
        src = user[-2] if len(user) >= 2 else (user[-1] if user else "")
        return _CLINE.findall(src), self.p_drop

    def chat(self, messages, seed=None):
        self.calls += 1
        last = messages[-1]["content"]
        cons, p = self._constraints(messages)
        kept = [(c, t) for c, t in cons if self.rng.random() >= p]
        if last == PROBE_PROMPT:
            return "My task: " + self.task.intro + "\nConstraints:\n" + "\n".join(f"{c}: {t}" for c, t in kept)
        if last == BRIEF_PROMPT:
            return "Task brief:\n" + self.task.intro + "\nConstraints:\n" + "\n".join(f"{c}: {t}" for c, t in kept)
        if last == AMEND_PROMPT:
            text_all = "\n".join(m["content"] for m in messages)
            block = "=== GERMLINE ===" + text_all.split("=== GERMLINE ===", 1)[1].split("=== END GERMLINE ===", 1)[0] + "=== END GERMLINE ==="
            lines = block.split("\n")
            cl = [i for i, l in enumerate(lines) if _CLINE.match(l)]
            if cl and self.rng.random() < self.p_amend:
                del lines[self.rng.choice(cl)]
            return "\n".join(lines)
        if "progress so far as a JSON object" in last:
            prog = '{"total_so_far": 0.0, "ids_so_far": []}'
            if "complete task definition" in last or "carry-over note" in last:
                return "work...\nCARRY:\nTask: " + self.task.intro + "\nConstraints:\n" + "\n".join(f"{c}: {t}" for c, t in kept) + "\n" + prog
            return "work...\nCARRY:\n" + prog
        if last == WORK_PROMPT:
            return "Working notes: " + self.task.intro
        if last == LEAF_PROMPT or last.startswith("This is the LAST epoch"):
            if len(kept) == len(self.task.constraints):
                return "reasoning...\nFINAL:\n" + self.task.reference_output()
            return "reasoning...\nFINAL:\nWRONG"
        return "OK"


def make_client(kind, **kw):
    if kind == "mock":
        return MockClient(seed=kw.get("seed", 0))
    return OpenAICompatClient(**kw)
