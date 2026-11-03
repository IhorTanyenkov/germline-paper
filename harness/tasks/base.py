import re
from dataclasses import dataclass, field


@dataclass
class Constraint:
    cid: str              # stable internal id
    text: str             # as shown in tau (without the "Cn:" prefix)
    inconvenient: bool
    probe_patterns: list  # regexes; any match => constraint present in a restatement
    label: str = ""       # "C1".."CK", assigned at task build

    def present_in(self, text):
        t = normalize_probe_text(text or "")
        return any(re.search(p, t, re.I | re.S) for p in self.probe_patterns)


_NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
              "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def normalize_probe_text(t):
    """Strip markdown emphasis/code markers and spell number words as digits so probes match
    '**exactly four sentences**' the same as 'exactly 4 sentences'."""
    t = re.sub(r"[*`]+", "", t)          # bold/italic stars, code ticks
    t = re.sub(r"(?<!\w)_+|_+(?!\w)", "", t)  # emphasis underscores, but keep snake_case identifiers
    t = re.sub(r"[“”]", '"', t)
    t = re.sub(r"[‘’]", "'", t)
    t = re.sub(r"\b(" + "|".join(_NUM_WORDS) + r")\b", lambda m: str(_NUM_WORDS[m.group(1).lower()]), t, flags=re.I)
    return t


@dataclass
class Task:
    task_id: str
    family: str
    intro: str
    constraints: list
    kappa: str
    payload: dict = field(default_factory=dict)
    attachment: str = ""   # data every agent gets (soma, not spec)
    tau: str = ""

    def finalize(self):
        for i, c in enumerate(self.constraints, 1):
            c.label = f"C{i}"
        lines = [self.intro, "Constraints:"] + [f"{c.label}: {c.text}" for c in self.constraints]
        tail = self.payload.get("tail")
        if tail:
            lines.append(tail)
        self.tau = "\n".join(lines)
        return self

    def check_output(self, text):
        raise NotImplementedError

    def reference_output(self):
        raise NotImplementedError


def final_part(text):
    """Text after the last 'FINAL:' marker (case-insensitive, optional markdown around it);
    whole text if no marker."""
    ms = list(re.finditer(r"[*_#\s]*final[*_]*\s*:[*_]*", text, re.I))
    return text[ms[-1].end():] if ms else text


def extract_json(text):
    """Last parseable JSON object in the final part."""
    import json
    t = final_part(text)
    # try progressively from each '{' to the last '}' candidates, prefer the last object
    ends = [m.start() for m in re.finditer(r"}", t)]
    starts = [m.start() for m in re.finditer(r"{", t)]
    for e in reversed(ends):
        for s in starts:
            if s < e:
                try:
                    return json.loads(t[s:e + 1])
                except Exception:
                    continue
    return None


def extract_code(text):
    """Last python code block in the final part (fallback: last block anywhere, then raw text)."""
    t = final_part(text)
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", t, re.S) or re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    return blocks[-1] if blocks else t
