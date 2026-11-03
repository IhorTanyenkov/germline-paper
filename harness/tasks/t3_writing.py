"""T3: constrained writing, all constraints regex/program checkable."""
import random
import re
from .base import Task, Constraint, final_part

VOCAB = {
    "A": ["autumn", "air", "all", "along", "across"], "B": ["birds", "bright", "brook", "balmy", "big"],
    "C": ["calm", "cold", "clouds", "crisp", "chill"], "D": ["dark", "dawn", "dusk", "dry", "dim"],
    "F": ["fog", "frost", "fast", "far", "fall"], "G": ["gusts", "gray", "grass", "glow", "gold"],
    "H": ["hills", "high", "hush", "hot", "hard"], "I": ["icy", "ivory", "inland", "into", "its"],
    "L": ["light", "long", "low", "lull", "lush"], "M": ["moon", "mist", "mild", "murky", "march"],
    "N": ["night", "north", "now", "nimbly", "noon"], "O": ["old", "only", "out", "our", "owls"],
    "P": ["path", "plain", "pond", "pull", "push"], "R": ["rain", "roads", "rocks", "runs", "rolls"],
    "S": ["sky", "sun", "snow", "still", "slow"], "T": ["this", "that", "trails", "truly", "turns"],
    "U": ["up", "upon", "until", "unto", "us"], "V": ["vast", "vivid", "valid", "vows", "vials"],
    "W": ["wind", "warm", "wild", "walks", "with"], "Y": ["young", "yards", "yawns", "yarn", "you"],
}
FILL = ["and", "with", "on", "in", "at", "to", "of", "by", "for", "from", "its", "our", "this", "that",
        "a", "as", "all", "cold", "dark", "light", "old", "small", "big", "sky", "sun", "moon", "hill",
        "wind", "rain", "snow", "fog", "calm", "slow", "fast", "long", "runs", "falls", "flows", "turns",
        "stays", "rolls", "drifts", "hangs", "sits", "still", "high", "low", "warm", "soft", "hard", "far",
        "city", "roads", "night", "dawn", "dusk", "morning", "autumn", "winds", "clouds", "storm", "coast"]
PATTERNS = {3: ["FOG", "SUN", "OWL", "ASH", "ICY"], 4: ["WIND", "CALM", "SNOW", "FROG", "HILL"],
            5: ["STORM", "CLOUD", "RIVAL", "BIRDS", "MOUTH"], 6: ["WINTRY", "ISLAND", "FROSTY", "SPRING", "BRIGHT"]}
# Calibrated defaults (E0, 2026-10-31): Qwen3-32B d=1 acceptance ≈ 0.80 with these.
# Original settings: words=exact lipogram=e N=4-6 → acceptance 0.00 (lipogram and exact count infeasible).
CFG = {"words": "range", "lipogram": "lower", "N": (3, 4), "width": 3}
# --variant t3.words=exact t3.lipogram=e|the t3.N=4-6 t3.width=2

TOPICS = ["a mountain town in winter", "a river at dawn", "a city at night", "an autumn forest", "a coastal storm"]
for _w in FILL + sum(VOCAB.values(), []) + sum(PATTERNS.values(), []):
    assert "e" not in _w.lower(), _w


def _sentences(text):
    parts = [p.strip() for p in text.strip().split(".")]
    return [p for p in parts if p]


class T3Task(Task):
    def check_output(self, text):
        p = self.payload
        text = final_part(text).strip().strip("`").strip()
        sents = _sentences(text)
        per = {}
        for c in self.constraints:
            if c.cid == "count":
                per[c.cid] = len(sents) == p["N"]
            elif c.cid == "words":
                if p["words"] == "exact":
                    per[c.cid] = bool(sents) and all(len(s.split()) == p["W"] for s in sents)
                else:
                    wd = p.get("width", 2)
                    per[c.cid] = bool(sents) and all(p["W"] - wd <= len(s.split()) <= p["W"] + wd for s in sents)
            elif c.cid == "init":
                per[c.cid] = len(sents) == p["N"] and all(s[0].upper() == p["pattern"][i] for i, s in enumerate(sents))
            elif c.cid == "period":
                per[c.cid] = text.strip().endswith(".") and not re.search(r"[!?;:]", text)
            elif c.cid == "noe":
                if p["lipogram"] == "e":
                    per[c.cid] = "e" not in text.lower()
                elif p["lipogram"] == "lower":
                    per[c.cid] = text == text.lower()
                else:
                    per[c.cid] = not re.search(r"\bthe\b", text, re.I)
            elif c.cid == "kw":
                per[c.cid] = all(re.search(rf"\b{k}\b", text, re.I) for k in p["kws"])
            elif c.cid == "nonum":
                per[c.cid] = not re.search(r"\d", text)
            elif c.cid == "nocomma":
                per[c.cid] = "," not in text
        return {"passed": all(per.values()), "per_constraint": per, "details": f"sents={len(sents)}"}

    def reference_output(self):
        p = self.payload
        rng = random.Random(p["seed"] + 1)
        out = []
        for i, L in enumerate(p["pattern"]):
            first = rng.choice(VOCAB[L])
            words = [first if p["lipogram"] == "lower" else first.capitalize()] + [rng.choice(FILL) for _ in range(p["W"] - 1)]
            if i < len(p["kws"]):
                words[1 + rng.randrange(p["W"] - 1)] = p["kws"][i]
            out.append(" ".join(words) + ".")
        return " ".join(out)


def make_task(idx, K, seed):
    rng = random.Random(seed)
    N = rng.randint(*CFG["N"])
    wd = int(CFG["width"])
    W = rng.choice([7, 8, 9])
    pattern = rng.choice(PATTERNS[N])
    kws = rng.sample([w for w in FILL if len(w) > 3], 2)
    topic = rng.choice(TOPICS)
    letters = ", ".join(pattern)
    cons = [
        Constraint("count", f"Exactly {N} sentences.", False, [rf"\b{N}\b.{{0,40}}sentenc|sentenc.{{0,40}}\b{N}\b"]),
        (Constraint("words", f"Every sentence has exactly {W} words.", False, [rf"\b{W}\b.{{0,30}}word|word.{{0,30}}\b{W}\b"])
         if CFG["words"] == "exact" else
         Constraint("words", f"Every sentence has between {W-wd} and {W+wd} words.", False,
                    [rf"\b{W-wd}\b.{{0,15}}\b{W+wd}\b|between.{{0,25}}words|words.{{0,25}}between"])),
        Constraint("init", f'Sentence i begins with the i-th letter of "{pattern}" (so the first letters are: {letters}).', False,
                   [pattern, r"first letter|begins? with|starts? with"]),
        Constraint("period", "Every sentence ends with a period; no other sentence-ending punctuation.", False,
                   [r"period|full stop"]),
    ]
    if K >= 6:
        noe = (Constraint("noe", 'The letter "e" must not appear anywhere in the text.', True,
                          [r"letter\s*[\"'“]?e[\"'”]?|\bno\b.{0,20}\be\b|without.{0,20}\be\b|\be\b.{0,20}(forbidden|banned|avoid)|lipogram"])
               if CFG["lipogram"] == "e" else
               Constraint("noe", "The entire text must be in lowercase; no capital letters anywhere, not even at sentence starts.", True,
                          [r"lower ?case|no capital|capital letters"])
               if CFG["lipogram"] == "lower" else
               Constraint("noe", 'Do not use the word "the" anywhere in the text.', True,
                          [r"[\"'“]the[\"'”]|\bword\s+the\b|\bthe\b.{0,20}(forbidden|banned|not allowed|must not|avoid)"]))
        cons += [noe,
                 Constraint("kw", f'The words "{kws[0]}" and "{kws[1]}" must each appear at least once.', False,
                            [rf"\b{kws[0]}\b.*\b{kws[1]}\b|\b{kws[1]}\b.*\b{kws[0]}\b"])]
    if K >= 8:
        cons += [Constraint("nonum", "No digits anywhere in the text.", False, [r"digit|numeral|\bnumbers?\b"]),
                 Constraint("nocomma", "No commas anywhere in the text.", True, [r"comma"])]
    t = T3Task(task_id=f"T3-{idx:03d}-K{K}", family="T3",
               intro=f"Write a short text about {topic}. Output only the text.",
               constraints=cons,
               kappa="class: constrained composition; rigor: formal constraints outrank content quality; done: every constraint mechanically satisfied",
               payload={"N": N, "W": W, "pattern": pattern, "kws": kws, "seed": seed,
                        "words": CFG["words"], "lipogram": CFG["lipogram"], "width": wd})
    return t.finalize()
