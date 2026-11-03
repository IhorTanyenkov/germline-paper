"""T2: constrained code. Static (AST) + dynamic (subprocess) checks."""
import ast
import json
import os
import random
import subprocess
import sys
import tempfile
from .base import Task, Constraint, extract_code

TEMPLATES = {
    "running_max": dict(params=["xs"], arg_type="list", desc="return the list of prefix maxima of xs",
                        empty=("[]", "[]"), banned=["max", "sorted", "sort"],
                        ref='''def running_max(xs):
    """Returns the list of prefix maxima of xs."""
    if not isinstance(xs, list):
        raise TypeError("xs must be a list")
    out = []
    cur = None
    for x in xs:
        if cur is None or x > cur:
            cur = x
        out.append(cur)
    return out
''', gen=lambda r: ([r.randint(-20, 20) for _ in range(r.randint(1, 10))],)),
    "count_pairs": dict(params=["xs", "k"], arg_type="list",
                        desc="return the number of index pairs i<j with xs[i]+xs[j]==k",
                        empty=("[]", "0"), banned=["sorted", "sort", "itertools"],
                        ref='''def count_pairs(xs, k):
    """Returns the number of index pairs i<j with xs[i]+xs[j]==k."""
    if not isinstance(xs, list):
        raise TypeError("xs must be a list")
    seen = {}
    total = 0
    for x in xs:
        total += seen.get(k - x, 0)
        seen[x] = seen.get(x, 0) + 1
    return total
''', gen=lambda r: ([r.randint(-5, 5) for _ in range(r.randint(1, 12))], r.randint(-4, 4))),
    "rotate_right": dict(params=["xs", "n"], arg_type="list", desc="return xs rotated right by n positions",
                         empty=("[]", "[]"), banned=["reversed", "deque", "sorted"],
                         ref='''def rotate_right(xs, n):
    """Returns xs rotated right by n positions."""
    if not isinstance(xs, list):
        raise TypeError("xs must be a list")
    if not xs:
        return []
    m = len(xs)
    return [xs[(i - n) % m] for i in range(m)]
''', gen=lambda r: ([r.randint(0, 9) for _ in range(r.randint(1, 8))], r.randint(0, 12))),
    "dedupe_sorted": dict(params=["xs"], arg_type="list",
                          desc="return xs (already sorted) with duplicate values removed, order preserved",
                          empty=("[]", "[]"), banned=["set", "sorted", "sort", "dict"],
                          ref='''def dedupe_sorted(xs):
    """Returns xs with duplicate values removed, order preserved."""
    if not isinstance(xs, list):
        raise TypeError("xs must be a list")
    out = []
    for x in xs:
        if not out or out[-1] != x:
            out.append(x)
    return out
''', gen=lambda r: (sorted(r.randint(0, 6) for _ in range(r.randint(1, 12))),)),
    "rle": dict(params=["s"], arg_type="str",
                desc="return the run-length encoding of s as a list of [char, count] pairs",
                empty=('""', "[]"), banned=["groupby", "itertools", "re"],
                ref='''def rle(s):
    """Returns the run-length encoding of s as a list of [char, count] pairs."""
    if not isinstance(s, str):
        raise TypeError("s must be a string")
    out = []
    for ch in s:
        if out and out[-1][0] == ch:
            out[-1][1] += 1
        else:
            out.append([ch, 1])
    return out
''', gen=lambda r: ("".join(r.choice("aabbbc") for _ in range(r.randint(1, 12))),)),
}

RUNNER = '''
import json, sys
RES = {"tests": [], "typeerror": None, "error": None}
try:
%(code)s
except Exception as e:
    RES["error"] = "define: " + repr(e)
    print(json.dumps(RES)); sys.exit(0)
tests = %(tests)s
f = globals().get(%(fname)r)
if f is None:
    RES["error"] = "missing function"; print(json.dumps(RES)); sys.exit(0)
for args, exp in tests:
    try:
        got = json.loads(json.dumps(f(*args)))
        RES["tests"].append(got == exp)
    except Exception as e:
        RES["tests"].append(False)
try:
    f(*%(bad)s)
    RES["typeerror"] = False
except TypeError:
    RES["typeerror"] = True
except Exception:
    RES["typeerror"] = False
print(json.dumps(RES))
'''


def _static(code, fname, params, cons_ids, banned):
    res = {k: True for k in cons_ids}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {k: False for k in cons_ids}, None
    fdefs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    target = next((f for f in fdefs if f.name == fname), None)
    if "sig" in res:
        res["sig"] = target is not None and [a.arg for a in target.args.args] == params
    if "imports" in res:
        res["imports"] = not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
    if "doc" in res:
        d = ast.get_docstring(target) if target else None
        res["doc"] = bool(d) and d.strip().split("\n")[0].startswith("Returns")
    if "norec" in res:
        res["norec"] = target is not None and not any(
            isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == fname
            for n in ast.walk(target))
    if "banned" in res:
        hit = False
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Name) and n.func.id in banned:
                    hit = True
                if isinstance(n.func, ast.Attribute) and n.func.attr in banned:
                    hit = True
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in n.names] + ([n.module] if isinstance(n, ast.ImportFrom) else [])
                if any(x in banned for x in names if x):
                    hit = True
        res["banned"] = not hit
    if "oneloop" in res:
        loops = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While, ast.AsyncFor)))
        res["oneloop"] = loops <= 1
    return res, target


class T2Task(Task):
    def check_output(self, text):
        p = self.payload
        code = extract_code(text)
        ids = [c.cid for c in self.constraints]
        static, target = _static(code, p["fname"], p["params"], ids, p["banned"])
        per = dict(static)
        dyn = {"tests": [], "typeerror": None, "error": "not run"}
        if target is not None:
            script = RUNNER % {"code": "\n".join("    " + l for l in code.split("\n")),
                               "tests": json.dumps(p["tests"]), "fname": p["fname"], "bad": json.dumps(p["bad"])}
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir=p.get("tmpdir")) as f:
                f.write(script)
                path = f.name
            try:
                out = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True, timeout=10)
                dyn = json.loads(out.stdout.strip().split("\n")[-1]) if out.stdout.strip() else {"tests": [], "typeerror": None, "error": out.stderr[-300:]}
            except Exception as e:
                dyn = {"tests": [], "typeerror": None, "error": repr(e)}
            finally:
                try:
                    os.unlink(path)
                except OSError:
                    pass
        tests_ok = bool(dyn["tests"]) and all(dyn["tests"])
        if "empty" in per:
            per["empty"] = bool(dyn["tests"]) and dyn["tests"][0]  # first test is the empty case
        if "typeerr" in per:
            per["typeerr"] = dyn.get("typeerror") is True
        return {"passed": tests_ok and all(per.values()), "per_constraint": per,
                "details": f"tests={sum(bool(t) for t in dyn['tests'])}/{len(dyn['tests'])} err={dyn.get('error')}"}

    def reference_output(self):
        return "```python\n" + self.payload["ref"] + "```"


def make_task(idx, K, seed, tmpdir=None):
    rng = random.Random(seed)
    name = rng.choice(list(TEMPLATES))
    T = TEMPLATES[name]
    ns = {}
    exec(T["ref"], ns)
    ref = ns[name]
    empty_arg = json.loads(T["empty"][0]) if T["arg_type"] == "list" else ""
    extra = [0] * (len(T["params"]) - 1)
    tests = [[[empty_arg] + extra, ref(empty_arg, *extra)]]
    for _ in range(6):
        args = list(T["gen"](rng))
        tests.append([args, json.loads(json.dumps(ref(*args)))])
    bad = [12345] + extra if T["arg_type"] == "list" else [["not", "a", "string"]] + extra
    sig = f"{name}({', '.join(T['params'])})"
    cons = [
        Constraint("sig", f"The function must be named exactly `{name}` with parameters ({', '.join(T['params'])}) in that order.",
                   False, [name]),
        Constraint("empty", f"For empty input ({T['empty'][0]}) return {T['empty'][1]}.", False, [r"empty"]),
        Constraint("imports", "Do not use any import statements.", False, [r"import"]),
        Constraint("doc", 'The docstring\'s first line must begin with the word "Returns".', False, [r"docstring|Returns"]),
    ]
    if K >= 6:
        cons += [Constraint("typeerr", f"Raise TypeError if the first argument is not a {T['arg_type']}.", False, [r"TypeError"]),
                 Constraint("norec", "Do not use recursion.", False, [r"recurs"])]
    if K >= 8:
        cons += [Constraint("banned", f"Do not use any of: {', '.join(T['banned'])}.", True,
                            [rf"\b{b}\b" for b in T["banned"]]),
                 Constraint("oneloop", "Use at most one loop statement (for/while) in total; comprehensions are allowed.", True,
                            [r"one loop|single loop|at most one|single pass|1 loop"])]
    t = T2Task(task_id=f"T2-{idx:03d}-K{K}", family="T2",
               intro=f"Implement a Python function `{sig}` that will {T['desc']}. Return it in a single ```python code block.",
               constraints=cons,
               kappa="class: constrained implementation; rigor: static constraints are binding, not stylistic; done: tests and every constraint pass",
               payload={"fname": name, "params": T["params"], "tests": tests, "bad": bad, "banned": T["banned"],
                        "ref": T["ref"], "tmpdir": tmpdir})
    return t.finalize()
