"""T1: constrained aggregation over prose support tickets."""
import random
import re
from .base import Task, Constraint, extract_json


def _norm_ids(ids):
    """Accept 'T-008', 'T-8', 8, '8' as the same ticket id."""
    out = []
    for x in ids:
        m = re.search(r"(\d+)", str(x))
        out.append(f"T-{int(m.group(1)):03d}" if m else str(x))
    return out

# Calibrated defaults (E0, 2026-10-31): Qwen3-32B d=1 acceptance ≈ 0.55 with these.
# Original (uncalibrated) settings: records=60-90 zero_share=high output=full → acceptance 0.00.
CFG = {"records": (12, 16), "zero_share": "low", "output": "simple"}
# --variant t1.records=25-35  t1.zero_share=high  t1.output=full

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
COUNTRIES = {"DE": ("Germany", ["Berlin", "Munich", "Hamburg"]),
             "AT": ("Austria", ["Vienna", "Graz", "Linz"]),
             "CZ": ("Czechia", ["Prague", "Brno"]),
             "PL": ("Poland", ["Warsaw", "Krakow"]),
             "FR": ("France", ["Paris", "Lyon"])}
NAMES = ["K. Weber", "J. Novak", "L. Huber", "M. Kowalski", "A. Dubois", "P. Fischer",
         "S. Horvath", "T. Mayer", "E. Laurent", "R. Zelenka", "N. Bauer", "O. Wagner"]
STATUSES = ["resolved", "open", "cancelled", "escalated"]
KEYWORDS = ["shipping", "damaged", "refund", "size"]
COMPLAINTS = ["Complaint about {kw} delay.", "Item arrived {kw2}.", "Wrong {kw3} delivered.",
              "Asked for a {kw4} twice.", "Package lost during {kw}.", "Color differs from photo.",
              "Late delivery, no tracking.", "Box crushed, {kw2} goods inside."]


def _suffix(d):
    return "th" if 11 <= d <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")


def _fmt_date(rng, d, m, allow_slash):
    forms = [lambda: f"{d} {MONTHS[m-1][:3]}", lambda: f"{MONTHS[m-1][:3]} {d}",
             lambda: f"{d}{_suffix(d)} of {MONTHS[m-1]}", lambda: f"2024-{m:02d}-{d:02d}"]
    if allow_slash and d <= 12 and d != m:  # ambiguous on purpose
        forms.append(lambda: f"{d:02d}/{m:02d}")
    return rng.choice(forms)()


class T1Task(Task):
    def check_output(self, text):
        gt = self.payload["gt"]
        ids = {c.cid for c in self.constraints}
        obj = extract_json(text)
        per = {}
        if not isinstance(obj, dict):
            return {"passed": False, "per_constraint": {c.cid: False for c in self.constraints},
                    "details": "no json"}
        keys_ok = set(obj.keys()) == ({"total", "ids"} if self.payload.get("simple") else {"total", "ids", "excluded_count"})
        out_ids = _norm_ids(obj["ids"]) if isinstance(obj.get("ids"), list) else None
        ids_ok = out_ids is not None and sorted(out_ids) == sorted(gt["ids"])
        tol = 0.005 if "round2" in ids else 0.011
        try:
            total_ok = abs(float(obj.get("total")) - gt["total"]) <= tol
        except Exception:
            total_ok = False
        exc_ok = True if self.payload.get("simple") else obj.get("excluded_count") == gt["excluded_count"]
        data_ok = ids_ok and total_ok and exc_ok
        for c in self.constraints:
            if c.cid in ("month", "country", "cancelled", "keyword", "slash"):
                per[c.cid] = data_ok
            elif c.cid == "json":
                per[c.cid] = keys_ok
            elif c.cid == "sorted":
                per[c.cid] = out_ids is not None and out_ids == sorted(out_ids)
            elif c.cid == "round2":
                t = obj.get("total")
                per[c.cid] = isinstance(t, (int, float)) and abs(round(t, 2) - t) < 1e-9
        return {"passed": all(per.values()) and data_ok, "per_constraint": per,
                "details": f"ids_ok={ids_ok} total_ok={total_ok} exc_ok={exc_ok} keys_ok={keys_ok}"}

    def reference_output(self):
        import json
        return json.dumps(self.payload["gt"])


def make_task(idx, K, seed, n_records=None):
    rng = random.Random(seed)
    m = rng.randint(2, 11)
    cc = rng.sample(list(COUNTRIES), 2)
    kw = rng.choice(KEYWORDS)
    n = n_records or rng.randint(*CFG["records"])
    contrib = []  # (ticket id, included, amount) per record, in order — for prefix ground truth
    use_keyword = K >= 6
    use_slash = K >= 8

    recs, gt_ids, total, excluded = [], [], 0.0, 0
    for i in range(1, n + 1):
        mm = rng.choice([m - 1, m, m, m + 1])
        d = rng.randint(1, 28)
        c = rng.choice(list(COUNTRIES))
        city = rng.choice(COUNTRIES[c][1])
        st = rng.choices(STATUSES, weights=[5, 3, 2, 1])[0]
        amt = round(rng.choice([0, 0, rng.uniform(5, 300)]) if CFG["zero_share"] == "high"
                    else (0.0 if rng.random() < 0.15 else rng.uniform(5, 300)), 2)
        tmpl = rng.choice(COMPLAINTS)
        comp = tmpl.format(kw=rng.choice(KEYWORDS), kw2=rng.choice(["damaged", "crushed"]),
                           kw3=rng.choice(["size", "color"]), kw4=rng.choice(["refund", "replacement"]))
        recs.append(f"T-{i:03d}: Opened {_fmt_date(rng, d, mm, use_slash)} by {rng.choice(NAMES)} "
                    f"({city}, {c}). Status: {st}. Refund €{amt:.2f}. {comp}")
        included = False
        if mm == m and c in cc:
            kw_hit = use_keyword and kw.lower() in comp.lower()
            if st == "cancelled" or kw_hit:
                excluded += 1
            else:
                gt_ids.append(f"T-{i:03d}")
                total += amt
                included = True
        contrib.append((f"T-{i:03d}", included, amt))
    simple = CFG["output"] == "simple"
    gt = {"total": round(total, 2), "ids": sorted(gt_ids)} if simple else \
         {"total": round(total, 2), "ids": sorted(gt_ids), "excluded_count": excluded}

    cnames = " or ".join(f"{COUNTRIES[c][0]} ({c})" for c in cc)
    cons = [
        Constraint("month", f"Include only tickets opened in {MONTHS[m-1]} 2024.", False,
                   [MONTHS[m-1], MONTHS[m-1][:3] + r"\b"]),
        Constraint("country", f"Include only customers in {cnames}.", False,
                   [COUNTRIES[cc[0]][0], COUNTRIES[cc[1]][0], rf"\b{cc[0]}\b", rf"\b{cc[1]}\b"]),
        Constraint("cancelled", 'Exclude tickets whose status is "cancelled".', False, [r"cancel"]),
        Constraint("json", 'Output exactly one JSON object with keys "total", "ids" and nothing else.' if simple else
                   'Output exactly one JSON object with keys "total", "ids", "excluded_count" and nothing else.',
                   False, [r"json"]),
    ]
    if use_keyword:
        cons += [Constraint("keyword", f'Exclude any ticket whose complaint text mentions "{kw}" anywhere.', True, [kw]),
                 Constraint("sorted", '"ids" must be sorted in ascending order.', False, [r"sort|ascend"])]
    if use_slash:
        cons += [Constraint("slash", "Dates written with a slash are day/month (DD/MM), never month/day.", True,
                            [r"dd/mm|day[ /-]?month|slash"]),
                 Constraint("round2", '"total" must be rounded to exactly 2 decimal places.', False,
                            [r"2 decimal|two decimal|round"])]
    tail = None if simple else (
        f'"excluded_count" is the number of tickets from {MONTHS[m-1]} in those countries that were '
        f'excluded by the status rule{" or the keyword rule" if use_keyword else ""}.')
    t = T1Task(task_id=f"T1-{idx:03d}-K{K}", family="T1",
               intro="Compute a refund total from the support tickets in the DATA attachment.",
               constraints=cons,
               kappa="class: exact data aggregation; rigor: every stated filter is binding; done: output passes every stated constraint",
               payload={"gt": gt, "tail": tail, "simple": simple, "contrib": contrib},
               attachment="All tickets below were opened in 2024.\n" + "\n".join(recs))
    return t.finalize()
