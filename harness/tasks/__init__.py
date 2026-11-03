from . import t1_aggregation, t2_code, t3_writing

FAMILIES = {"T1": t1_aggregation.make_task, "T2": t2_code.make_task, "T3": t3_writing.make_task}


def make_tasks(families, n, Ks, base_seed=0, **kw):
    out = []
    for fam in families:
        for K in Ks:
            for i in range(n):
                seed = base_seed * 1_000_003 + hash((fam, K, i)) % 1_000_000
                seed = (base_seed + 1) * 7919 + {"T1": 0, "T2": 1, "T3": 2}[fam] * 100_000 + K * 1000 + i
                extra = {"tmpdir": kw.get("tmpdir")} if fam == "T2" else {}
                out.append(FAMILIES[fam](i, K, seed, **extra))
    return out


def apply_variants(specs):
    """specs like ['t1.records=25-35', 't3.words=range', 't3.lipogram=the']"""
    for spec in specs or []:
        key, val = spec.split("=", 1)
        mod, field = key.split(".", 1)
        cfg = {"t1": t1_aggregation.CFG, "t3": t3_writing.CFG}[mod]
        if "-" in val and val.replace("-", "").isdigit():
            lo, hi = val.split("-"); cfg[field] = (int(lo), int(hi))
        elif val.isdigit():
            cfg[field] = int(val)
        else:
            cfg[field] = val


def cfg_snapshot():
    """Serializable copy of the generator settings in force (stored in every output row)."""
    return {"t1": {k: list(v) if isinstance(v, tuple) else v for k, v in t1_aggregation.CFG.items()},
            "t3": {k: list(v) if isinstance(v, tuple) else v for k, v in t3_writing.CFG.items()}}


def apply_cfg(snap):
    for mod, cfg in (("t1", t1_aggregation.CFG), ("t3", t3_writing.CFG)):
        for k, v in (snap or {}).get(mod, {}).items():
            cfg[k] = tuple(v) if isinstance(v, list) else v


LEGACY_VARIANTS = ["t1.records=60-90", "t1.zero_share=high", "t1.output=full",
                   "t3.words=exact", "t3.lipogram=e", "t3.N=4-6", "t3.width=2"]


def restore_from_rows(rows, override=None):
    """Put the generators in the state that produced `rows`. Priority: CLI override > cfg snapshot
    > variant string (+ legacy defaults for files written before snapshots existed)."""
    if override:
        apply_variants(LEGACY_VARIANTS); apply_variants(override); return "override"
    r0 = rows[0]
    if r0.get("cfg"):
        apply_cfg(r0["cfg"]); return "snapshot"
    apply_variants(LEGACY_VARIANTS)
    apply_variants([x for x in r0.get("variant", "").split(",") if x])
    return "legacy+variant"
