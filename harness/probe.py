def retention_vector(task, text):
    """Which original constraints are still present in a restatement/brief."""
    return {c.cid: c.present_in(text) for c in task.constraints}


def summarize_retention(task, vec):
    n = len(task.constraints)
    kept = sum(1 for v in vec.values() if v)
    inc = [c for c in task.constraints if c.inconvenient]
    easy = [c for c in task.constraints if not c.inconvenient]
    return {
        "kept": kept, "n": n, "frac": kept / n if n else 1.0,
        "inc_kept": sum(1 for c in inc if vec[c.cid]), "inc_n": len(inc),
        "easy_kept": sum(1 for c in easy if vec[c.cid]), "easy_n": len(easy),
    }
