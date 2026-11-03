from .germline import BARRIER_NOTE, BARRIER_NOTE_STRONG, AMEND_NOTE

BARRIER = {"mode": "normal"}   # set to "strong" via --barrier strong

ARMS = ["B0", "B1", "B2", "B3", "L"]

AGENT_SYS = (
    "You are one agent in a multi-agent system working on a task. Follow the task "
    "exactly as stated. Be precise and complete."
)


def build_messages(arm, task, block, parent_brief):
    """Context the agent at this hop sees. `block` is the (possibly amended) germline block."""
    sys = AGENT_SYS
    if task.attachment:
        sys += "\n\nDATA attachment (available to every agent):\n" + task.attachment
    msgs = []
    if arm == "B0":
        msgs.append({"role": "system", "content": sys})
    elif arm == "B1":
        msgs.append({"role": "system", "content": sys + "\n\nOriginal task statement (for reference):\n" + task.tau})
    elif arm == "B2":
        note = BARRIER_NOTE_STRONG if BARRIER["mode"] == "strong" else BARRIER_NOTE
        msgs.append({"role": "system", "content": block + "\n" + note + "\n\n" + sys})
    elif arm == "B3":
        note = BARRIER_NOTE_STRONG if BARRIER["mode"] == "strong" else BARRIER_NOTE
        msgs.append({"role": "germline", "content": block + "\n" + note})
        msgs.append({"role": "system", "content": sys})
    elif arm == "L":
        msgs.append({"role": "system", "content": block + "\n" + AMEND_NOTE + "\n\n" + sys})
    else:
        raise ValueError(arm)
    msgs.append({"role": "user", "content": parent_brief})
    return msgs
