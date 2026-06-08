import sys, traceback
sys.path.insert(0, r"C:\Users\rudol\Documents\veilguard")
try:
    from pii import get_redactor
    r = get_redactor()
    sid = "pii-repro"
    text = ("The class TaskQueue coordinates a Worker and a Scheduler. "
            "Call enqueue(task) then Worker.run(). Sarah Chen owns the module.")
    red = r.redact_text(text, sid)
    print("=== ORIGINAL ===\n" + text)
    print("=== REDACTED  ===\n" + red)
    print("has REF_ tokens:", "REF_" in red)
    # simulate the model echoing redacted tokens inside a write_file tool_use arg
    blk = [{"type": "tool_use", "id": "t1", "name": "write_file",
            "input": {"path": "queue.py", "content": "# " + red}}]
    reh = r.rehydrate_blocks(blk, sid)
    out = reh[0]["input"]["content"]
    print("=== REHYDRATED tool_use content ===\n" + out)
    print("tool-arg fully restored:", out == "# " + text)
    print("tool-arg STILL has REF_:", "REF_" in out)
except Exception as e:
    print("ERROR:", repr(e))
    traceback.print_exc()
