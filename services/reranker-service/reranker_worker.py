"""
TCMM Reranker out-of-process worker.

2026-05-13: spawned as a long-lived subprocess from RerankerStub so the
FlashRank ONNX inference runs in a process with zero competing Python
threads. In-process the same model takes ~900ms for 64 pairs because of
GIL contention from 4+ background workers (semantic_polling_worker etc.);
standalone the SAME call is 17ms. Dedicating a child gets us close to
the standalone number.

Protocol: newline-delimited JSON on stdin/stdout.

Request:
  {"op": "predict", "query": "<q>", "passages": ["<doc1>", "<doc2>", ...]}
Response:
  {"ok": true, "scores": [<float>, ...]}
  {"ok": false, "error": "<msg>"}

Health:
  {"op": "ping"} -> {"ok": true, "pid": <pid>, "model": "<name>"}
"""
from __future__ import annotations
import json
import os
import sys
import time


def _log(msg: str) -> None:
    # stderr so it doesn't pollute the JSON-on-stdout protocol
    print(f"[reranker_worker pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


def main() -> int:
    _log("booting")
    try:
        from flashrank import Ranker, RerankRequest
    except Exception as e:
        _log(f"flashrank import failed: {e}")
        return 2

    model_name = os.environ.get("VEILGUARD_RERANKER_MODEL", "ms-marco-TinyBERT-L-2-v2")
    cache_dir = os.environ.get("VEILGUARD_RERANKER_CACHE_DIR", "./.flashrank_cache")

    t0 = time.perf_counter()
    ranker = Ranker(model_name=model_name, cache_dir=cache_dir)
    init_ms = (time.perf_counter() - t0) * 1000.0
    _log(f"ranker ready model={model_name} init_ms={init_ms:.0f}")

    # ONNX thread tuning (same as RerankerStub._tune_onnx_threads, but
    # without the GIL contention from the parent process — we can use all
    # cores here).
    try:
        import onnxruntime as _ort
        num_threads = int(os.environ.get("VEILGUARD_RERANKER_THREADS", "4"))
        onnx_path = None
        model_dir = getattr(ranker, "model_dir", None)
        if model_dir:
            import glob as _glob
            matches = _glob.glob(os.path.join(model_dir, "*.onnx"))
            if matches:
                onnx_path = matches[0]
        if onnx_path:
            so = _ort.SessionOptions()
            so.intra_op_num_threads = num_threads
            so.inter_op_num_threads = 1
            so.execution_mode = _ort.ExecutionMode.ORT_SEQUENTIAL
            so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            ranker.session = _ort.InferenceSession(
                onnx_path,
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )
            _log(f"onnx session rebuilt intra={num_threads} inter=1 model={onnx_path}")
    except Exception as e:
        _log(f"thread-pin failed (non-fatal): {e}")

    # Warmup pass so first real predict() is hot.
    try:
        warmup = RerankRequest(query="warmup", passages=[{"id": "0", "text": "warmup doc"}])
        ranker.rerank(warmup)
        _log("warmup done")
    except Exception as e:
        _log(f"warmup failed: {e}")

    # Signal ready to parent.
    sys.stdout.write(json.dumps({"ok": True, "ready": True, "pid": os.getpid()}) + "\n")
    sys.stdout.flush()

    # Main loop: read request lines, respond.
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as e:
            sys.stdout.write(json.dumps({"ok": False, "error": f"bad json: {e}"}) + "\n")
            sys.stdout.flush()
            continue

        op = req.get("op")
        if op == "ping":
            sys.stdout.write(json.dumps({"ok": True, "pid": os.getpid(), "model": model_name}) + "\n")
            sys.stdout.flush()
            continue

        if op == "predict":
            try:
                query = req["query"]
                passages = req["passages"]
                # passages: list of strings — wrap into FlashRank's expected shape
                pkts = [{"id": str(i), "text": txt or ""} for i, txt in enumerate(passages)]
                rr = RerankRequest(query=query, passages=pkts)
                results = ranker.rerank(rr)
                # FlashRank returns list of {id, text, score} in re-ranked order;
                # map back to input order.
                scores = [0.0] * len(passages)
                for res in results:
                    try:
                        i = int(res["id"])
                        scores[i] = float(res["score"])
                    except (KeyError, ValueError, IndexError):
                        continue
                sys.stdout.write(json.dumps({"ok": True, "scores": scores}) + "\n")
                sys.stdout.flush()
            except Exception as e:
                sys.stdout.write(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}) + "\n")
                sys.stdout.flush()
            continue

        sys.stdout.write(json.dumps({"ok": False, "error": f"unknown op: {op}"}) + "\n")
        sys.stdout.flush()

    _log("stdin closed, exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
