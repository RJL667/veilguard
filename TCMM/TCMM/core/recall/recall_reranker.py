"""
TCMM Recall Reranker — FlashRank wrapper.
"""

import os

try:
    from flashrank import Ranker, RerankRequest
except Exception:
    pass


class RerankerStub:
    """FlashRank-backed Reranker.

    Uses ms-marco-TinyBERT-L-2-v2 (~4MB INT8-quantized ONNX). Standalone
    throughput measured 23 Apr 2026 on a c3-standard-4 (4 vCPU): 45 pairs
    in 54ms (1.15ms/pair). Under production load the same workload took
    ~428ms — an 8x slowdown caused by 20 concurrent background worker
    threads fighting the reranker's ONNX session for cores. FlashRank's
    default InferenceSession has ``intra_op_num_threads=0`` which means
    "use all cores", which on a multi-threaded service is the wrong
    default: ONNX grabs all 4, gets constantly preempted by workers
    doing Vertex HTTP + NLP work, and wall time stretches 8x.

    Fix: rebuild the session with a pinned thread count so reranker
    latency is predictable. Controlled by env ``VEILGUARD_RERANKER_THREADS``
    (default 2 — leaves half the cores free for workers). Measured
    effect should bring rerank back to ~60-80ms in production.
    """

    def __init__(self):
        # Nano is fast, ~4MB
        self.ranker = Ranker(
            model_name="ms-marco-TinyBERT-L-2-v2",
            cache_dir="./.flashrank_cache",
        )
        self._tune_onnx_threads()

    def _tune_onnx_threads(self):
        """Rebuild the ONNX session with explicit thread limits.

        FlashRank exposes ``self.ranker.session`` as the
        ``onnxruntime.InferenceSession``. We locate the original .onnx
        file, then create a fresh session with tuned SessionOptions
        and hot-swap it in. If anything fails, leave the original
        session in place (fail-open — slow but correct).
        """
        try:
            import onnxruntime as _ort
            num_threads = int(os.environ.get("VEILGUARD_RERANKER_THREADS", "2"))

            onnx_path = None
            model_dir = getattr(self.ranker, "model_dir", None)
            if model_dir:
                candidates = [
                    os.path.join(model_dir, "flashrank-TinyBERT-L-2-v2.onnx"),
                    os.path.join(model_dir, "model.onnx"),
                    os.path.join(model_dir, "model_quantized.onnx"),
                ]
                for c in candidates:
                    if os.path.exists(c):
                        onnx_path = c
                        break
            if onnx_path is None:
                return  # can't find the model file — keep the default session

            so = _ort.SessionOptions()
            so.intra_op_num_threads = num_threads
            so.inter_op_num_threads = 1
            so.execution_mode = _ort.ExecutionMode.ORT_SEQUENTIAL
            so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self.ranker.session = _ort.InferenceSession(
                onnx_path,
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )

            import logging as _lg
            _lg.getLogger("tcmm.reranker").info(
                f"[RERANKER] ONNX session rebuilt with "
                f"intra_op_num_threads={num_threads} inter_op_num_threads=1 "
                f"model={onnx_path}"
            )
        except Exception as e:
            import logging as _lg
            _lg.getLogger("tcmm.reranker").warning(
                f"[RERANKER] thread-pin failed, keeping default session: {e}"
            )

    def predict(self, pairs):
        # Adapt pair-list [(query, doc_text)] to FlashRank format
        if not pairs:
            return []

        # Assuming all pairs have same query
        query = pairs[0][0]

        passages = []
        for i, (_, text) in enumerate(pairs):
             passages.append({"id": f"{i}", "text": text})

        request = RerankRequest(query=query, passages=passages)
        results = self.ranker.rerank(request)

        score_map = {}
        for res in results:
             score_map[int(res["id"])] = res["score"]

        output_scores = []
        for i in range(len(pairs)):
             output_scores.append(score_map.get(i, 0.0))

        return output_scores
