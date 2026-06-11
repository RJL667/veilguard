"""Offline test of documents-server OCR wiring: fake /v1/ocr shim on
localhost, generated scan-style + text PDFs + PNG, direct calls into the
server module (mcp framework stubbed if absent)."""
import io, json, os, sys, tempfile, threading, types
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = 4977
SECRET = "testsecret"
tmp = tempfile.mkdtemp()
os.environ["OCR_BASEURL"] = f"http://localhost:{PORT}/v1"
os.environ["OCR_API_KEY"] = SECRET
os.environ["WORKSPACE_ROOT"] = tmp

calls = []

class FakeShim(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        auth_ok = self.headers.get("Authorization") == f"Bearer {SECRET}"
        calls.append({"path": self.path, "auth_ok": auth_ok,
                      "doc_type": body["document"]["type"]})
        out = {"pages": [{"index": 0, "markdown": "FAKE OCR MARKDOWN 42",
                          "images": [], "dimensions": {}}],
               "model": body.get("model"), "usage_info": {}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *a): pass

srv = HTTPServer(("127.0.0.1", PORT), FakeShim)
threading.Thread(target=srv.serve_forever, daemon=True).start()

# stub mcp framework if not installed locally
try:
    import mcp.server.fastmcp  # noqa
except ImportError:
    m = types.ModuleType("mcp"); s = types.ModuleType("mcp.server")
    f = types.ModuleType("mcp.server.fastmcp")
    class FastMCP:
        def __init__(self, *a, **k): pass
        def tool(self, *a, **k):
            return lambda fn: fn
        def run(self, *a, **k): pass
    f.FastMCP = FastMCP
    sys.modules.update({"mcp": m, "mcp.server": s, "mcp.server.fastmcp": f})

import fitz
# (a) scan-style PDF: page is one big image, zero text layer
text_page = fitz.open()
pg = text_page.new_page(width=400, height=200)
pg.insert_text((40, 90), "SCANNED CONTENT 999", fontsize=18)
pix = pg.get_pixmap(dpi=120)
scan = fitz.open()
sp = scan.new_page(width=400, height=200)
sp.insert_image(sp.rect, pixmap=pix)
scan.save(os.path.join(tmp, "scan.pdf")); scan.close(); text_page.close()
# (b) normal text PDF
norm = fitz.open()
np_ = norm.new_page()
np_.insert_text((72, 100), "This is a perfectly normal text layer with plenty of characters to clear the threshold.", fontsize=11)
norm.save(os.path.join(tmp, "normal.pdf")); norm.close()
# (c) PNG
pix.save(os.path.join(tmp, "shot.png"))

sys.path.insert(0, r"C:\Users\rudol\Documents\veilguard\services\documents")
import server

r1 = server.read_pdf("scan.pdf")
assert "FAKE OCR MARKDOWN 42" in r1 and "OCR'd via Gemini" in r1, r1[:300]
print("1) read_pdf scan -> OCR fallback OK")

r2 = server.read_pdf("normal.pdf")
assert "normal text layer" in r2 and "FAKE OCR" not in r2, r2[:300]
print("2) read_pdf text PDF -> normal path (no OCR call) OK")

r3 = server.ocr_document("shot.png")
assert "FAKE OCR MARKDOWN 42" in r3, r3[:300]
print("3) ocr_document PNG OK")

r4 = server.ocr_document("normal.pdf")
assert "FAKE OCR MARKDOWN 42" in r4, r4[:300]
print("4) ocr_document forced on text PDF OK")

doc_types = [c["doc_type"] for c in calls]
assert all(c["auth_ok"] for c in calls)
assert doc_types.count("document_url") >= 2 and "image_url" in doc_types, doc_types
print(f"5) shim saw {len(calls)} calls, auth OK, doc types {doc_types}")
print("ALL PASS")
