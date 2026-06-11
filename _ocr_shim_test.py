"""Live test of the Gemini OCR shim (app/ocr_gemini.py) — generates a PNG
with known text, runs it through gemini_ocr, checks the Mistral-shape
response. Uses GOOGLE_API_KEY from ./\.env (real AI Studio call, ~1 image)."""
import asyncio, base64, io, os, re, sys

for line in open(".env", encoding="utf-8", errors="ignore"):
    m = re.match(r"^(GOOGLE_API_KEY|GEMINI_API_URL)=(.+)$", line.strip())
    if m:
        os.environ[m.group(1)] = m.group(2).strip()

sys.path.insert(0, "agent-proxy")
from app.ocr_gemini import gemini_ocr, parse_document, OCRUnsupportedType

from PIL import Image, ImageDraw
img = Image.new("RGB", (640, 220), "white")
d = ImageDraw.Draw(img)
d.text((20, 20), "VEILGUARD OCR TEST", fill="black")
d.text((20, 60), "Invoice #12345", fill="black")
d.text((20, 100), "Client: Johan van der Merwe", fill="black")
d.text((20, 140), "Total: R 1,234.56", fill="black")
buf = io.BytesIO()
img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

doc = {"type": "image_url", "image_url": f"data:image/png;base64,{b64}"}

# 1) unsupported mimetype raises the fallback signal
try:
    parse_document({"type": "document_url",
                    "document_url": "data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,QUJD"})
    raise AssertionError("docx should be unsupported")
except OCRUnsupportedType:
    print("1) docx -> OCRUnsupportedType (falls back to document_parser) OK")

# 2) live Gemini OCR on the generated image
res = asyncio.run(gemini_ocr(doc, "mistral-ocr-latest"))  # bad model name -> default gemini
assert res["pages"] and res["pages"][0]["markdown"], res
md = res["pages"][0]["markdown"]
print("2) model used:", res["model"])
print("   markdown:", repr(md[:200]))
for needle in ("12345", "Merwe", "1,234.56"):
    assert needle in md, f"missing {needle!r} in OCR output"
print("3) all expected strings transcribed OK")
print("ALL PASS")
