
import subprocess
import sys

def run(path: str) -> str:
    try:
        # Install pdfplumber if not present
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pdfplumber"], 
                      capture_output=True, timeout=30)
        
        import pdfplumber
        
        with pdfplumber.open(path) as pdf:
            text = ""
            for i, page in enumerate(pdf.pages):
                text += f"\n--- PAGE {i+1} ---\n"
                text += page.extract_text() or "[No text found on this page]"
            return text if text.strip() else "No extractable text found in PDF"
    except Exception as e:
        return f"Error: {str(e)}"
