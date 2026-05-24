
import subprocess
def run(domain: str) -> str:
    try:
        result = subprocess.run(["nslookup", "-type=TXT", domain], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            if "v=spf1" in result.stdout:
                return f"SPF record found for {domain}:\n{result.stdout}"
            else:
                return f"No SPF record (v=spf1) found in TXT records for {domain}:\n{result.stdout}"
        else:
            return f"Error checking SPF for {domain}: {result.stderr}"
    except Exception as e:
        return f"An error occurred: {e}"
