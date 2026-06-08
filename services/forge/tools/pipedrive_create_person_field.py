
import urllib.request
import urllib.error
import json

def run(api_token: str, name: str, field_type: str, options: str = "") -> str:
    """
    Create a custom Person field in Pipedrive (v1 API, JSON body).
    field_type: 'enum', 'varchar', 'date', etc.
    options: comma-separated labels for enum fields.
    """
    url = f"https://api.pipedrive.com/v1/personFields?api_token={api_token}"
    payload = {"name": name, "field_type": field_type}
    if options and field_type == "enum":
        payload["options"] = [{"label": o.strip()} for o in options.split(",") if o.strip()]
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = resp.read().decode("utf-8")
            parsed = json.loads(body)
            d = parsed.get("data") or {}
            summary = {
                "success": parsed.get("success"),
                "id": d.get("id"),
                "key": d.get("key"),
                "name": d.get("name"),
                "field_type": d.get("field_type"),
                "options": [{"id": o.get("id"), "label": o.get("label")} for o in (d.get("options") or [])],
            }
            return json.dumps(summary, indent=2)
    except urllib.error.HTTPError as e:
        return f"HTTPError {e.code}: {e.read().decode('utf-8')}"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"
