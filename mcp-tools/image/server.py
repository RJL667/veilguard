"""Image MCP Tool Server for Veilguard.

Tools: describe_image, resize_image, convert_image, create_chart
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless backend — must be before pyplot import

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "image",
    instructions="Image tools for inspecting, resizing, converting images and creating charts.",
)

import re as _re

WORKSPACE = os.environ.get("WORKSPACE_ROOT", "/workspace")

# Windows absolute path pattern, e.g. ``C:\Users\foo``, ``D:/data``.
# We run on Linux inside a docker container so a Path() over a string
# like this is interpreted as a relative POSIX filename — Path.is_absolute()
# returns False, the workspace prefix gets prepended, and you end up
# trying to write to ``/workspace/C:\Users\foo`` which fails with EACCES.
_WINDOWS_ABS_PATH_RE = _re.compile(r'^[A-Za-z]:[\\/]')


def _safe_path(path: str) -> Path:
    """Resolve a tool path relative to the workspace and prevent escape.

    The workspace lives at $WORKSPACE_ROOT (default /workspace) inside
    the docker container. LibreChat agents sometimes echo the user's
    Windows desktop paths verbatim ("C:\\Users\\...\\foo.png") into
    tool calls — that path doesn't exist in the container, and silently
    re-rooting it under /workspace produces a nonsense filename that
    fails with a confusing "Permission denied". Reject it with a clear
    error so the LLM (and the user) can correct course.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("Path is required")
    if _WINDOWS_ABS_PATH_RE.match(path):
        suggestion = _WINDOWS_ABS_PATH_RE.sub("", path).replace("\\", "/")
        raise ValueError(
            f"Path {path!r} looks like a Windows absolute path, but this "
            f"server runs in a Linux container with workspace at "
            f"{WORKSPACE}. The container does not mirror the user's "
            f"Windows filesystem. Pass a workspace-relative path instead, "
            f"e.g. {suggestion!r} (lands at {WORKSPACE}/{suggestion}). "
            f"For files that must end up on the user's Windows host, use "
            f"the host-exec server's host_file_write tool instead."
        )
    if "\\" in path:
        raise ValueError(
            f"Path {path!r} contains backslashes (Windows separators). "
            f"Use forward slashes — try {path.replace(chr(92), '/')!r}."
        )
    p = Path(path)
    if not p.is_absolute():
        p = Path(WORKSPACE) / p
    resolved = p.resolve()
    workspace_resolved = Path(WORKSPACE).resolve()
    # is_relative_to (3.9+) avoids the str.startswith adjacent-name bug
    # that would have allowed e.g. /workspace2 to pass the old check.
    try:
        resolved.relative_to(workspace_resolved)
    except ValueError:
        raise ValueError(f"Access denied: path must be within {WORKSPACE}")
    return resolved


@mcp.tool()
def describe_image(path: str) -> str:
    """Get image metadata: format, dimensions, mode, file size, and EXIF data.

    Args:
        path: Image file path (absolute or relative to workspace)
    """
    from PIL import Image
    from PIL.ExifTags import TAGS

    p = _safe_path(path)
    if not p.exists():
        return f"Error: File not found: {p}"

    try:
        img = Image.open(str(p))
        size_bytes = p.stat().st_size
        if size_bytes < 1024:
            size_str = f"{size_bytes}B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f}KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.1f}MB"

        info = [
            f"# {p.name}",
            f"Format: {img.format}",
            f"Size: {img.width} x {img.height} px",
            f"Mode: {img.mode}",
            f"File size: {size_str}",
        ]

        # EXIF data (JPEG)
        exif = img.getexif()
        if exif:
            info.append("\nEXIF:")
            for tag_id, value in list(exif.items())[:20]:
                tag = TAGS.get(tag_id, tag_id)
                info.append(f"  {tag}: {value}")

        img.close()
        return "\n".join(info)
    except Exception as e:
        return f"Error describing image: {e}"


@mcp.tool()
def resize_image(path: str, width: int, height: int, output_path: str = "") -> str:
    """Resize an image to specified dimensions.

    Args:
        path: Source image path
        width: Target width in pixels
        height: Target height in pixels
        output_path: Output path. Empty = overwrite original.
    """
    from PIL import Image

    p = _safe_path(path)
    if not p.exists():
        return f"Error: File not found: {p}"

    out = _safe_path(output_path) if output_path else p

    try:
        img = Image.open(str(p))
        original_size = f"{img.width}x{img.height}"
        resized = img.resize((width, height), Image.LANCZOS)
        out.parent.mkdir(parents=True, exist_ok=True)
        resized.save(str(out))
        resized.close()
        img.close()
        return f"Resized {original_size} → {width}x{height}: {out}"
    except Exception as e:
        return f"Error resizing image: {e}"


@mcp.tool()
def convert_image(path: str, format: str, output_path: str = "") -> str:
    """Convert an image to a different format.

    Args:
        path: Source image path
        format: Target format: png, jpg, jpeg, webp, bmp, tiff
        output_path: Output path. Empty = same name with new extension.
    """
    from PIL import Image

    p = _safe_path(path)
    if not p.exists():
        return f"Error: File not found: {p}"

    fmt = format.lower().strip(".")
    fmt_map = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP", "bmp": "BMP", "tiff": "TIFF"}
    pil_format = fmt_map.get(fmt)
    if not pil_format:
        return f"Error: Unsupported format '{format}'. Use: {list(fmt_map.keys())}"

    ext = "jpg" if fmt in ("jpg", "jpeg") else fmt
    out = _safe_path(output_path) if output_path else p.with_suffix(f".{ext}")

    try:
        img = Image.open(str(p))
        # Convert RGBA to RGB for JPEG
        if pil_format == "JPEG" and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out), format=pil_format)
        img.close()
        return f"Converted {p.name} → {out}"
    except Exception as e:
        return f"Error converting image: {e}"


@mcp.tool()
def create_chart(chart_type: str, data: str, title: str, output_path: str) -> str:
    """Create a chart/graph and save as PNG.

    Args:
        chart_type: One of: bar, line, pie, scatter
        data: JSON string with chart data. Format depends on type:
              bar/line: {"labels": ["A","B","C"], "values": [10,20,30]} or {"labels": [...], "series": {"name1": [...], "name2": [...]}}
              pie: {"labels": ["A","B","C"], "values": [10,20,30]}
              scatter: {"x": [1,2,3], "y": [4,5,6]}
        title: Chart title
        output_path: Output PNG path (absolute or relative to workspace)
    """
    import matplotlib.pyplot as plt

    out = _safe_path(output_path)

    try:
        parsed = json.loads(data)
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar":
            labels = parsed["labels"]
            if "series" in parsed:
                import numpy as np
                x = np.arange(len(labels))
                n = len(parsed["series"])
                width = 0.8 / n
                for i, (name, values) in enumerate(parsed["series"].items()):
                    ax.bar(x + i * width, values, width, label=name)
                ax.set_xticks(x + width * (n - 1) / 2)
                ax.set_xticklabels(labels)
                ax.legend()
            else:
                ax.bar(labels, parsed["values"])

        elif chart_type == "line":
            labels = parsed["labels"]
            if "series" in parsed:
                for name, values in parsed["series"].items():
                    ax.plot(labels, values, marker="o", label=name)
                ax.legend()
            else:
                ax.plot(labels, parsed["values"], marker="o")

        elif chart_type == "pie":
            ax.pie(parsed["values"], labels=parsed["labels"], autopct="%1.1f%%", startangle=90)
            ax.axis("equal")

        elif chart_type == "scatter":
            ax.scatter(parsed["x"], parsed["y"])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

        else:
            plt.close(fig)
            return f"Error: Unknown chart type '{chart_type}'. Use: bar, line, pie, scatter"

        ax.set_title(title)
        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return f"Created {chart_type} chart: {out}"
    except json.JSONDecodeError:
        return "Error: 'data' must be valid JSON"
    except KeyError as e:
        return f"Error: Missing required field in data: {e}"
    except Exception as e:
        return f"Error creating chart: {e}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
