"""Generate Veilguard Summary PDF."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable,
)
from reportlab.lib.enums import TA_CENTER

doc = SimpleDocTemplate(
    "veilguard-summary.pdf", pagesize=A4,
    topMargin=2*cm, bottomMargin=2*cm, leftMargin=2.5*cm, rightMargin=2.5*cm,
)

styles = getSampleStyleSheet()
accent = HexColor("#0f3460")
accent_light = HexColor("#e8edf3")

styles.add(ParagraphStyle("CoverTitle", parent=styles["Title"], fontSize=36, spaceAfter=10,
    textColor=HexColor("#1a1a2e"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("CoverSub", parent=styles["Normal"], fontSize=16, spaceAfter=30,
    textColor=HexColor("#16213e"), fontName="Helvetica"))
styles.add(ParagraphStyle("SH", parent=styles["Heading1"], fontSize=20, spaceBefore=20, spaceAfter=10,
    textColor=accent, fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("SH2", parent=styles["Heading2"], fontSize=14, spaceBefore=12, spaceAfter=6,
    textColor=HexColor("#1a1a2e"), fontName="Helvetica-Bold"))
styles.add(ParagraphStyle("B", parent=styles["Normal"], fontSize=10, spaceAfter=6, leading=14, fontName="Helvetica"))
styles.add(ParagraphStyle("BL", parent=styles["Normal"], fontSize=10, spaceAfter=4, leading=14,
    fontName="Helvetica", leftIndent=20, bulletIndent=10))
styles.add(ParagraphStyle("CO", parent=styles["Normal"], fontSize=9, spaceAfter=6, leading=12,
    fontName="Courier", backColor=HexColor("#f0f0f0"), leftIndent=10, rightIndent=10))
styles.add(ParagraphStyle("CB", parent=styles["Normal"], fontSize=10, spaceAfter=6, leading=14,
    fontName="Helvetica", alignment=TA_CENTER))
styles.add(ParagraphStyle("FT", parent=styles["Normal"], fontSize=8, textColor=HexColor("#888888"), fontName="Helvetica"))

def hr():
    return HRFlowable(width="100%", thickness=1, color=accent_light, spaceAfter=10)

def tbl(data, widths):
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), accent),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, accent_light]),
    ]))
    return t

story = []

# ── Cover ──
story.append(Spacer(1, 80))
story.append(Paragraph("VEILGUARD", styles["CoverTitle"]))
story.append(HRFlowable(width="60%", thickness=3, color=accent, spaceAfter=15))
story.append(Paragraph("Phishield Internal AI System", styles["CoverSub"]))
story.append(Paragraph("Phase 1 \u2014 Technical Summary", styles["CoverSub"]))
story.append(Spacer(1, 40))

cover = Table([
    ["Version", "1.0"],
    ["Date", "March 2026"],
    ["Classification", "Internal"],
    ["Repository", "github.com/RJL667/veilguard"],
    ["Status", "Phase 1 Complete"],
], colWidths=[120, 300])
cover.setStyle(TableStyle([
    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 10),
    ("TEXTCOLOR", (0, 0), (0, -1), accent),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ("TOPPADDING", (0, 0), (-1, -1), 8),
    ("LINEBELOW", (0, 0), (-1, -2), 0.5, HexColor("#dddddd")),
]))
story.append(cover)
story.append(Spacer(1, 60))
story.append(Paragraph("8 MCP Servers  \u2022  76 Tools  \u2022  PII-Safe  \u2022  Self-Extending", styles["CB"]))
story.append(PageBreak())

# ── Executive Summary ──
story.append(Paragraph("Executive Summary", styles["SH"]))
story.append(hr())
story.append(Paragraph(
    "Veilguard is Phishield's internal AI workspace built on LibreChat with Gemini as the LLM backend. "
    "All user input passes through a PII redaction proxy (Presidio) before reaching Gemini, ensuring "
    "sensitive data never leaves the organization.", styles["B"]))
story.append(Paragraph(
    "The system provides 76 tools across 8 MCP servers, giving the AI capabilities for file management, "
    "code execution, document creation, web research, host control, sub-agent orchestration, and "
    "self-extending tool creation via the Tool Forge.", styles["B"]))

# ── Architecture ──
story.append(Paragraph("Architecture", styles["SH"]))
story.append(hr())
story.append(tbl([
    ["Component", "Technology", "Role"],
    ["Frontend", "LibreChat v0.8.4", "Chat UI with MCP tool integration"],
    ["LLM", "Gemini 2.5 Flash", "Reasoning and tool orchestration"],
    ["PII Gateway", "FastAPI + Presidio", "Redact PII before LLM, rehydrate after"],
    ["Database", "MongoDB 7.0", "Conversation and user storage"],
    ["Search", "MeiliSearch v1.12", "Full-text search of conversations"],
    ["Container", "Docker Compose", "Orchestration of all services"],
], [100, 130, 230]))
story.append(Spacer(1, 10))
story.append(Paragraph("Request Flow", styles["SH2"]))
story.append(Paragraph(
    "User Input \u2192 PII Redaction \u2192 Gemini Reasoning \u2192 Tool Execution \u2192 PII Rehydration \u2192 Response",
    styles["CO"]))

# ── MCP Servers ──
story.append(Paragraph("MCP Tool Servers", styles["SH"]))
story.append(hr())

srv_data = [
    ["#", "Server", "Transport", "Tools", "Purpose"],
    ["1", "filesystem", "stdio", "14", "File CRUD, precision editing, diffs, CSV/JSON"],
    ["2", "code-exec", "stdio", "2", "Python and bash execution in container"],
    ["3", "web", "stdio", "3", "Google search, URL browsing, page fetch"],
    ["4", "documents", "stdio", "16", "PDF/DOCX/XLSX/PPTX read, create, edit"],
    ["5", "image", "stdio", "4", "Image manipulation, matplotlib charts"],
    ["6", "host-exec", "SSE :8808", "6", "CMD, PowerShell, Docker, Git on host"],
    ["7", "sub-agents", "SSE :8809", "24", "Agent orchestration, background tasks"],
    ["8", "forge", "SSE :8810", "7", "Create custom tools at runtime"],
]
st = Table(srv_data, colWidths=[20, 75, 70, 35, 260])
st.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), accent),
    ("TEXTCOLOR", (0, 0), (-1, 0), white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#cccccc")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, accent_light]),
]))
story.append(st)

story.append(PageBreak())

# ── Key Capabilities ──
story.append(Paragraph("Key Capabilities", styles["SH"]))
story.append(hr())

for section, bullets in [
    ("PII Protection", [
        "All user input scanned and redacted before reaching Gemini",
        "Supports: names, email addresses, phone numbers, SA ID numbers",
        "Deterministic tokens (REF_PERSON_1, REF_EMAIL_1) for consistent mapping",
        "Response rehydration restores original values for the user",
        "Custom South African recognizers (SA ID format, +27 phone numbers)",
    ]),
    ("Sub-Agent Orchestration", [
        "<b>spawn_agent</b> \u2014 delegate to specialized roles (analyst, threat, writer, coder)",
        "<b>parallel_agents</b> \u2014 run multiple agents concurrently for speed",
        "<b>review_loop</b> \u2014 worker + critic quality assurance cycle",
        "<b>background / smart_task</b> \u2014 async execution with auto-wait and polling",
        "<b>scheduled tasks</b> \u2014 cron-style recurring background jobs",
    ]),
    ("Document Pipeline", [
        "Create and read PDF, Word, Excel, PowerPoint files",
        "Precision editing: exact string replace, line ranges, unified diffs",
        "PDF: text replace, merge, watermark, page delete, text overlay",
        "Excel: cell-level and range editing with auto-width columns",
        "Word: find and replace across paragraphs and tables",
    ]),
    ("Tool Forge (Self-Extending)", [
        "AI creates new tools at runtime when it encounters missing capabilities",
        "Tools are Python functions saved to disk, persist across sessions",
        "Built-in testing, editing, and version tracking",
        "Example: AI auto-forged a check_spf tool for DNS SPF record lookups",
    ]),
    ("Host Control & Operations", [
        "Run CMD, PowerShell, Docker, and Git commands from chat",
        "Read/write files on the Windows host filesystem",
        "Windows notifications for task completion alerts",
        "Clipboard system for passing data between tool calls",
    ]),
]:
    story.append(Paragraph(section, styles["SH2"]))
    for b in bullets:
        story.append(Paragraph("\u2022 " + b, styles["BL"]))

story.append(PageBreak())

# ── Deployment ──
story.append(Paragraph("Deployment", styles["SH"]))
story.append(hr())
story.append(Paragraph("Docker Services", styles["SH2"]))
story.append(Paragraph("docker compose up -d", styles["CO"]))
story.append(Paragraph(
    "Starts: LibreChat (port 3080), MongoDB, MeiliSearch, PII Proxy (port 4000)", styles["B"]))
story.append(Paragraph("Host-Side MCP Servers", styles["SH2"]))
story.append(Paragraph("python mcp-tools/host-exec/server.py --sse --port 8808", styles["CO"]))
story.append(Paragraph("python mcp-tools/sub-agents/server.py --sse --port 8809", styles["CO"]))
story.append(Paragraph("python mcp-tools/forge/server.py --sse --port 8810", styles["CO"]))
story.append(Spacer(1, 10))
story.append(Paragraph("Access the UI at <b>http://localhost:3080</b>", styles["B"]))

# ── Repo Structure ──
story.append(Paragraph("Repository Structure", styles["SH"]))
story.append(hr())
for line in [
    "veilguard/",
    "\u251c\u2500\u2500 docker-compose.yml          # All Docker services",
    "\u251c\u2500\u2500 Dockerfile.librechat        # LibreChat + Python + tool deps",
    "\u251c\u2500\u2500 librechat.yaml              # Config, MCP servers, system prompt",
    "\u251c\u2500\u2500 .env / .env.example         # API keys",
    "\u251c\u2500\u2500 pii-proxy/                  # PII redaction gateway (Presidio)",
    "\u251c\u2500\u2500 mcp-tools/",
    "\u2502   \u251c\u2500\u2500 filesystem/server.py    # 14 tools",
    "\u2502   \u251c\u2500\u2500 code-exec/server.py     # 2 tools",
    "\u2502   \u251c\u2500\u2500 web/server.py           # 3 tools",
    "\u2502   \u251c\u2500\u2500 documents/server.py     # 16 tools",
    "\u2502   \u251c\u2500\u2500 image/server.py         # 4 tools",
    "\u2502   \u251c\u2500\u2500 host-exec/server.py     # 6 tools  (host SSE)",
    "\u2502   \u251c\u2500\u2500 sub-agents/server.py    # 24 tools (host SSE)",
    "\u2502   \u2514\u2500\u2500 forge/server.py         # 7 tools  (host SSE)",
    "\u251c\u2500\u2500 agents/                     # Role definitions",
    "\u2514\u2500\u2500 workspace/                  # User-generated files",
]:
    story.append(Paragraph(line, styles["CO"]))

# ── Phase 2 ──
story.append(Spacer(1, 20))
story.append(Paragraph("Phase 2 Roadmap \u2014 TCMM Intelligence Layer", styles["SH"]))
story.append(hr())
story.append(tbl([
    ["Capability", "Description", "Status"],
    ["Persistent Memory", "Vector store for cross-conversation knowledge", "Planned"],
    ["Dream Consolidation", "Nightly summarization of work patterns", "Planned"],
    ["Skill Extraction", "Learn reusable skills from staff behavior", "Planned"],
    ["Auto-Agent Generation", "Create specialized agents from workflow patterns", "Planned"],
    ["Authentication", "Role-based access control", "Planned"],
], [130, 230, 60]))

# ── Footer ──
story.append(Spacer(1, 40))
story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#cccccc"), spaceAfter=10))
story.append(Paragraph(
    "Veilguard v1.0 \u2014 Phishield Internal AI System \u2014 github.com/RJL667/veilguard", styles["FT"]))
story.append(Paragraph("Generated March 2026 \u2014 Confidential", styles["FT"]))

doc.build(story)
print("PDF created: veilguard-summary.pdf")
