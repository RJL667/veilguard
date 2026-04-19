"""Notification tools — Windows toast notifications."""

import subprocess
import time
from core import state


def register(mcp):
    @mcp.tool()
    async def notify(message: str, title: str = "Veilguard") -> str:
        """Send a Windows toast notification to the user."""
        state.notifications.append({"title": title, "message": message, "time": time.time()})
        safe_title = title.replace("'", "''")
        safe_msg = message.replace("'", "''")
        ps_script = f"""
[void] [System.Reflection.Assembly]::LoadWithPartialName('System.Windows.Forms')
$n = New-Object System.Windows.Forms.NotifyIcon
$n.Icon = [System.Drawing.SystemIcons]::Information
$n.BalloonTipTitle = '{safe_title}'
$n.BalloonTipText = '{safe_msg}'
$n.Visible = $true
$n.ShowBalloonTip(5000)
Start-Sleep -Seconds 6
$n.Dispose()
"""
        try:
            subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"Notification sent: {title} — {message}"
        except Exception as e:
            return f"Notification stored (failed: {e}): {title} — {message}"

    @mcp.tool()
    async def get_notifications() -> str:
        """Get all notifications sent this session."""
        if not state.notifications:
            return "No notifications."
        lines = ["# Notifications\n"]
        for n in state.notifications:
            t = time.strftime("%H:%M:%S", time.localtime(n["time"]))
            lines.append(f"- [{t}] **{n['title']}**: {n['message']}")
        return "\n".join(lines)
