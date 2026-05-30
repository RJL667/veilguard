"""Windows system-tray icon + menu + task viewer for the Veilguard daemon.

Subscribes to TaskRegistry events to update the icon's tooltip + menu in
near-real-time without polling.  Click → spawns a tkinter window listing
running / completed / pending-approval tasks.

The pystray library runs the tray loop on a dedicated thread so it
doesn't tangle with the daemon's asyncio event loop.  All cross-thread
state mutation goes through ``TaskRegistry`` (which has its own lock).

Icon states (PIL-rendered at import time, 16x16 PNGs):

  idle      — solid blue circle
  busy      — blue with green dot (a task is running)
  approval  — yellow with red ring (pending approval, demands attention)

Graceful degradation: any import failure (pystray / PIL) leaves
``HAS_TRAY_UI = False`` and the daemon runs headless.  Same pattern as
toast.py.
"""

from __future__ import annotations

import logging
import platform
import threading
import time
from typing import Optional

from .tasks import TaskRegistry, TaskStatus

logger = logging.getLogger("veilguard.daemon.tray")


HAS_TRAY_UI = False
_pystray = None
_PIL = None

if platform.system() == "Windows":
    try:
        import pystray
        from PIL import Image, ImageDraw
        HAS_TRAY_UI = True
        _pystray = pystray
        _PIL = (Image, ImageDraw)
    except Exception as _e:   # pragma: no cover
        logger.warning(
            f"[tray] pystray/PIL unavailable ({_e}); tray UI disabled. "
            "Add `pystray>=0.19` and `Pillow>=10` to requirements-windows.txt "
            "and rebuild the installer to enable Phase C tray."
        )


# ── Icon rendering ─────────────────────────────────────────────────────


def _make_icon(state: str):
    """Return a 64x64 PIL Image for the given state.  16x16 is too small
    on HiDPI; pystray downscales for the tray automatically."""
    if _PIL is None:
        return None
    Image, ImageDraw = _PIL
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    if state == "approval":
        # Yellow fill with red ring — demands attention.
        d.ellipse((4, 4, size - 4, size - 4), fill=(255, 200, 0, 255))
        d.ellipse((4, 4, size - 4, size - 4),
                  outline=(220, 30, 30, 255), width=6)
    elif state == "busy":
        # Blue with green dot center.
        d.ellipse((4, 4, size - 4, size - 4),
                  fill=(48, 96, 200, 255))
        d.ellipse((22, 22, size - 22, size - 22),
                  fill=(60, 200, 90, 255))
    elif state == "offline":
        # Greyed-out — daemon can't talk to cloud right now.
        d.ellipse((4, 4, size - 4, size - 4),
                  fill=(140, 140, 140, 255))
        # Add a small "x" so it's obvious even at 16px.
        d.line((20, 20, size - 20, size - 20),
               fill=(60, 60, 60, 255), width=5)
        d.line((size - 20, 20, 20, size - 20),
               fill=(60, 60, 60, 255), width=5)
    else:
        # idle — solid Veilguard blue.
        d.ellipse((4, 4, size - 4, size - 4),
                  fill=(48, 96, 200, 255))
    return img


# ── Tray controller ────────────────────────────────────────────────────


class TrayController:
    """Owns the pystray icon + the live menu.

    Use:
      tray = TrayController(on_quit=lambda: ...)
      tray.start()    # spawns the pystray thread
      tray.refresh()  # re-render after task changes
      tray.set_connection_state(bool)  # cloud WS connect/disconnect
    """

    def __init__(
        self,
        *,
        on_quit=None,
        on_open_viewer=None,
        on_set_level=None,           # cb(level: str, scope: str)
        on_repair=None,              # cb() -> None — wipes config + opens LibreChat
        get_status_summary=None,     # cb() -> str (extra status line)
    ):
        self._on_quit = on_quit or (lambda: None)
        self._on_open_viewer = on_open_viewer or self._default_viewer
        self._on_set_level = on_set_level
        self._on_repair = on_repair or self._default_repair
        self._get_status_summary = get_status_summary
        self._icon = None
        self._thread: Optional[threading.Thread] = None
        self._cur_state: str = "idle"
        self._connected: bool = False
        # Cached current permission level — drives the radio indicator in
        # the Permission-level submenu.  Updated optimistically when the
        # user clicks an option (we don't get a server confirmation back
        # before they expect the menu to reflect the choice).  Default is
        # 'confirm' to match sub-agents' DEFAULT_LEVEL.
        self._current_level: str = "confirm"


    def _set_level_clicked(self, level: str, scope: str) -> None:
        """Tray-menu wrapper: update local cache, repaint, then fire callback."""
        self._current_level = level
        self.refresh()  # rebuild menu so radio indicator moves
        try:
            self._on_set_level(level, scope)
        except Exception as e:
            logger.warning(f"[tray] on_set_level callback raised: {e}")
        # Subscribe to task changes so the icon repaints automatically.
        TaskRegistry.get().add_listener(self.refresh)

    def set_connection_state(self, connected: bool) -> None:
        """Reflect WS connect/disconnect in the tray.

        Disconnected = icon dims to grey, title appends "(offline)".
        Reconnect repaints back to the task-derived state.  Call from
        the daemon's WS lifecycle: True after auth succeeds, False in
        the reconnect backoff loop.
        """
        prev = self._connected
        self._connected = connected
        if prev != connected:
            self.refresh()

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        if not HAS_TRAY_UI:
            logger.warning(
                "[tray] HAS_TRAY_UI=False — pystray/PIL imports failed at "
                "module load time.  Tray will not appear."
            )
            return
        try:
            self._icon = _pystray.Icon(
                "veilguard",
                icon=_make_icon(self._derive_state()),
                title=self._title(),
                menu=self._build_menu(),
            )
        except Exception as e:
            import traceback
            logger.error(
                f"[tray] Icon ctor failed: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )
            return
        self._thread = threading.Thread(
            target=self._icon_run_wrapped, name="veilguard-tray", daemon=True,
        )
        self._thread.start()
        logger.info("[tray] icon thread started")

    def _icon_run_wrapped(self) -> None:
        """pystray.Icon.run() in a thread without a top-level handler
        eats exceptions silently.  Wrap it so a crash inside the tray
        loop actually shows up in logs instead of just disappearing."""
        try:
            self._icon.run()
        except Exception as e:
            import traceback
            logger.error(
                f"[tray] icon.run() crashed: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )

    def stop(self) -> None:
        if self._icon is not None:
            try:
                self._icon.stop()
            except Exception:
                pass

    def refresh(self) -> None:
        """Repaint icon + rebuild menu from current registry state."""
        if self._icon is None:
            return
        state = self._derive_state()
        try:
            self._icon.icon = _make_icon(state)
            self._icon.title = self._title()
            self._icon.menu = self._build_menu()
            self._cur_state = state
        except Exception as e:
            logger.debug(f"[tray] refresh failed: {e}")

    # ── State derivation ───────────────────────────────────────────────

    def _derive_state(self) -> str:
        if not self._connected:
            return "offline"
        c = TaskRegistry.get().counts()
        if c["pending"] > 0:
            return "approval"
        if c["running"] > 0:
            return "busy"
        return "idle"

    def _title(self) -> str:
        c = TaskRegistry.get().counts()
        parts = [f"Veilguard"]
        if not self._connected:
            parts.append("offline — reconnecting…")
        if c["pending"]:
            parts.append(f"⚠ {c['pending']} approval{'s' if c['pending']>1 else ''} waiting")
        if c["running"]:
            parts.append(f"{c['running']} running")
        if self._get_status_summary:
            try:
                extra = self._get_status_summary()
                if extra:
                    parts.append(extra)
            except Exception:
                pass
        return " — ".join(parts)

    # ── Menu construction ──────────────────────────────────────────────

    def _build_menu(self):
        Menu = _pystray.Menu
        Item = _pystray.MenuItem

        counts = TaskRegistry.get().counts()

        # IMPORTANT: pystray 0.19's MenuItem._assert_action rejects any
        # callable with MORE than 1 parameter.  Old habit of
        # `lambda icon, item: ...` (the docs example) is what pystray
        # rejects with `ValueError: <function ...>`.  Use single-param
        # `lambda item:` OR `lambda *_:` to be defensive.

        items = [
            Item(
                f"Open task viewer ({counts['running']}r / "
                f"{counts['pending']}p / {counts['completed']}c)",
                lambda _item: self._on_open_viewer(),
                default=True,
            ),
        ]

        # Pending-approval sub-menu surfaces each waiting toast so the
        # user can re-trigger or inspect (the toast itself may have
        # auto-dismissed after the OS's screen-time budget).
        if counts["pending"]:
            snap = TaskRegistry.get().snapshot()
            pending_items = []
            for rec in snap["pending"][:10]:
                label = f"{rec['tool']} — {rec['args_preview'][:40]}"
                pending_items.append(
                    Item(label, lambda _item, r=rec: self._on_open_viewer())
                )
            items.append(Item("Pending approvals", Menu(*pending_items)))

        items.append(Menu.SEPARATOR)

        # Permission-level submenu.  Sends a `set_permission_level` WS
        # message back to sub-agents (handled by daemon code that
        # registered the on_set_level callback).  pystray supports
        # `checked=fn` which renders a radio-style indicator next to
        # the currently-active option — we read the local cached
        # level (set when the user clicks; stays in sync with what
        # sub-agents has) so the menu shows ● next to the active.
        if self._on_set_level:
            level_items = []
            for level in ("auto", "confirm", "strict"):
                # NOTE on lambda shape: pystray invokes action callbacks
                # with (icon, item) — two positional args.  A
                # `lambda _item, lv=level:` lets the SECOND positional
                # arg override the kwarg default, binding `lv` to the
                # MenuItem object instead of the captured level string.
                # That was the "TypeError: MenuItem is not JSON
                # serializable" smoking gun in 0.9.2.  `*args` swallows
                # whatever pystray passes and preserves the closure
                # binding — works the same for the `checked` callback.
                level_items.append(
                    Item(
                        f"{level.capitalize()}",
                        lambda *args, lv=level: self._set_level_clicked(lv, "user"),
                        radio=True,
                        checked=lambda *args, lv=level: (
                            self._current_level == lv
                        ),
                    )
                )
            items.append(Item("Permission level", Menu(*level_items)))

        items.append(Menu.SEPARATOR)
        # Re-pair: wipe credentials + open LibreChat so the user can
        # click "Set up Veilguard" with a fresh config slate.  Useful
        # when switching between dev/prod or to recover from a
        # poisoned token state.
        items.append(
            Item("Re-pair to a new server…",
                 lambda _item: self._on_repair()),
        )
        items.append(Menu.SEPARATOR)
        items.append(
            Item("Quit Veilguard daemon",
                 lambda _item: self._on_quit_clicked()),
        )
        return Menu(*items)

    def _on_quit_clicked(self):
        try:
            self._on_quit()
        finally:
            self.stop()

    def _default_repair(self) -> None:
        """Default re-pair flow: pop a dialog asking which server.

        The dialog has quick-pick buttons for known envs (local / prod)
        + a text field for custom URLs.  On submit we wipe the token +
        user_id from config.yaml and open the chosen URL in the browser
        so the user can click 'Set up Veilguard' there to mint a fresh
        deep-link.
        """
        # Has to run on a fresh thread because pystray's menu callback
        # is on the tray thread and tkinter wants its own mainloop.
        threading.Thread(
            target=_open_repair_dialog,
            name="veilguard-repair-dialog",
            daemon=True,
        ).start()

    # ── Default viewer ─────────────────────────────────────────────────

    def _default_viewer(self) -> None:
        """Spawn the dark-themed tkinter task viewer + logs window."""
        logger.info("[tray] _default_viewer fired — preparing tk window")

        # Re-open path: if we've already built the Tk root, the Tk
        # event-loop thread is still alive (mainloop only exits when
        # root.destroy() is called, which we no longer do — close button
        # withdraws instead).  Just schedule a deiconify on the Tk thread.
        # Recreating tk.Tk() in the same Python process after a prior
        # destroy() leaves the Tcl interpreter in a broken state where
        # subsequent ttk/widget calls hang silently — that's the bug
        # behind "second click does nothing until daemon restart".
        with _VIEWER_STATE["lock"]:
            existing_root = _VIEWER_STATE.get("root")
        if existing_root is not None:
            try:
                existing_root.after(0, _reveal_existing_viewer, existing_root)
                logger.info("[tray] re-shown existing viewer via root.after")
                return
            except Exception as e:
                logger.warning(
                    f"[tray] root.after failed (root likely dead — "
                    f"recreating): {e}"
                )
                with _VIEWER_STATE["lock"]:
                    _VIEWER_STATE["root"] = None
                # fall through to fresh build

        try:
            import tkinter as tk
            from tkinter import ttk
            logger.info(f"[tray] tk imported OK; TkVersion={tk.TkVersion}")
        except Exception as e:
            logger.error(
                f"[tray] tkinter import FAILED — viewer cannot open: {e}",
                exc_info=True,
            )
            return

        # ── Dark palette (tuned to match the LibreChat web aesthetic) ──
        # Deeper navy backgrounds + teal-mint accent (matches the avatar
        # circle hue in the web UI's chat sidebar).  Borders are nearly
        # invisible — LibreChat relies on background contrast, not
        # divider lines, to separate regions.
        BG          = "#0a0e14"   # window background — deeper than GitHub dark
        BG_PANEL    = "#11151c"   # panel / row hover
        BG_HEADER   = "#161b24"   # table header — subtle differentiation
        BG_ELEVATED = "#1b2230"   # selected tab / hover surface
        FG          = "#e8ecf3"   # primary text — slightly warmer
        FG_MUTED    = "#7a8497"   # muted text
        ACCENT      = "#3ddc97"   # teal-mint (Veilguard's signature)
        ACCENT_DIM  = "#1f9b6b"   # darker shade for active states
        BORDER      = "#1f2630"   # near-invisible divider

        def _open():
            logger.info("[tray] viewer _open() running on thread; calling tk.Tk()")
            try:
                root = tk.Tk()
            except Exception as _e:
                logger.error(
                    f"[tray] tk.Tk() FAILED — DISPLAY/COM init issue?: {_e}",
                    exc_info=True,
                )
                return
            logger.info("[tray] tk root created; configuring window")
            root.title("Veilguard")
            root.geometry("1080x640")   # 16:9.5 — more breathing room
            root.minsize(880, 480)
            root.configure(bg=BG)

            # Register the root globally so subsequent "Open task
            # viewer" clicks can re-show this window instead of
            # spawning a new tk.Tk() (which silently breaks the second
            # time around — Tcl interpreter state leakage).
            with _VIEWER_STATE["lock"]:
                _VIEWER_STATE["root"] = root

            # Custom ttk style — overrides default OS-native look with
            # a dark palette + Segoe UI everywhere.  Have to use
            # theme_use('clam') first because the default 'vista' /
            # 'xpnative' themes ignore most colour overrides.
            style = ttk.Style(root)
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
            base_font = ("Segoe UI", 10)
            mono_font = ("Cascadia Mono", 9) if _font_available("Cascadia Mono") \
                        else ("Consolas", 9)

            # Backgrounds + base text (always BG, never inherit OS gray).
            style.configure("TFrame", background=BG)
            style.configure("Panel.TFrame", background=BG_PANEL)
            style.configure("TLabel", background=BG, foreground=FG, font=base_font)
            style.configure(
                "Muted.TLabel",
                background=BG, foreground=FG_MUTED, font=base_font,
            )
            style.configure(
                "H1.TLabel",
                background=BG, foreground=FG,
                font=("Segoe UI", 16, "bold"),
            )
            style.configure(
                "Pill.TLabel",
                background=BG_ELEVATED, foreground=FG_MUTED,
                font=("Segoe UI", 8),
                padding=(10, 4),
            )

            # Notebook = pill-style tabs (LibreChat "My Agents" pill).
            # No active-tab underline, just colour/background change.
            style.configure(
                "TNotebook",
                background=BG, borderwidth=0,
                tabmargins=(0, 4, 0, 0),
            )
            style.configure(
                "TNotebook.Tab",
                background=BG, foreground=FG_MUTED,
                padding=(18, 10),
                font=("Segoe UI", 9, "bold"),
                borderwidth=0,
                focuscolor=BG,
            )
            style.map(
                "TNotebook.Tab",
                background=[
                    ("selected", BG_ELEVATED),
                    ("active",   BG_PANEL),
                ],
                foreground=[
                    ("selected", FG),
                    ("active",   FG),
                ],
            )

            # Treeview — taller rows, no harsh grid, slim header.
            style.configure(
                "Treeview",
                background=BG, foreground=FG, fieldbackground=BG,
                rowheight=34, borderwidth=0, font=base_font,
            )
            style.configure(
                "Treeview.Heading",
                background=BG, foreground=FG_MUTED,
                font=("Segoe UI", 8, "bold"),
                borderwidth=0,
                padding=(10, 8),
                relief="flat",
            )
            style.map(
                "Treeview.Heading",
                background=[("active", BG)],
                relief=[("pressed", "flat"), ("!pressed", "flat")],
            )
            style.map(
                "Treeview",
                background=[("selected", BG_ELEVATED)],
                foreground=[("selected", ACCENT)],
            )
            style.layout("Treeview", [
                ("Treeview.treearea", {"sticky": "nswe"}),
            ])

            # Scrollbars — thin, subtle, dark.
            style.configure(
                "Vertical.TScrollbar",
                background=BG_PANEL, troughcolor=BG, borderwidth=0,
                arrowcolor=FG_MUTED,
            )
            style.map(
                "Vertical.TScrollbar",
                background=[("active", BG_ELEVATED)],
            )

            # ── Top bar — title + version pill on the left, status on the right
            top = ttk.Frame(root)
            top.pack(fill="x", padx=24, pady=(18, 4))
            ttk.Label(
                top, text="Veilguard",
                style="H1.TLabel",
            ).pack(side="left")
            ttk.Label(
                top, text=f"v{_VERSION}",
                style="Pill.TLabel",
            ).pack(side="left", padx=(10, 0), pady=(8, 0))
            # Right side — environment / connection hint (best-effort;
            # falls back to a generic label if no env state is known).
            try:
                env_count = len(_VIEWER_STATE.get("envs") or [])
            except Exception:
                env_count = 0
            ttk.Label(
                top,
                text=("Connected" if env_count else "Local"),
                style="Pill.TLabel",
            ).pack(side="right", pady=(8, 0))

            # Subtle subtitle row.
            sub = ttk.Frame(root)
            sub.pack(fill="x", padx=24, pady=(0, 14))
            ttk.Label(
                sub,
                text="Approve, deny, and review agent tool requests in real time.",
                style="Muted.TLabel",
            ).pack(side="left")

            # ── Notebook with 4 tabs ───────────────────────────────
            nb = ttk.Notebook(root)
            nb.pack(fill="both", expand=True, padx=24, pady=(0, 20))

            snap = TaskRegistry.get().snapshot()

            # 1: Running
            running_frame = ttk.Frame(nb, style="TFrame")
            nb.add(running_frame, text=f"Running  ({len(snap['running'])})")
            _populate_tree(running_frame, snap["running"], mono_font, BG_PANEL)

            # 2: Pending  ← interactive: Approve / Deny / Always buttons
            pending_frame = ttk.Frame(nb, style="TFrame")
            nb.add(pending_frame, text=f"⚠ Pending  ({len(snap['pending'])})")
            pending_tree = _build_pending_pane(
                pending_frame, snap["pending"], mono_font,
                BG, BG_PANEL, FG, FG_MUTED, ACCENT, ACCENT_DIM, BORDER,
            )

            # 3: Completed
            completed_frame = ttk.Frame(nb, style="TFrame")
            nb.add(completed_frame, text=f"Completed  ({len(snap['completed'])})")
            _populate_tree(completed_frame, snap["completed"], mono_font, BG_PANEL)

            # 4: Logs (tails the rotating log file)
            log_frame = ttk.Frame(nb, style="TFrame")
            nb.add(log_frame, text="Logs")
            _build_log_pane(log_frame, mono_font, BG_PANEL, FG, FG_MUTED, BORDER)

            # Bind tab-change to recompute the heading counts so they
            # don't go stale while the window's open.  Also auto-jumps
            # to the Pending tab whenever the count goes 0 → N (so a
            # user with the window open doesn't miss a new approval).
            _state = {"last_pending_count": len(snap["pending"]), "auto_switched": False}

            def _refresh_counts(_event=None):
                s = TaskRegistry.get().snapshot()
                # Bold + warning glyph when there's something to act on.
                running_label   = f"Running  ({len(s['running'])})"
                pending_count   = len(s["pending"])
                pending_label   = (
                    f"⚠ Pending  ({pending_count})" if pending_count
                    else "Pending  (0)"
                )
                completed_label = f"Completed  ({len(s['completed'])})"
                nb.tab(0, text=running_label)
                nb.tab(1, text=pending_label)
                nb.tab(2, text=completed_label)

                # Re-render the pending tree contents whenever it changes.
                _refresh_pending_tree(pending_tree, s["pending"])

                # Auto-switch + bell on new pending.
                if pending_count > _state["last_pending_count"]:
                    try:
                        import winsound as _ws
                        _ws.MessageBeep(_ws.MB_ICONASTERISK)
                    except Exception:
                        try:
                            root.bell()
                        except Exception:
                            pass
                    # Bring the window forward + select the Pending tab.
                    try:
                        nb.select(1)
                        root.lift()
                        root.attributes("-topmost", True)
                        root.after(80, lambda: root.attributes("-topmost", False))
                    except Exception:
                        pass
                _state["last_pending_count"] = pending_count

            nb.bind("<<NotebookTabChanged>>", _refresh_counts)

            # Tight polling so a new pending shows within ~1s.  The
            # try/except around _refresh_counts AND the .after()
            # rescheduling is critical — without it, ANY exception in
            # the chain silently kills the periodic refresh and the
            # viewer goes stale (pending counts wrong, new rows never
            # appear).  This was the "second approval doesn't show in
            # viewer" bug in 0.7-0.8.
            _tick_state = {"n": 0}

            def _tick():
                _tick_state["n"] += 1
                try:
                    _refresh_counts()
                except Exception as _e:
                    logger.warning(
                        f"[tray] _refresh_counts raised (tick #"
                        f"{_tick_state['n']}): {_e}",
                        exc_info=True,
                    )
                # Re-schedule UNCONDITIONALLY — even a Tcl error in
                # after() shouldn't permanently kill the chain.
                try:
                    root.after(1000, _tick)
                except Exception as _e:
                    logger.error(
                        f"[tray] root.after(1000,_tick) FAILED — "
                        f"poll chain dead: {_e}",
                        exc_info=True,
                    )

            root.after(1000, _tick)

            # Close/Escape behaviour: HIDE the window, don't destroy
            # the root.  Destroying the root in the daemon's lifetime
            # makes any subsequent tk.Tk() in this process hang
            # (Tcl interpreter state leak).  Withdrawal keeps the root
            # alive in the background so the next "Open task viewer"
            # click can deiconify it instantly.
            root.protocol("WM_DELETE_WINDOW", root.withdraw)
            root.bind("<Escape>", lambda e: root.withdraw())
            root.lift()
            root.attributes("-topmost", True)
            root.after(80, lambda: root.attributes("-topmost", False))
            logger.info("[tray] viewer window built; entering tk.mainloop()")
            try:
                root.mainloop()
            except Exception as _e:
                logger.error(
                    f"[tray] mainloop raised: {_e}", exc_info=True,
                )
            # Mainloop exits only when root is actually destroyed
            # (daemon shutdown).  Clear the state so we don't try to
            # re-show a dead root.
            with _VIEWER_STATE["lock"]:
                if _VIEWER_STATE.get("root") is root:
                    _VIEWER_STATE["root"] = None
            logger.info("[tray] viewer mainloop exited (root destroyed)")

        try:
            threading.Thread(
                target=_open, name="veilguard-tray-viewer", daemon=True,
            ).start()
            logger.info("[tray] viewer thread started (target=_open)")
        except Exception as _e:
            logger.error(
                f"[tray] failed to start viewer thread: {_e}", exc_info=True,
            )


# ── Tk helpers (module-level so they're picklable / testable) ──────────


def _font_available(name: str) -> bool:
    """Probe tkinter for a font without spamming the log."""
    try:
        import tkinter as _tk
        from tkinter import font as _tkfont
        _root_exists = bool(_tk._default_root)
        if not _root_exists:
            _tk.Tk().withdraw()
        return name in _tkfont.families()
    except Exception:
        return False


def _build_pending_pane(
    parent, records, mono_font,
    BG, BG_PANEL, FG, FG_MUTED, ACCENT, ACCENT_DARK, BORDER,
):
    """Pending tab: Treeview + bottom action bar with Approve / Deny / Always.

    The Treeview's row id is the task's request_id, which is ALSO the
    token in toast._pending — so clicking a button can resolve the
    same future the Windows toast banner would.  Useful when Windows
    silences the banner (Focus Assist, app not registered, etc.) — the
    viewer is then the only way to act.

    Returns the Treeview so the outer poll loop can refresh its rows
    in place without rebuilding the pane.
    """
    import tkinter as tk
    from tkinter import ttk

    cols = ("flag", "tool", "conv", "age", "preview")
    tree = ttk.Treeview(
        parent, columns=cols, show="headings", selectmode="browse",
    )
    headers = (
        ("flag",    "",         34),
        ("tool",    "Tool",    160),
        ("conv",    "Conv",    100),
        ("age",     "Age",      80),
        ("preview", "Details", 360),
    )
    for col_id, col_label, width in headers:
        tree.heading(col_id, text=col_label, anchor="w")
        tree.column(col_id, width=width, anchor="w", stretch=(col_id == "preview"))
    tree.tag_configure("awaiting_user", foreground="#d29922")

    _refresh_pending_tree(tree, records)

    tree.pack(fill="both", expand=True, padx=(2, 2), pady=(10, 0))

    # ── Empty state — only visible when there are no pending rows ─────
    # Drawn as a sibling Label that sits ON TOP of the tree area when
    # there's nothing to show, then withdrawn when rows appear.  Using a
    # Canvas would let us do an icon, but a simple muted Label is clean
    # enough and keeps the painted area honest.
    empty_label = tk.Label(
        parent,
        text="✓  All caught up — no pending approvals.",
        font=("Segoe UI", 11),
        bg="#0a0e14",   # match BG; can't pull from outer scope so hardcode
        fg="#7a8497",
    )
    # Show/hide is managed by _refresh_pending_tree.

    # ── Action bar ────────────────────────────────────────────────────
    bar = ttk.Frame(parent, style="TFrame")
    bar.pack(fill="x", padx=2, pady=(14, 4))

    hint = ttk.Label(
        bar,
        text="Select a request to act on.  Double-click = Approve, Delete = Deny.",
        style="Muted.TLabel",
    )
    hint.pack(side="left")

    # Buttons — wired via tk.Button (not ttk) so we can colour them
    # individually.  Background contrast matters for the green/red
    # affordance.
    def _selected_task_id() -> str:
        sel = tree.selection()
        return sel[0] if sel else ""

    def _act(action: str):
        tid = _selected_task_id()
        if not tid:
            return
        try:
            from daemon import toast as _toast
            ok = _toast.resolve_pending(tid, action=action)
            logger.info(
                f"[viewer] resolve_pending tid={tid[:12]} action={action} ok={ok}"
            )
        except Exception as _e:
            logger.error(f"[viewer] resolve_pending failed: {_e}", exc_info=True)
        # Optimistic UI: remove the row immediately; the next poll will
        # re-sync from the registry.
        try:
            tree.delete(tid)
        except Exception:
            pass

    # Buttons — refined to match LibreChat's flat, modern look.  Larger
    # padding, slightly thinner font, accent on the primary action.
    btn_kw = dict(
        font=("Segoe UI", 10),
        bd=0, relief="flat", padx=18, pady=8,
        activeforeground="#ffffff",
        cursor="hand2",
        highlightthickness=0,
    )
    btn_approve = tk.Button(
        bar, text="✓  Approve",
        bg="#1f9b6b", fg="#ffffff", activebackground="#3ddc97",
        command=lambda: _act("approve"), **btn_kw,
    )
    btn_always = tk.Button(
        bar, text="✓  Always for this conv",
        bg="#1b2230", fg="#e8ecf3", activebackground="#252e3e",
        command=lambda: _act("always"), **btn_kw,
    )
    btn_deny = tk.Button(
        bar, text="✕  Deny",
        bg="#1b2230", fg="#f85149", activebackground="#2a1517",
        command=lambda: _act("deny"), **btn_kw,
    )
    btn_deny.pack(side="right", padx=(0, 0))
    btn_always.pack(side="right", padx=(0, 8))
    btn_approve.pack(side="right", padx=(0, 8))

    # Double-click row = approve (quick path).
    tree.bind("<Double-Button-1>", lambda _e: _act("approve"))
    # Enter key on selected row = approve too.
    tree.bind("<Return>", lambda _e: _act("approve"))
    # Delete / D = deny.
    tree.bind("<Delete>", lambda _e: _act("deny"))

    # Attach the empty-state label so _refresh_pending_tree can
    # show/hide it without needing the closure.
    tree._vg_empty_label = empty_label
    tree._vg_parent_frame = parent

    return tree


def _refresh_pending_tree(tree, records):
    """Re-render rows in the Pending Treeview without rebuilding it."""
    # Show/hide the "all caught up" empty-state label.  Place() it over
    # the tree area when records is empty; place_forget() otherwise.
    try:
        empty_label = getattr(tree, "_vg_empty_label", None)
        if empty_label is not None:
            if records:
                empty_label.place_forget()
            else:
                # Centered roughly within the tree area.
                empty_label.place(
                    relx=0.5, rely=0.4, anchor="center",
                )
    except Exception as _e:
        logger.debug(f"[tree] empty-state toggle failed: {_e}")

    try:
        # Build the new id-set first so we can diff cheaply.
        new_ids = set()
        by_id: dict[str, dict] = {}
        for rec in records:
            tid = rec.get("task_id", "")
            if not tid:
                continue
            new_ids.add(tid)
            by_id[tid] = rec

        # Diagnostic: log what we're working with on every refresh that
        # actually has records — without this, "tab says (1) but no row
        # appears" is unfalsifiable.  Quiet when empty so the logs
        # don't drown in idle polls.
        if records:
            logger.info(
                f"[tree] refresh pending: n_records={len(records)} "
                f"task_ids={sorted(new_ids)}"
            )

        # Remove rows no longer pending (user approved/denied → poll caught it).
        for iid in list(tree.get_children("")):
            if iid not in new_ids:
                try:
                    tree.delete(iid)
                except Exception as _e:
                    logger.warning(f"[tree] delete iid={iid!r} failed: {_e}")

        # Insert new rows + refresh existing ones.
        now = time.time()
        existing = set(tree.get_children(""))
        for tid, rec in by_id.items():
            started_ts = rec.get("started_at") or rec.get("created_at") or now
            age_s = int(max(0, now - started_ts))
            age_str = (
                f"{age_s}s ago" if age_s < 60
                else f"{age_s // 60}m {age_s % 60}s ago"
            )
            conv = (rec.get("conv_id") or "")[:12] or "—"
            preview = rec.get("result_preview") or rec.get("args_preview") or ""
            values = ("⚠", rec.get("tool", ""), conv, age_str, preview[:80])
            if tid in existing:
                try:
                    tree.item(tid, values=values)
                except Exception as _e:
                    logger.warning(f"[tree] item update tid={tid!r} failed: {_e}")
            else:
                try:
                    tree.insert(
                        "", "end", iid=tid, values=values,
                        tags=("awaiting_user",),
                    )
                    logger.info(
                        f"[tree] inserted row iid={tid!r} tool={rec.get('tool')!r}"
                    )
                except Exception as _e:
                    # Most likely: iid syntax not Tcl-safe.  Fall back
                    # to auto-generated iid (Tk picks one for us); the
                    # row will appear but Approve/Deny buttons can't
                    # resolve by selection -> tid map.  Surface loudly.
                    logger.error(
                        f"[tree] INSERT iid={tid!r} FAILED: {_e}",
                        exc_info=True,
                    )
                    try:
                        tree.insert(
                            "", "end", values=values,
                            tags=("awaiting_user",),
                        )
                    except Exception as _e2:
                        logger.error(
                            f"[tree] auto-iid insert ALSO FAILED: {_e2}",
                            exc_info=True,
                        )
    except Exception as e:
        logger.error(
            f"[tree] _refresh_pending_tree raised: {e}", exc_info=True,
        )


def _populate_tree(parent, records, mono_font, hover_bg):
    """Build a styled Treeview inside `parent` showing task records."""
    import tkinter as tk
    from tkinter import ttk
    cols = ("tool", "status", "executor", "started", "preview")
    tree = ttk.Treeview(parent, columns=cols, show="headings", selectmode="browse")
    headers = (
        ("tool",      "Tool",     160),
        ("status",    "Status",    90),
        ("executor",  "Executor",  90),
        ("started",   "Started",  120),
        ("preview",   "Preview",  360),
    )
    for col_id, col_label, width in headers:
        tree.heading(col_id, text=col_label, anchor="w")
        tree.column(col_id, width=width, anchor="w", stretch=(col_id == "preview"))

    # Status-coloured tags.
    tree.tag_configure("running",   foreground="#58a6ff")
    tree.tag_configure("pending",   foreground="#d29922")
    tree.tag_configure("awaiting_user", foreground="#d29922")
    tree.tag_configure("completed", foreground="#3fb950")
    tree.tag_configure("failed",    foreground="#f85149")
    tree.tag_configure("cancelled", foreground="#8b949e")

    for rec in records:
        status = rec.get("status", "")
        started_ts = rec.get("started_at") or rec.get("created_at")
        started = (
            time.strftime("%H:%M:%S", time.localtime(started_ts))
            if started_ts else ""
        )
        tree.insert(
            "", "end",
            values=(
                rec.get("tool", "") or "",
                status,
                rec.get("executor", "daemon-legacy"),
                started,
                (rec.get("result_preview") or rec.get("args_preview") or "")[:80],
            ),
            tags=(status,),
        )

    vsb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
    tree.configure(yscroll=vsb.set)
    tree.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
    vsb.pack(side="right", fill="y", pady=6, padx=(0, 6))


def _build_log_pane(parent, mono_font, panel_bg, fg, fg_muted, border):
    """Tail the daemon log file in a scrollable text widget.

    Polls the file every 1.5s and appends new content.  Caps to the
    last 500 lines on first paint so the buffer stays small.
    """
    import tkinter as tk
    from tkinter import ttk

    # Header bar with the log file path so the user knows where it lives.
    top = ttk.Frame(parent, style="TFrame")
    top.pack(fill="x", padx=6, pady=(6, 0))

    # Resolve the log path the SAME way the daemon does.  We can't
    # import it cleanly without a circular dep, so duplicate the
    # tiny resolver here.
    import os as _os
    if _os.name == "nt":
        base = _os.environ.get("LOCALAPPDATA") or _os.path.expanduser("~\\AppData\\Local")
        log_path = _os.path.join(base, "Veilguard", "daemon.log")
    else:
        log_path = _os.path.join(_os.path.expanduser("~"), ".veilguard", "daemon.log")

    ttk.Label(top, text=f"log: {log_path}",
              style="Muted.TLabel").pack(side="left")

    txt_frame = ttk.Frame(parent, style="TFrame")
    txt_frame.pack(fill="both", expand=True, padx=6, pady=6)

    text = tk.Text(
        txt_frame,
        bg=panel_bg, fg=fg, insertbackground=fg,
        font=mono_font, wrap="none",
        relief="flat", borderwidth=0, padx=8, pady=6,
        selectbackground="#1f6feb", selectforeground="#ffffff",
    )
    text.config(state="disabled")
    ysb = ttk.Scrollbar(txt_frame, orient="vertical", command=text.yview)
    xsb = ttk.Scrollbar(txt_frame, orient="horizontal", command=text.xview)
    text.configure(yscroll=ysb.set, xscroll=xsb.set)
    text.grid(row=0, column=0, sticky="nsew")
    ysb.grid(row=0, column=1, sticky="ns")
    xsb.grid(row=1, column=0, sticky="ew")
    txt_frame.rowconfigure(0, weight=1)
    txt_frame.columnconfigure(0, weight=1)

    # Tag-based colorisation for severity levels.
    text.tag_configure("ERROR",   foreground="#f85149")
    text.tag_configure("WARNING", foreground="#d29922")
    text.tag_configure("INFO",    foreground=fg)
    text.tag_configure("DEBUG",   foreground=fg_muted)

    state = {"offset": 0, "lines_loaded": 0}

    def _read_chunk():
        """Read new bytes since last call + append to widget."""
        try:
            if not _os.path.exists(log_path):
                return
            size = _os.path.getsize(log_path)
            if state["offset"] == 0 and size > 200_000:
                # On first paint, jump to the last ~50 KB so the
                # widget doesn't try to render a multi-MB file.
                state["offset"] = max(0, size - 50_000)
            if size < state["offset"]:
                # Log rotated — start fresh.
                state["offset"] = 0
            if size == state["offset"]:
                return
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                f.seek(state["offset"])
                chunk = f.read()
                state["offset"] = f.tell()
            if not chunk:
                return
            text.config(state="normal")
            for line in chunk.splitlines(keepends=True):
                tag = "INFO"
                if "[ERROR]" in line: tag = "ERROR"
                elif "[WARNING]" in line: tag = "WARNING"
                elif "[DEBUG]" in line: tag = "DEBUG"
                text.insert("end", line, tag)
            # Cap to last 500 lines to keep the widget responsive.
            line_count = int(text.index("end-1c").split(".")[0])
            if line_count > 500:
                text.delete("1.0", f"{line_count - 500}.0")
            text.see("end")
            text.config(state="disabled")
        except Exception:
            pass

    # Kick off the first read + poll loop.
    _read_chunk()
    parent.after(1500, lambda: (_read_chunk(),
                                parent.after(1500, lambda p=parent: _build_log_pane.__poll(p)) if False
                                else None))

    # Persistent polling — wire as a proper after-loop.
    def _poll():
        _read_chunk()
        try:
            parent.after(1500, _poll)
        except Exception:
            pass
    parent.after(1500, _poll)


# Single source of truth: pull from veilguard_client.__version__ so a
# bump there flows to every window title / "About" surface without a
# second edit.  Frozen PyInstaller bundles flatten the package layout,
# so we try a few import paths before falling back.
def _resolve_version() -> str:
    try:
        from veilguard_client import __version__  # type: ignore[import-not-found]
        return __version__
    except Exception:
        pass
    try:
        # Fallback: read directly from the source file if the import
        # path is unavailable (dev runs from outside the daemon dir).
        import os as _os
        src = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "veilguard_client.py",
        )
        with open(src, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return "0.0.0+unknown"


_VERSION = _resolve_version()


# ── Viewer-window state (persistent across open/close cycles) ───────────
#
# The first time the user clicks "Open task viewer", we spawn a thread
# that creates the Tk root + runs mainloop().  We DELIBERATELY never
# destroy that root — subsequent close-the-window actions just
# withdraw() it (hide), and subsequent opens deiconify().  Recreating
# tk.Tk() in the same Python process is fragile (Tcl interpreter state
# leaks, ttk styles fail silently), so we keep ONE root for the
# daemon's lifetime.
_VIEWER_STATE: dict = {"root": None, "lock": threading.Lock()}


def _reveal_existing_viewer(root) -> None:
    """Bring the persistent viewer window forward.

    Called via ``root.after(0, _reveal_existing_viewer, root)`` from
    the tray-menu thread so the actual Tk manipulation runs on the Tk
    thread (mandatory — Tk is not free-threaded).
    """
    try:
        root.deiconify()
        root.lift()
        root.attributes("-topmost", True)
        root.after(80, lambda: root.attributes("-topmost", False))
    except Exception as e:
        logger.warning(f"[tray] _reveal_existing_viewer raised: {e}")


# ── Re-pair dialog ─────────────────────────────────────────────────────


def _open_repair_dialog() -> None:
    """Show the dark Re-pair dialog and act on the result.

    Quick-pick buttons cover the two environments users actually
    switch between (local dev + prod).  Custom URL field handles
    everything else (staging, alternate ports, etc.).  On submit we
    wipe the token + user_id from config.yaml so the spear-phish
    guard in deeplink.apply_configure won't block the next pairing.
    """
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as e:
        logger.warning(f"[tray] tkinter unavailable for re-pair dialog: {e}")
        return

    # Resolve current server from config.yaml so the input field can
    # show what we'd hit by default (lets the user just hit Enter to
    # re-pair to the same place, if that's actually what they want).
    import os
    try:
        import yaml
    except Exception:
        yaml = None
    cfg_path = os.path.join(os.path.expanduser("~"), ".veilguard", "config.yaml")
    current_server_url = ""
    if os.path.exists(cfg_path) and yaml is not None:
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            srv = (cfg.get("server") or "").strip()
            if srv.startswith("wss://"):
                current_server_url = "https://" + srv[len("wss://"):].split("/")[0]
            elif srv.startswith("ws://"):
                current_server_url = "http://" + srv[len("ws://"):].split("/")[0]
        except Exception as e:
            logger.debug(f"[tray] read current server failed: {e}")

    # Same palette as the task viewer.
    BG          = "#0d1117"
    BG_PANEL    = "#161b22"
    BG_INPUT    = "#0d1117"
    FG          = "#e6edf3"
    FG_MUTED    = "#8b949e"
    ACCENT      = "#1f6feb"
    ACCENT_HOV  = "#388bfd"
    BORDER      = "#30363d"
    DANGER      = "#f85149"

    root = tk.Tk()
    root.title("Veilguard — Re-pair")
    root.geometry("560x440")
    root.configure(bg=BG)
    root.resizable(False, False)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=FG,
                    font=("Segoe UI", 10))
    style.configure("Muted.TLabel", background=BG, foreground=FG_MUTED,
                    font=("Segoe UI", 9))
    style.configure("Heading.TLabel", background=BG, foreground=FG,
                    font=("Segoe UI", 14, "bold"))
    style.configure("Danger.TLabel", background=BG, foreground=DANGER,
                    font=("Segoe UI", 9))
    # Buttons.
    style.configure(
        "Accent.TButton",
        background=ACCENT, foreground="#ffffff",
        font=("Segoe UI", 10, "bold"),
        padding=(16, 8), borderwidth=0,
    )
    style.map(
        "Accent.TButton",
        background=[("active", ACCENT_HOV), ("pressed", ACCENT_HOV)],
    )
    style.configure(
        "Quiet.TButton",
        background=BG_PANEL, foreground=FG,
        font=("Segoe UI", 10), padding=(12, 6), borderwidth=0,
    )
    style.map("Quiet.TButton", background=[("active", BG_PANEL)])

    # ── Header ──────────────────────────────────────────────────────
    container = ttk.Frame(root, style="TFrame")
    container.pack(fill="both", expand=True, padx=24, pady=20)
    ttk.Label(container, text="Pair this machine", style="Heading.TLabel").pack(anchor="w")
    ttk.Label(
        container,
        text=(
            "Wipes the saved token + user_id so the next pairing link "
            "can configure a fresh server.  Pick a quick environment "
            "below or paste any LibreChat URL."
        ),
        style="Muted.TLabel",
        wraplength=510,
        justify="left",
    ).pack(anchor="w", pady=(6, 16))

    if current_server_url:
        ttk.Label(
            container,
            text=f"Currently paired with: {current_server_url}",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(0, 12))

    # ── URL entry ───────────────────────────────────────────────────
    ttk.Label(container, text="LibreChat URL").pack(anchor="w")
    entry_var = tk.StringVar(value=current_server_url or "http://localhost:3080")
    entry = tk.Entry(
        container, textvariable=entry_var,
        bg=BG_INPUT, fg=FG, insertbackground=FG,
        font=("Cascadia Mono", 10) if _font_available("Cascadia Mono")
              else ("Consolas", 10),
        relief="flat", borderwidth=1,
        highlightthickness=1, highlightbackground=BORDER,
        highlightcolor=ACCENT,
    )
    entry.pack(fill="x", pady=(4, 14), ipady=6)
    entry.focus_set()
    entry.select_range(0, "end")

    # ── Quick-picks ─────────────────────────────────────────────────
    ttk.Label(container, text="Quick picks", style="Muted.TLabel").pack(anchor="w")

    quick_frame = ttk.Frame(container, style="TFrame")
    quick_frame.pack(fill="x", pady=(6, 18))

    def _fill(url: str) -> None:
        entry_var.set(url)
        entry.focus_set()

    ttk.Button(
        quick_frame, text="Local  http://localhost:3080",
        style="Quiet.TButton",
        command=lambda: _fill("http://localhost:3080"),
    ).pack(side="left", padx=(0, 8))
    ttk.Button(
        quick_frame, text="Production  veilguard.phishield.com",
        style="Quiet.TButton",
        command=lambda: _fill("https://veilguard.phishield.com"),
    ).pack(side="left")

    # ── Status line ─────────────────────────────────────────────────
    status_var = tk.StringVar(value="")
    status_lbl = ttk.Label(container, textvariable=status_var, style="Danger.TLabel")
    status_lbl.pack(anchor="w", pady=(4, 0))

    # ── Action buttons ──────────────────────────────────────────────
    def _on_repair() -> None:
        url = entry_var.get().strip().rstrip("/")
        if not url:
            status_var.set("URL is required.")
            return
        if not (url.startswith("http://") or url.startswith("https://")):
            status_var.set("URL must start with http:// or https://")
            return
        # Wipe credentials.
        if yaml is None or not os.path.exists(cfg_path):
            status_var.set("No config.yaml to wipe — will still open the URL.")
        else:
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                cfg.pop("token", None)
                cfg.pop("user_id", None)
                # Best-effort: clear the server field too so the daemon
                # genuinely has nothing to authenticate against until
                # the user clicks Set up.
                cfg.pop("server", None)
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, default_flow_style=False)
                logger.info("[tray] re-pair wiped server + token + user_id")
            except Exception as e:
                logger.warning(f"[tray] config wipe failed: {e}")
                status_var.set(f"Could not wipe config: {e}")
                return

        import webbrowser
        try:
            webbrowser.open_new_tab(url)
            logger.info(f"[tray] re-pair → opened {url}")
        except Exception as e:
            logger.warning(f"[tray] couldn't open browser: {e}")
            status_var.set(f"Browser failed: {e}.  URL: {url}")
            return

        root.destroy()

    buttons = ttk.Frame(container, style="TFrame")
    buttons.pack(fill="x", side="bottom", pady=(20, 0))
    ttk.Button(
        buttons, text="Cancel", style="Quiet.TButton",
        command=root.destroy,
    ).pack(side="right", padx=(8, 0))
    ttk.Button(
        buttons, text="Wipe creds & open",
        style="Accent.TButton", command=_on_repair,
    ).pack(side="right")

    # Enter / Escape shortcuts.
    root.bind("<Return>", lambda e: _on_repair())
    root.bind("<Escape>", lambda e: root.destroy())

    root.lift()
    root.attributes("-topmost", True)
    root.after(80, lambda: root.attributes("-topmost", False))
    root.mainloop()


__all__ = ["HAS_TRAY_UI", "TrayController"]
