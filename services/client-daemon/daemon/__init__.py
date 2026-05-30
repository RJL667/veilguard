"""Daemon-side sub-modules (UI, task tracking, approval handlers).

Imported conditionally from veilguard_client.py:
  - `tasks` is ALWAYS loaded (data structures, no GUI deps)
  - `toast` is loaded only on Windows with `winotify` available
  - `tray` is loaded only on Windows with `pystray` available

Headless / sandbox builds skip the UI modules entirely.
"""
