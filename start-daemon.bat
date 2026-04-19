@echo off
title Veilguard Client Daemon
cd /d "C:\Users\rudol\Documents\veilguard\mcp-tools\client-daemon"
python veilguard_client.py --server ws://localhost:8809/ws/client --token 63fb85a1b78e14f1a455ef5c20b17934 --client-id rudolph-desktop --project-root "C:\Users\rudol\Documents\veilguard"
pause
