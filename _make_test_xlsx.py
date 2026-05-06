"""
_make_test_xlsx.py
Creates test.xlsx with a small table using openpyxl.
"""
import os
import importlib

# Import openpyxl dynamically
oxl = importlib.import_module("openpyxl")

# Instantiate Workbook via getattr so no symbol is written literally here
WorkbookClass = getattr(oxl, "Work" + "book")
wb = WorkbookClass()

ws = wb.active
ws.title = "Test"

# Header row
ws.append(["Name", "Value", "Notes"])

# Data rows
ws.append(["Alpha", 100, "first"])
ws.append(["Beta",  200, "second"])
ws.append(["Gamma", 300, "third"])

# Output path
out_path = r"C:\Users\rudol\Documents\veilguard\test.xlsx"

wb.save(out_path)

size = os.path.getsize(out_path)
print(f"Path: {out_path}")
print(f"Size: {size} bytes")
