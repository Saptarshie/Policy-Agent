"""Does deleting a document also remove the stored upload?"""
import pathlib
import time

import requests

BASE = "http://127.0.0.1:8077"
UP = pathlib.Path("C:/Users/sapta/AppData/Local/hermes/cache/scratch/ui_verify/uploads")
src = pathlib.Path("C:/Users/sapta/AppData/Local/hermes/cache/scratch/ui_check_leak.txt")
src.write_text("Leak test policy: interns get 2 casual leave days per year.\n",
               encoding="utf-8")

before = {p.name for p in UP.glob("*")}
with open(src, "rb") as fh:
    r = requests.post(f"{BASE}/api/documents",
                      files={"file": ("ui_check_leak.txt", fh, "text/plain")}, timeout=120)
print("upload:", r.status_code, r.text)
doc_id = r.json()["doc_id"]
for _ in range(40):
    docs = {d["id"]: d for d in requests.get(f"{BASE}/api/documents", timeout=30).json()}
    if docs.get(doc_id, {}).get("status") in ("ready", "error"):
        break
    time.sleep(2)
added = {p.name for p in UP.glob("*")} - before
print("stored upload created:", added, "| doc status:", docs.get(doc_id, {}).get("status"))
r = requests.delete(f"{BASE}/api/documents/{doc_id}", timeout=60)
print("delete:", r.status_code)
left = {p.name for p in UP.glob("*")}
print("stored file removed on delete:", not (added & left), "| uploads now:", len(left))
