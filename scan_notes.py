
import csv
import getpass
import re
import requests

BASE = "http://127.0.0.1:41184"
TOKEN = getpass.getpass("Joplin API token: ")

def api(path, **params):
    response = requests.get(
        BASE + path,
        params={"token": TOKEN, **params},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()

# Only examine notes with generic-looking titles
pattern = re.compile(
    r"^(pages from scan|scan[\s_-]|document\b|note\s*[-–])",
    re.IGNORECASE,
)

results = []
page = 1

while len(results) < 25:
    batch = api(
        "/notes",
        fields="id,title",
        limit=100,
        page=page,
    )

    for note in batch["items"]:
        if not pattern.search(note["title"]):
            continue

        resources = api(
            f"/notes/{note['id']}/resources",
            fields="id,title,mime,ocr_text,ocr_status",
            limit=100,
        )["items"]

        pdfs = [
            r for r in resources
            if r.get("mime") == "application/pdf"
        ]

        if not pdfs:
            continue

        ocr = "\n".join(
            r.get("ocr_text") or "" for r in pdfs
        )

        results.append({
            "note_id": note["id"],
            "current_title": note["title"],
            "pdf_count": len(pdfs),
            "ocr_chars": len(ocr),
            "ocr_preview": ocr[:600],
        })

        if len(results) >= 25:
            break

    if not batch.get("has_more"):
        break
    page += 1

with open(
    "joplin_candidates.csv",
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:
    writer = csv.DictWriter(
        f, fieldnames=[
            "note_id", "current_title",
            "pdf_count", "ocr_chars", "ocr_preview"
        ]
    )
    writer.writeheader()
    writer.writerows(results)

print(f"Exported {len(results)} candidates.")
