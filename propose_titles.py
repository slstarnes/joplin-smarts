
import json
import os
import re
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

JOPLIN = "http://127.0.0.1:41184"
OLLAMA = "http://127.0.0.1:11434/api/generate"
MODEL = "qwen2.5:3b"
MAX_RESULTS = 5

TOKEN = os.getenv("joplin_token")
session = requests.Session()

def joplin_get(path, **params):
    response = session.get(
        JOPLIN + path,
        params={"token": TOKEN, **params},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()

def propose_title(text):
    prompt = f"""
You are organizing a personal document archive.

Read the document text and propose a concise,
descriptive title.

Rules:
- Identify the organization and type of document.
- Include a date only if clearly supported by the text.
- Do not infer a document date from a scan filename.
- Never invent information.
- Exclude account numbers and other sensitive identifiers.
- If the document is unclear, return an empty title.
- Return JSON with suggested_title and explanation.

Preferred format:
YYYY-MM-DD - Organization - Document Type

DOCUMENT:
{text[:5000]}
"""
    response = session.post(
        OLLAMA,
        json={
            "model": MODEL,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    return json.loads(response.json()["response"])

generic = re.compile(
    r"^(pages from scan|scan[\s_-]|"
    r"document\b|note\s*[-–]|img[_-])",
    re.IGNORECASE,
)

results = []
page = 1

while len(results) < MAX_RESULTS:
    batch = joplin_get(
        "/notes",
        fields="id,title",
        page=page,
        limit=100,
    )

    for note in batch["items"]:
        if not generic.search(note["title"]):
            continue

        attached = joplin_get(
            f"/notes/{note['id']}/resources",
            fields="id,mime",
            limit=100,
        )

        texts = []
        for resource in attached["items"]:
            if resource["mime"] != "application/pdf":
                continue

            details = joplin_get(
                f"/resources/{resource['id']}",
                fields="ocr_text,ocr_status",
            )
            if details.get("ocr_text"):
                texts.append(details["ocr_text"])

        if not texts:
            continue

        text = "\n\n".join(texts)

        try:
            suggestion = propose_title(text)
        except (requests.RequestException, ValueError) as e:
            print(f"AI error: {e}")
            continue

        row = {
            "note_id": note["id"],
            "old_title": note["title"],
            "proposed_title": suggestion.get(
                "suggested_title", ""
            ),
            "explanation": suggestion.get(
                "explanation", ""
            ),
            "ocr_chars": len(text),
            "approved": "",
        }
        results.append(row)

        print(f"\nOLD: {row['old_title']}")
        print(f"NEW: {row['proposed_title']}")

        if len(results) >= MAX_RESULTS:
            break

    if not batch.get("has_more"):
        break
    page += 1

columns = [
    "note_id", "old_title", "proposed_title",
    "explanation", "ocr_chars", "approved",
]

with open(
    "title_suggestions.csv",
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(results)

print(f"\nGenerated {len(results)} suggestions.")
print("No Joplin notes were modified.")
