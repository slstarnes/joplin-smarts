
import csv
import os
import json
import re
import time
from datetime import date, datetime

import requests
from dateutil import parser as date_parser
from dotenv import load_dotenv

load_dotenv(override=True)


# --------------------------------------------------
# Configuration
# --------------------------------------------------
JOPLIN_URL = "http://127.0.0.1:41184"
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

MODEL = "qwen2.5:3b"
MAX_RESULTS = 5
MAX_OCR_CHARS = 5000
OUTPUT_CSV = "title_suggestions.csv"
CONTEXT_SIZE = 8192

# Approximate character budget, not tokens.
OCR_HEAD_CHARS = 9000
OCR_TAIL_CHARS = 1500

# Generic titles to investigate
GENERIC_TITLE = re.compile(
    r"^(pages from scan|scan[\s_-]|"
    r"document\b|note\s*[-–]|img[_-])",
    re.IGNORECASE,
)

OUTPUT_CSV = (
    "title_suggestions_"
    + datetime.now().strftime("%Y%m%d_%H%M%S")
    + ".csv"
)


TOKEN = os.getenv("joplin_token")
session = requests.Session()


# --------------------------------------------------
# Joplin API
# --------------------------------------------------
# -------------------------------------------
# Joplin API
# -------------------------------------------

def joplin_get(path, **params):
    response = session.get(
        JOPLIN_URL + path,
        params={"token": TOKEN, **params},
        timeout=45,
    )
    response.raise_for_status()
    return response.json()


def joplin_pages(path, fields):
    page = 1

    while True:
        result = joplin_get(
            path,
            fields=fields,
            limit=100,
            page=page,
        )

        yield from result["items"]

        if not result.get("has_more"):
            break

        page += 1


def get_pdf_content(note_id):
    texts = []
    filenames = []
    pdf_count = 0

    for resource in joplin_pages(
        f"/notes/{note_id}/resources",
        "id,title,mime",
    ):
        if resource.get("mime") != "application/pdf":
            continue

        pdf_count += 1

        filenames.append(
            resource.get("title") or ""
        )

        details = joplin_get(
            f"/resources/{resource['id']}",
            fields="ocr_text,ocr_status",
        )

        text = details.get("ocr_text") or ""

        if text.strip():
            texts.append(text)

    return "\n\n".join(texts), filenames, pdf_count


# -------------------------------------------
# Filename date extraction
# -------------------------------------------

def extract_filename_date(filenames):
    """
    Recognizes:
      scan_20190128_0001.pdf
      scan11192018-175050.pdf
      document_2023-10-01.pdf

    Returns (ISO date, source filename).
    """

    patterns = [
        (
            re.compile(
                r"(?<!\d)((?:19|20)\d{2})"
                r"[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)"
            ),
            "ymd",
        ),
        (
            re.compile(
                r"(?<!\d)(0[1-9]|1[0-2])"
                r"([0-3]\d)((?:19|20)\d{2})(?!\d)"
            ),
            "mdy",
        ),
    ]

    for filename in filenames:
        for pattern, order in patterns:
            for match in pattern.finditer(filename):
                a, b, c = map(int, match.groups())

                if order == "ymd":
                    year, month, day = a, b, c
                else:
                    month, day, year = a, b, c

                try:
                    result = date(year, month, day)
                    return result.isoformat(), filename
                except ValueError:
                    continue

    return "", ""


def filename_description(filename):
    """
    Extracts a useful human label from a
    generic scanned filename.
    """

    name = Path(filename).name

    name = re.sub(
        r"\.(pdf|jpg|jpeg|png|tif|tiff)$",
        "",
        name,
        flags=re.IGNORECASE,
    )

    name = re.sub(
        r"^pages\s+from\s+",
        "",
        name,
        flags=re.IGNORECASE,
    )

    name = re.sub(
        r"^scan",
        "",
        name,
        flags=re.IGNORECASE,
    )

    # Remove scan date / timestamp patterns.
    name = re.sub(
        r"(?<!\d)(?:19|20)\d{2}"
        r"[-_]?[01]\d[-_]?[0-3]\d(?!\d)",
        "",
        name,
    )

    name = re.sub(
        r"(?<!\d)(?:0[1-9]|1[0-2])"
        r"(?:[0-3]\d)(?:19|20)\d{2}(?!\d)",
        "",
        name,
    )

    # Remove leading page numbers, IDs, etc.
    name = re.sub(
        r"^[\s_-]*(?:\d+[\s_-]*)+",
        "",
        name,
    )

    name = re.sub(r"[_-]+", " ", name)
    name = " ".join(name.split()).strip()

    # These are not meaningful labels.
    if re.fullmatch(
        r"(document|scan|image|pages?)(?:\s*\d+)?",
        name,
        flags=re.IGNORECASE,
    ):
        return ""

    return name


# -------------------------------------------
# OCR excerpt
# -------------------------------------------

def ocr_excerpt(text):
    limit = OCR_HEAD_CHARS + OCR_TAIL_CHARS

    if len(text) <= limit:
        return text

    return (
        text[:OCR_HEAD_CHARS]
        + "\n\n[OCR MIDDLE OMITTED]\n\n"
        + text[-OCR_TAIL_CHARS:]
    )


# -------------------------------------------
# Ollama structured output
# -------------------------------------------

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "organization": {"type": "string"},
        "document_type": {"type": "string"},
        "document_date": {"type": "string"},
        "date_evidence": {"type": "string"},
        "explanation": {"type": "string"},
    },
    "required": [
        "organization",
        "document_type",
        "document_date",
        "date_evidence",
        "explanation",
    ],
}


def propose_title(note_title, filenames, ocr_text):
    prompt = f"""
You organize a personal archive of scanned documents.

Identify what this document is.

Return JSON matching the supplied schema.

GENERAL RULES:
- Prefer information from OCR document contents.
- Use filename labels as secondary clues.
- The filename may describe the document.
- Never invent organizations, dates, or document types.
- Do not include personal identifiers or account numbers.
- Use empty strings for unknown fields.
- Never output placeholders like "Not Specified".

DOCUMENT DATE RULES:
- document_date must come from OCR text ONLY.
- Do not use filename dates in document_date.
- Never invent or estimate a date.
- Prefer the document's issue or effective date.
- Distinguish document dates from due dates,
  expiration dates, and unrelated dates.
- Use YYYY-MM-DD only for complete dates.
- Copy the supporting date text VERBATIM
  into date_evidence.
- If uncertain, leave both fields empty.

DOCUMENT TYPE:
- Identify the most specific supported type.
- Examples: Insurance Renewal, Traffic Citation,
  Business Associate Agreement, 529 Confirmation.
- Do not infer more than the evidence supports.

ORGANIZATION:
- Use the actual issuing organization if identified.
- Otherwise leave empty.

EXPLANATION:
- Briefly explain the identification.
- Mention uncertainty when appropriate.

EXISTING NOTE TITLE:
{note_title}

ATTACHMENT FILENAMES:
{json.dumps(filenames)}

OCR TEXT:
{ocr_excerpt(ocr_text)}
"""

    response = session.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "format": OUTPUT_SCHEMA,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": CONTEXT_SIZE,
                "num_predict": 400,
            },
        },
        timeout=300,
    )

    response.raise_for_status()

    result = json.loads(response.json()["response"])

    if not isinstance(result, dict):
        raise ValueError("Model did not return a JSON object")

    return result


# -------------------------------------------
# OCR date validation
# -------------------------------------------

def normalize(text):
    return " ".join(str(text).split()).casefold()


def verify_ocr_date(document_date, evidence, ocr_text):
    if not document_date:
        return False, "No OCR date"

    if not evidence:
        return False, "Missing date evidence"

    if normalize(evidence) not in normalize(ocr_text):
        return False, "Evidence missing from OCR"

    try:
        proposed = date.fromisoformat(
            str(document_date).strip()
        )
    except ValueError:
        return False, "Invalid ISO date"

    # Require an explicit year in the evidence.
    year_pattern = (
        r"\b(?:19|20)\d{2}\b"
        r"|[/.-]\d{2}(?!\d)"
    )

    if not re.search(year_pattern, evidence):
        return False, "Year not supported"

    try:
        parsed = date_parser.parse(
            evidence,
            fuzzy=True,
            dayfirst=False,
            default=datetime(1900, 1, 1),
        ).date()
    except (ValueError, TypeError, OverflowError):
        return False, "Cannot parse evidence"

    if parsed != proposed:
        return False, (
            f"Evidence date {parsed} differs "
            f"from proposed {proposed}"
        )

    return True, "Verified against OCR"


# -------------------------------------------
# Build proposed title
# -------------------------------------------

def clean_field(value):
    result = " ".join(str(value or "").split())

    if result.casefold() in {
        "unknown",
        "not specified",
        "none",
        "null",
        "n/a",
    }:
        return ""

    return result[:100]


def build_title(suggestion, ocr_text, filenames):
    organization = clean_field(
        suggestion.get("organization")
    )

    document_type = clean_field(
        suggestion.get("document_type")
    )

    ocr_date = clean_field(
        suggestion.get("document_date")
    )

    evidence = clean_field(
        suggestion.get("date_evidence")
    )

    verified, status = verify_ocr_date(
        ocr_date,
        evidence,
        ocr_text,
    )

    filename_date, source_filename = (
        extract_filename_date(filenames)
    )

    if verified:
        date_used = ocr_date
        date_source = "ocr_verified"
    elif filename_date:
        date_used = filename_date
        date_source = "filename_scan_date"
    else:
        date_used = ""
        date_source = "none"

    # Use a filename description only if AI
    # could not identify the document type.
    if not document_type:
        for filename in filenames:
            label = filename_description(filename)
            if label:
                document_type = label
                break

    if not document_type:
        return {
            "title": "",
            "date_used": date_used,
            "date_source": date_source,
            "date_status": status,
            "filename_date": filename_date,
            "filename_source": source_filename,
        }

    parts = []

    if date_used:
        parts.append(date_used)

    if organization:
        parts.append(organization)

    if document_type.casefold() != organization.casefold():
        parts.append(document_type)

    return {
        "title": " - ".join(parts),
        "date_used": date_used,
        "date_source": date_source,
        "date_status": status,
        "filename_date": filename_date,
        "filename_source": source_filename,
    }


# -------------------------------------------
# Main
# -------------------------------------------

COLUMNS = [
    "note_id",
    "old_title",
    "proposed_title",
    "filenames",
    "organization",
    "document_type",
    "ocr_date_proposed",
    "date_evidence",
    "date_used",
    "date_source",
    "date_status",
    "filename_date",
    "explanation",
    "ocr_chars",
    "pdf_count",
    "processing_seconds",
    "error",
    "approved",
]


def main():
    processed = 0

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=COLUMNS,
        )
        writer.writeheader()
        file.flush()

        for note in joplin_pages(
            "/notes",
            "id,title",
        ):
            if not GENERIC_TITLE.search(note["title"]):
                continue

            try:
                ocr_text, filenames, pdf_count = (
                    get_pdf_content(note["id"])
                )
            except requests.RequestException as exc:
                print(f"Joplin error: {exc}")
                continue

            if pdf_count == 0:
                continue

            print(f"\nProcessing: {note['title']}")

            start = time.perf_counter()
            suggestion = {}
            error = ""

            try:
                if ocr_text.strip():
                    suggestion = propose_title(
                        note["title"],
                        filenames,
                        ocr_text,
                    )

            except (requests.RequestException,
                    ValueError, KeyError, TypeError) as exc:
                error = str(exc)
                print(f"AI error: {error}")

            # Filename-only fallback still works
            # if OCR is absent or AI fails.
            result = build_title(
                suggestion, ocr_text, filenames
            )

            elapsed = time.perf_counter() - start

            row = {
                "note_id": note["id"],
                "old_title": note["title"],
                "proposed_title": result["title"],
                "filenames": "; ".join(filenames),
                "organization": suggestion.get(
                    "organization", ""
                ),
                "document_type": suggestion.get(
                    "document_type", ""
                ),
                "ocr_date_proposed": suggestion.get(
                    "document_date", ""
                ),
                "date_evidence": suggestion.get(
                    "date_evidence", ""
                ),
                "date_used": result["date_used"],
                "date_source": result["date_source"],
                "date_status": result["date_status"],
                "filename_date": result["filename_date"],
                "explanation": suggestion.get(
                    "explanation", ""
                ),
                "ocr_chars": len(ocr_text),
                "pdf_count": pdf_count,
                "processing_seconds": round(elapsed, 2),
                "error": error,
                "approved": "",
            }

            writer.writerow(row)
            file.flush()

            print(f"OLD: {row['old_title']}")
            print(
                f"NEW: {row['proposed_title'] or '[No title]'}"
            )
            print(f"DATE SOURCE: {row['date_source']}")
            print(f"OCR DATE CHECK: {row['date_status']}")
            print(f"OCR LENGTH: {row['ocr_chars']}")
            print(f"TIME: {elapsed:.1f}s")

            processed += 1

            if processed >= MAX_RESULTS:
                break

    print(f"\nProcessed {processed} documents.")
    print(f"Saved: {OUTPUT_CSV}")
    print("No Joplin notes were modified.")


if __name__ == "__main__":
    main()
