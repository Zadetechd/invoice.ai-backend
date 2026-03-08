# AI Invoice Extractor

A document intelligence pipeline that extracts structured data from invoice files using OCR and large language models.

Upload a PDF or image invoice and receive clean JSON with vendor name, invoice number, date, line items, totals, and a confidence score back in seconds.



## Architecture


Invoice Upload
     ↓
File Validation
     ↓
OCR Processing       ← pdfplumber (native PDF) or pytesseract (scanned / image)
     ↓
Text Preprocessing   ← whitespace normalisation, deduplication, truncation
     ↓
LLM Extraction       ← Gemini 1.5 Flash (or OpenAI GPT via provider switch)
     ↓
Schema Validation    ← Pydantic
     ↓
Confidence Scoring   ← field completeness weighting
     ↓
Structured JSON Output




## Project Structure


ai-invoice-extractor/
├── app/
│   ├── main.py                  FastAPI app entry point
│   ├── config.py                Environment-based settings
│   ├── api/
│   │   └── routes.py            All HTTP endpoints
│   ├── pipeline/
│   │   ├── ocr.py               PDF and image text extraction
│   │   ├── preprocessing.py     Text cleaning before LLM input
│   │   ├── extractor.py         Main pipeline orchestrator
│   │   └── scoring.py           Confidence score calculation
│   ├── schemas/
│   │   └── invoice_schema.py    Pydantic data models
│   ├── services/
│   │   └── batch_processor.py   Async batch job management
│   ├── llm/
│   │   ├── llm_provider.py      Abstract base class
│   │   ├── gemini_client.py     Google Gemini implementation
│   │   ├── openai_client.py     OpenAI implementation
│   │   └── factory.py           Provider loader
│   └── utils/
│       └── file_utils.py        Upload validation and temp storage
├── sample_invoices/             Two ready-to-use test invoices
├── exports/                     JSON and CSV exports land here
├── requirements.txt
├── .env.example
└── README.md


---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd ai-invoice-extractor
pip install -r requirements.txt
```

> **System dependency for OCR:** If you plan to process scanned invoices or images,
> install Tesseract on your machine first.
>
> Ubuntu / Debian:
> ```bash
> sudo apt install tesseract-ocr poppler-utils
> ```
>
> macOS:
> ```bash
> brew install tesseract poppler
> ```
>
> Windows: Download the Tesseract installer from https://github.com/UB-Mannheim/tesseract/wiki

### 2. Set up your API key

```bash
cp .env.example .env
```

Open `.env` and add your Gemini API key:

```
GEMINI_API_KEY=your_key_here
```

Get a free Gemini key at https://aistudio.google.com/app/apikey (no credit card required).

### 3. Start the server

```bash
uvicorn app.main:app --reload
```

The API is now running at http://127.0.0.1:8000

Interactive API docs are at http://127.0.0.1:8000/docs

---

## Testing With the Sample Invoices

Two real-looking invoices are included in the `sample_invoices/` folder so you can test without needing your own files.

### Using the interactive docs (easiest)

1. Open http://127.0.0.1:8000/docs in your browser
2. Click **POST /upload**
3. Click **Try it out**
4. Upload `sample_invoices/invoice_techstore.pdf`
5. Click **Execute**

You will see a full JSON response with all extracted fields and a confidence score.

### Using curl

```bash
# Single invoice
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@sample_invoices/invoice_techstore.pdf"

# Batch upload
curl -X POST http://127.0.0.1:8000/upload-batch \
  -F "files=@sample_invoices/invoice_techstore.pdf" \
  -F "files=@sample_invoices/invoice_cloudservices.pdf"
```

### Using Python

```python
import requests

with open("sample_invoices/invoice_techstore.pdf", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/upload",
        files={"file": f}
    )

print(response.json())
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /upload | Extract data from one invoice |
| POST | /upload-batch | Upload multiple invoices (returns job_id) |
| GET | /status/{job_id} | Poll batch job status and get results |
| GET | /export/json | Download last batch result as JSON |
| GET | /export/csv | Download last batch result as CSV |
| GET | /health | Check API and LLM provider status |

---

## Example Output

```json
{
  "file_name": "invoice_techstore.pdf",
  "status": "success",
  "confidence_score": 0.97,
  "ocr_used": false,
  "raw_text_length": 843,
  "data": {
    "vendor_name": "TechStore Ghana Ltd.",
    "invoice_number": "INV-2024-0091",
    "invoice_date": "2024-11-15",
    "due_date": "2024-12-15",
    "currency": "USD",
    "subtotal": 3026.00,
    "tax_amount": 302.60,
    "total_amount": 3328.60,
    "bill_to": "Dataflow Analytics Inc.",
    "payment_terms": "Net 30",
    "line_items": [
      { "item": "Dell Latitude 5540 Laptop", "quantity": 2, "unit_price": 850.00, "price": 1700.00 },
      { "item": "Samsung 27in Monitor", "quantity": 2, "unit_price": 320.00, "price": 640.00 }
    ]
  }
}
```

---

## Switching LLM Providers

The system uses an abstraction layer so you can swap providers with one environment variable change.

To use OpenAI instead of Gemini:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

To add a new provider (e.g. Anthropic Claude, Mistral, or a local vLLM endpoint), create a new class in `app/llm/` that extends `BaseLLMProvider` and implement the `extract` and `health_check` methods, then register it in `app/llm/factory.py`.

---

## Confidence Scoring

Each extraction result includes a `confidence_score` between 0.0 and 1.0.

| Score Range | Status | Meaning |
|-------------|--------|---------|
| 0.75 to 1.0 | success | All core fields extracted reliably |
| 0.40 to 0.74 | partial | Some fields missing or uncertain |
| 0.00 to 0.39 | failed | Extraction did not produce usable data |

The score is calculated using field completeness weighting. Fields like `total_amount`, `vendor_name`, and `invoice_date` carry higher weight than supplementary fields like `payment_terms` or `notes`.

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| API Framework | FastAPI |
| LLM (default) | Google Gemini 1.5 Flash |
| LLM (alternative) | OpenAI GPT-4o-mini |
| OCR | pytesseract and pdf2image |
| PDF parsing | pdfplumber |
| Validation | Pydantic v2 |
| Export | Pandas |
| Server | Uvicorn |
