# AI Invoice Extractor

A document intelligence pipeline that extracts structured data from invoice files using OCR and large language models. Built by Godwin.

Live Demo: https://invoiceai-blue.vercel.app
Backend API: https://invoice-ai-backend-1.onrender.com/docs
Frontend Repo: https://github.com/Zadetechd/invoice.ai
Backend Repo: https://github.com/Zadetechd/invoice.ai-backend

> The backend is hosted on Render's free tier and may take up to 30 seconds to respond on the first request after a period of inactivity. This is a known free tier behaviour and not a bug.

---

## What It Does

Upload any PDF or image invoice. The pipeline extracts vendor name, invoice number, date, line items, totals, and confidence score then returns clean structured JSON in seconds. Both single and batch uploads are supported with CSV and JSON export.

---

## Architecture

```
Invoice Upload
     ↓
File Validation
     ↓
OCR Processing       pdfplumber for native PDFs, pytesseract for scanned files and images
     ↓
Text Preprocessing   whitespace normalisation, deduplication, truncation to 2000 chars
     ↓
LLM Extraction       Google Gemini 1.5 Flash, swappable to OpenAI via one env variable
     ↓
Schema Validation    Pydantic v2
     ↓
Confidence Scoring   field completeness weighting, 0.0 to 1.0
     ↓
Structured JSON Output
```

---

## Project Structure

```
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
├── runtime.txt
├── .env.example
└── README.md
```

---

## Quick Start

**1. Clone and install dependencies**

```bash
git clone https://github.com/Zadetechd/invoice.ai-backend
cd invoice.ai-backend
pip install -r requirements.txt
```

Install Tesseract for OCR support:

Ubuntu or Debian:
```bash
sudo apt install tesseract-ocr poppler-utils
```

macOS:
```bash
brew install tesseract poppler
```

Windows: download the installer from https://github.com/UB-Mannheim/tesseract/wiki

**2. Set up your API key**

```bash
cp .env.example .env
```

Open `.env` and add your Gemini API key. Get a free key at https://aistudio.google.com/app/apikey with no credit card required.

```
GEMINI_API_KEY=your_key_here
```

**3. Start the server**

```bash
python -m uvicorn app.main:app --reload
```

The API runs at http://127.0.0.1:8000 and the interactive docs are at http://127.0.0.1:8000/docs

---

## Testing With the Sample Invoices

Two invoices are included in the `sample_invoices/` folder so you can test without preparing your own files.

**Using the live demo**

Go to https://invoiceai-blue.vercel.app, upload either sample invoice, and the full extraction result appears on screen with a confidence score and line items table.

**Using the interactive API docs**

1. Open https://invoice-ai-backend-1.onrender.com/docs
2. Click POST /upload then Try it out
3. Upload `sample_invoices/invoice_techstore.pdf`
4. Click Execute

**Using curl**

```bash
curl -X POST https://invoice-ai-backend-1.onrender.com/upload \
  -F "file=@sample_invoices/invoice_techstore.pdf"
```

**Using Python**

```python
import requests

with open("sample_invoices/invoice_techstore.pdf", "rb") as f:
    response = requests.post(
        "https://invoice-ai-backend-1.onrender.com/upload",
        files={"file": f}
    )

print(response.json())
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /upload | Extract data from one invoice |
| POST | /upload-batch | Upload multiple invoices, returns job_id |
| GET | /status/{job_id} | Poll batch job status and get results |
| GET | /export/json | Download last result as JSON |
| GET | /export/csv | Download last result as CSV |
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

The system uses an abstraction layer so swapping providers requires changing one environment variable.

To use OpenAI instead of Gemini set these in your `.env`:

```
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

To add a new provider such as Anthropic Claude, Mistral, or a local vLLM endpoint, create a new class in `app/llm/` that extends `BaseLLMProvider`, implement the `extract` and `health_check` methods, then register it in `app/llm/factory.py`.

---

## Confidence Scoring

Each result includes a `confidence_score` between 0.0 and 1.0 based on field completeness weighting.

| Score | Status | Meaning |
|-------|--------|---------|
| 0.75 to 1.0 | success | All core fields extracted reliably |
| 0.40 to 0.74 | partial | Some fields missing or uncertain |
| 0.00 to 0.39 | failed | Extraction did not produce usable data |

Fields like `total_amount`, `vendor_name`, and `invoice_date` carry higher weight than supplementary fields like `payment_terms` or `notes`.

---





The hosted demo accepts PDF files only. 
To process scanned images run the project locally 
with Tesseract installed via the instructions above.

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 |
| API Framework | FastAPI |
| LLM default | Google Gemini 1.5 Flash |
| LLM alternative | OpenAI GPT-4o-mini |
| OCR | pytesseract and pdf2image |
| PDF parsing | pdfplumber |
| Validation | Pydantic v2 |
| Export | Pandas |
| Server | Uvicorn |
| Frontend | React TypeScript with Vite |

---

Built by Godwin as a document intelligence portfolio project.