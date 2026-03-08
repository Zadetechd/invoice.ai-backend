"""
Invoice Data Schemas

Defines every Pydantic model used across the pipeline.
Validation is automatic when data is loaded into these classes.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class LineItem(BaseModel):
    """One product or service line on an invoice."""
    item: str = Field(..., description="Name or description of the item")
    quantity: Optional[float] = Field(None, description="Number of units")
    unit_price: Optional[float] = Field(None, description="Price per unit")
    price: float = Field(..., description="Total price for this line")


class InvoiceData(BaseModel):
    """
    Structured output produced by the LLM extraction step.
    All fields are optional because real invoices vary widely.
    The confidence scorer penalises missing core fields.
    """
    vendor_name: Optional[str] = Field(None)
    invoice_number: Optional[str] = Field(None)
    invoice_date: Optional[str] = Field(None, description="ISO format preferred")
    due_date: Optional[str] = Field(None)
    currency: Optional[str] = Field(None, description="e.g. USD, GHS, NGN")
    subtotal: Optional[float] = Field(None)
    tax_amount: Optional[float] = Field(None)
    total_amount: Optional[float] = Field(None)
    line_items: Optional[List[LineItem]] = Field(default_factory=list)
    bill_to: Optional[str] = Field(None, description="Customer name or company")
    payment_terms: Optional[str] = Field(None, description="e.g. Net 30")
    notes: Optional[str] = Field(None)

    @field_validator("currency")
    @classmethod
    def normalise_currency(cls, v):
        if v:
            return v.strip().upper()
        return v

    @field_validator("total_amount", "subtotal", "tax_amount")
    @classmethod
    def non_negative_amounts(cls, v):
        if v is not None and v < 0:
            raise ValueError("Monetary amounts must be zero or greater")
        return v


class InvoiceExtractionResult(BaseModel):
    """Full result for one invoice file including pipeline metadata."""
    file_name: str
    status: str = Field(..., description="success | partial | failed")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    data: Optional[InvoiceData] = None
    error: Optional[str] = None
    ocr_used: bool = False
    raw_text_length: Optional[int] = None


class BatchExtractionResult(BaseModel):
    """Summary result for a batch upload."""
    total_files: int
    successful: int
    failed: int
    results: List[InvoiceExtractionResult]


class JobStatus(BaseModel):
    """Status response for async batch jobs."""
    job_id: str
    status: str = Field(..., description="pending | processing | completed | failed")
    total_files: int
    processed_files: int
    result: Optional[BatchExtractionResult] = None
