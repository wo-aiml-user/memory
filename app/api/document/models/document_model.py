from typing import Optional, ClassVar, Self
from pydantic import BaseModel, field_validator

class DocumentRequest(BaseModel):
    """Simple document request model for PDF/DOCX uploads"""
    file_id: str
    file_name: str
    file_path: str
    file_ext: str

    allowed_file_exts: ClassVar[set[str]] = {"doc", "docx", "pdf"}

    @field_validator("file_id", "file_name", "file_path", "file_ext")
    @classmethod
    def check_required_fields(cls, v: str, info) -> str:
        if not v.strip():
            raise ValueError(f"The '{info.field_name}' field cannot be empty.")
        return v

    @field_validator("file_ext")
    @classmethod
    def check_file_ext(cls, v: str) -> str:
        ext = v.lower().strip()
        if ext not in cls.allowed_file_exts:
            allowed = ", ".join(sorted(cls.allowed_file_exts))
            raise ValueError(f"Invalid file type. Allowed types are: {allowed}.")
        return ext