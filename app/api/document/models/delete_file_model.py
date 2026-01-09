from pydantic import BaseModel, field_validator


class DeleteFileRequest(BaseModel):
    """Request model for deleting files from vector database"""
    file_id: str

    @field_validator("file_id")
    @classmethod
    def validate_file_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("file_id cannot be empty")
        return v.strip()