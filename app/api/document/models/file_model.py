from fastapi import UploadFile, HTTPException, status
from pydantic import BaseModel, field_validator
from loguru import logger

# Define allowed MIME types for documents
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/msword",  # .doc
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"  # .docx
}


class UploadRequestValidator(BaseModel):
    """Validator for file uploads - PDF and DOCX only"""
    file: UploadFile

    async def validate(self):
        """Validate uploaded file"""
        # Ensure file is not empty
        content = await self.file.read()
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty"
            )

        # Validate content type
        content_type = self.file.content_type or ""
        if not content_type:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not determine file content type"
            )

        if content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed types: PDF, DOC, DOCX"
            )

        filename = self.file.filename or "uploaded_file"

        # Reset file pointer
        self.file.file.seek(0)

        return {
            "filename": filename,
            "buffer": self.file.file,
            "content_type": content_type
        }
