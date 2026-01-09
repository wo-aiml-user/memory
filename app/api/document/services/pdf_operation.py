from bs4 import BeautifulSoup
import os
import subprocess
from tempfile import NamedTemporaryFile
from typing import BinaryIO
from fastapi import HTTPException, status
from typing import Tuple
from tika import parser
from loguru import logger
import platform

def convert_docx_to_pdf_and_return_buffer(file_buffer: BinaryIO, file_extension: str) -> bytes:
    if not file_extension.startswith('.'):
        file_extension = f".{file_extension}"

    # Step 1: Create a temporary document file
    try:
        with NamedTemporaryFile(delete=False, suffix=file_extension) as temp_doc:
            temp_doc.write(file_buffer.read())
            temp_doc_path = temp_doc.name
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="There was a problem processing the uploaded file. Please try again."
        )

    pdf_file_path = temp_doc_path.replace(file_extension, '.pdf')

    try:
        system_platform = platform.system().lower()

        if system_platform == 'windows':
            # Use docx2pdf for Windows
            from docx2pdf import convert

            # Create a temp directory for output
            output_dir = os.path.dirname(pdf_file_path)
            convert(temp_doc_path, output_dir)

        else:
            # Use LibreOffice for Linux/macOS
            command = [
                "libreoffice",
                "--headless",
                "--convert-to", "pdf",
                temp_doc_path,
                "--outdir", os.path.dirname(pdf_file_path)
            ]

            process = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True
            )

            if process.returncode != 0 or not os.path.exists(pdf_file_path):
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to convert document to PDF. Please ensure the file is valid."
                )

        # Step 3: Read and validate the generated PDF file
        try:
            with open(pdf_file_path, 'rb') as pdf_file:
                pdf_buffer = pdf_file.read()

            if not pdf_buffer:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="The converted PDF file is empty. Please check the input document."
                )

            return pdf_buffer

        except HTTPException:
            raise  # Re-raise known user-friendly errors
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="PDF was created but couldn't be read. Please try again."
            )

    except subprocess.CalledProcessError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during document conversion. Please try again."
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while converting the document. Please try again."
        )
    finally:
        # Step 4: Clean up temporary files
        if os.path.exists(temp_doc_path):
            os.remove(temp_doc_path)
        if os.path.exists(pdf_file_path):
            os.remove(pdf_file_path)

def extract_page_texts(xhtml_content):
    soup = BeautifulSoup(xhtml_content, 'html.parser')
    page_divs = soup.find_all('div', attrs={'class': 'page'})
    
    page_texts = []
    if page_divs:
        for page_div in page_divs:
            text = page_div.get_text(separator=' ', strip=True)
            page_texts.append(text)
    else:
        text = soup.get_text(separator=' ', strip=True)
        page_texts.append(text)
    return page_texts

def process_pdf(file_id: str, byte_data: bytes) -> Tuple[str, int]:
    logger.info(f"Processing PDF for file_id: {file_id}")
    headers = {
        'Accept': 'application/json',
        "X-Tika-PDFocrStrategy": "auto"
    }

    try:
        file_data = parser.from_buffer(
            byte_data,
            xmlContent=True,
            requestOptions={'headers': headers, 'timeout': 900}
        )
    except Exception as e:
        logger.error(f"Error processing file_id {file_id}: {e}")
        if any(msg in str(e) for msg in [
            "Connection aborted",
            "Remote end closed connection without response",
            "Read timed out"
        ]):
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="File processing failed. The file might be too large or complex to process.")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Unexpected error during file processing.")

    xhtml_content = file_data.get('content', '')
    metadata = file_data.get('metadata', {})

    # Check for encrypted document
    container_exception = metadata.get('X-TIKA:EXCEPTION:container_exception', '')
    if container_exception and any(err in container_exception for err in ['EncryptedDocumentException', 'CryptographyException']):
        logger.error("Upload failed: Password-protected files are not supported.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload failed: Password-protected files are not supported.")

    # Check for corrupted file
    if metadata.get('X-TIKA:Parsed-By') == 'org.apache.tika.parser.EmptyParser':
        logger.error("Upload failed: Invalid file format.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload failed: Invalid file format. Please try again with a valid file.")

    num_pages = int(metadata.get('xmpTPg:NPages', 1))
    logger.info(f"Document has {num_pages} pages")

    # Extract and validate text
    page_texts = extract_page_texts(xhtml_content)
    total_text = ' '.join(page_texts).strip()
    text_length = len(total_text)
    logger.info(f"Extracted text length: {text_length}")

    if not total_text:
        logger.error("Upload failed: File appears to be empty.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload failed: The file appears to be empty or could not be processed.")

    if text_length < 100:
        logger.error("Upload failed: Not enough text to process.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload failed: The file does not contain enough text content to process.")

    return page_texts