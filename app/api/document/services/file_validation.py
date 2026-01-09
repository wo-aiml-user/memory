from loguru import logger
from fastapi import HTTPException, status

from app.api.document.services.file_operation import check_file_exists, update_file_name_in_milvus

async def is_valid_file_id(file_id: str, user_id: str, collection_type: str) -> tuple[bool, str | None]:
    exists = await check_file_exists(user_id, file_id, collection_type)
    if not exists:
        return False, f"file_id '{file_id}' not found in the database"
    return True, None

async def get_valid_file_ids(file_ids: list[str], user_id: str, collection_type: str) -> list[str]:
    valid_ids, errors = [], []

    for file_id in file_ids:
        is_valid, error = await is_valid_file_id(file_id, user_id, collection_type)
        if is_valid:
            valid_ids.append(file_id)
        else:
            errors.append({"file_id": file_id, "error": error})
    
    if not valid_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"message": "No valid file IDs to process", "invalid_files": errors}
        )

    if errors:
        logger.info(f"Some file_ids rejected: {errors}")

    return valid_ids

async def validate_and_update_file_name(data, user_id: str):
    # Check file exists
    file_exists = await check_file_exists(user_id, data.file_id, data.collection_type)
    if not file_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File with ID '{data.file_id}' not found in collection '{data.collection_type}'"
        )

    # Perform the update
    result = await update_file_name_in_milvus(
        user_id=user_id,
        file_id=data.file_id,
        new_file_name=data.new_file_name,
        collection_type=data.collection_type
    )

    if result.get("status") != "success":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("message", "Failed to update file name")
        )

    return result
