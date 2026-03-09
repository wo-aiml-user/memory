from app.utils.response import error_response, success_response
from fastapi import APIRouter
from datetime import timedelta
from app.api.auth.token import JWTAuth
from app.api.auth.auth_model import TokenRequest, TokenResponse
from loguru import logger

router = APIRouter()

@router.post("/token")
async def create_token(request: TokenRequest):
    """
    Create a new JWT token
    
    Parameters:
    - user_id: string - The user ID to encode in the token (3-50 chars, alphanumeric)
    
    Returns:
    - access_token: The generated JWT token
    """
    try:
        logger.info(f"Creating token for user_id: {request.user_id}")
        
        token = JWTAuth.create_token(
            {"user_id": request.user_id},
            expires_delta=timedelta(days=1)
        )
        
        logger.info(f"Token created successfully for user_id: {request.user_id}")
        
        return success_response({
            "access_token": token
        }, 200)
        
    except Exception as e:
        logger.error(f"Error creating token: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return error_response(f"Error creating token: {str(e)}", 500)