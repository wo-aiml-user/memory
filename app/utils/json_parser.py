"""
JSON Parser
Safe utilities for parsing JSON from LLM output.
"""

import logging
import json
import re
from typing import Any, Optional

logger = logging.getLogger("memory_chat.parser")


def safe_json_parse(text: str) -> Optional[Any]:
    """
    Safely parse JSON from text, handling common LLM output issues.
    
    Args:
        text: Text that may contain JSON
    
    Returns:
        Parsed JSON object or None if parsing fails
    """
    try:
        # First, try direct parsing
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON object pattern
    obj_match = re.search(r'\{[\s\S]*\}', text)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Try to find JSON array pattern
    arr_match = re.search(r'\[[\s\S]*\]', text)
    if arr_match:
        try:
            return json.loads(arr_match.group(0))
        except json.JSONDecodeError:
            pass
    
    logger.debug(f"Failed to parse JSON from text: {text[:100]}...")
    return None


def extract_tool_arguments(text: str) -> Optional[dict]:
    """
    Extract tool arguments from LLM text output.
    
    Args:
        text: Text that may contain tool arguments
    
    Returns:
        Dict of arguments or None
    """
    result = safe_json_parse(text)
    if isinstance(result, dict):
        return result
    return None


def validate_json_structure(data: Any, required_keys: list[str]) -> bool:
    """
    Validate that a JSON object has required keys.
    
    Args:
        data: JSON data to validate
        required_keys: List of required keys
    
    Returns:
        True if all required keys are present
    """
    if not isinstance(data, dict):
        return False
    return all(key in data for key in required_keys)


def serialize_for_llm(data: Any, indent: int = 2) -> str:
    """
    Serialize data to JSON for LLM consumption.
    
    Args:
        data: Data to serialize
        indent: Indentation level
    
    Returns:
        JSON string
    """
    try:
        return json.dumps(data, indent=indent, ensure_ascii=False, default=str)
    except Exception as e:
        logger.exception(f"Error serializing data: {e}")
        return str(data)
