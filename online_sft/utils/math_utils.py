"""Math utility functions."""
import re
from typing import Optional


def normalize_number(num_str: str) -> Optional[float]:
    """
    Normalize a number string to float.
    
    Args:
        num_str: Number string (may include commas, fractions, etc.)
    
    Returns:
        Normalized float or None if parsing fails
    """
    # Remove commas
    num_str = num_str.replace(",", "")
    
    # Handle fractions (e.g., "1/2")
    if "/" in num_str:
        try:
            parts = num_str.split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            pass
    
    # Try direct conversion
    try:
        return float(num_str)
    except ValueError:
        return None


def extract_number(text: str) -> Optional[float]:
    """
    Extract the first number from text.
    
    Args:
        text: Text containing number
    
    Returns:
        Extracted number or None
    """
    # Pattern for numbers (including decimals and negatives)
    pattern = r'-?\d+(?:\.\d+)?'
    match = re.search(pattern, text)
    
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            pass
    
    return None

