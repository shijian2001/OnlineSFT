"""JSON parser utility for extracting JSON from LLM responses."""

import json
import re
from typing import Union, Dict, Any


class JSONParser:
    """Utility class for parsing and extracting JSON from text responses.
    
    This parser intelligently handles both JSON and non-JSON responses:
    - If response contains valid JSON, returns the parsed dictionary
    - If response is not JSON, returns the original string unchanged
    """

    @staticmethod
    def parse(response: str) -> Union[str, Dict[str, Any]]:
        """
        Extract and parse JSON from a response string, or return original string if not JSON.

        Args:
            response: String potentially containing JSON

        Returns:
            - Parsed JSON dictionary if response contains valid JSON
            - Original string if response is not JSON format
            - Original dict if input is already a dict
        
        Examples:
            >>> JSONParser.parse('{"key": "value"}')
            {'key': 'value'}
            
            >>> JSONParser.parse('```json\\n{"key": "value"}\\n```')
            {'key': 'value'}
            
            >>> JSONParser.parse('This is plain text')
            'This is plain text'
        """
        # If input is already a dict, return it
        if isinstance(response, dict):
            return response
        
        if not isinstance(response, str):
            return response
        
        # Try to match JSON in a code block first (supports ```json and ```)
        json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON object or array in the string
            json_match = re.search(r'(\{.*?\}|\[.*?\])', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response
        
        try:
            parsed_data = json.loads(json_str)
            # Return parsed JSON (dict or list)
            return parsed_data
        except json.JSONDecodeError:
            # Not valid JSON, return original response
            return response