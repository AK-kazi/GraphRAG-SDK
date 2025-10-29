#!/usr/bin/env python3
"""
Standalone test script to verify the JSON repair functionality with the problematic JSON from the error.
This script directly imports only the necessary functions to avoid dependency issues.
"""

import sys
import os
import json
import re
import logging
from typing import Union, Optional, Tuple

# Add the current directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Import the libraries we need
try:
    from fix_busted_json import repair_json
except ImportError:
    print("fix_busted_json not available, installing...")
    os.system("pip install fix-busted-json")
    from fix_busted_json import repair_json

try:
    from json_repair import repair_json as json_repair_repair
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False
    print("json_repair library not available. Install with: pip install json-repair")

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def validate_json_with_details(json_text: str) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Validates JSON and provides detailed error information.
    
    Args:
        json_text (str): The JSON string to validate.
        
    Returns:
        Tuple[bool, Optional[str], Optional[int]]: A tuple containing:
            - bool: True if valid, False otherwise
            - Optional[str]: Error message if invalid, None otherwise
            - Optional[int]: Line number where error occurred if available, None otherwise
    """
    try:
        json.loads(json_text)
        return True, None, None
    except json.JSONDecodeError as e:
        # Extract line number from error message if available
        error_msg = str(e)
        line_num = None
        if "line" in error_msg.lower():
            try:
                # Extract line number from error message like "Expecting ':' delimiter: line 1 column 123"
                match = re.search(r'line (\d+)', error_msg.lower())
                if match:
                    line_num = int(match.group(1))
            except (ValueError, AttributeError):
                pass
        
        return False, error_msg, line_num
    except Exception as e:
        return False, str(e), None

def _apply_targeted_repairs(json_text: str) -> str:
    """Apply targeted repairs for common LLM JSON errors."""
    # Fix missing colons after quoted keys (e.g., "label""value" -> "label": "value")
    # This specifically targets the pattern: "key""value"
    json_text = re.sub(r'("([^"]+)")(")', r'\1: \3', json_text)
    
    # Fix single quotes around keys and values
    json_text = re.sub(r"'([^']+)':\s*'([^']*)'", r'"\1": "\2"', json_text)
    
    # Fix missing commas between objects
    json_text = re.sub(r'}\s*{', '}, {', json_text)
    
    # Fix trailing commas
    json_text = re.sub(r',\s*([}\]])', r'\1', json_text)
    
    return json_text

def _basic_json_cleanup(json_text: str) -> str:
    """Basic cleanup for common JSON syntax issues."""
    # Remove any leading/trailing non-JSON content
    json_text = json_text.strip()
    
    # Ensure proper JSON structure
    if not json_text.startswith('{') and not json_text.startswith('['):
        # Try to find the first { or [
        start_idx = min(
            json_text.find('{') if '{' in json_text else len(json_text),
            json_text.find('[') if '[' in json_text else len(json_text)
        )
        if start_idx < len(json_text):
            json_text = json_text[start_idx:]
    
    return json_text

def extract_json(text: Union[str, dict], skip_repair: Optional[bool] = False) -> str:
    """
    Extracts JSON from a string or dictionary, optionally skipping JSON repair.
    
    Args:
        text (Union[str, dict]): The input text or dictionary.
        skip_repair (Optional[bool]): Flag to skip JSON repair. Defaults to False.
        
    Returns:
        str: The extracted JSON as a string.
    """
    if not isinstance(text, str):
        text = str(text)
    regex = r"(?:```)?(?:json)?([^`]*)(?:\\n)?(?:```)?"
    matches = re.findall(regex, text, re.DOTALL)
    json_text = "".join(matches)

    if skip_repair:
        return json_text

    # First validate the original JSON
    is_valid, error_msg, line_num = validate_json_with_details(json_text)
    if is_valid:
        return json_text
    
    logger.debug(f"Invalid JSON detected: {error_msg}" + (f" at line {line_num}" if line_num else ""))

    # Try repair strategies in sequence
    repair_attempts = [
        ("fix_busted_json", lambda: repair_json(json_text)),
    ]
    
    # Add json_repair if available
    if JSON_REPAIR_AVAILABLE:
        repair_attempts.append(("json_repair", lambda: json_repair_repair(json_text)))
    
    # Add our custom repair strategies
    repair_attempts.extend([
        ("targeted_regex", lambda: _apply_targeted_repairs(json_text)),
        ("basic_cleanup", lambda: _basic_json_cleanup(json_text))
    ])
    
    for strategy_name, repair_func in repair_attempts:
        try:
            repaired = repair_func()
            # Validate the repaired JSON
            is_valid, error_msg, line_num = validate_json_with_details(repaired)
            if is_valid:
                logger.debug(f"Successfully repaired JSON using {strategy_name}")
                return repaired
            else:
                logger.debug(f"Repair strategy {strategy_name} produced invalid JSON: {error_msg}" + (f" at line {line_num}" if line_num else ""))
        except Exception as e:
            logger.debug(f"Repair strategy {strategy_name} failed: {e}")
            continue
    
    # All strategies failed
    logger.error(f"All JSON repair strategies failed. Original error: {error_msg}" + (f" at line {line_num}" if line_num else ""))
    return json_text

# The problematic JSON from the error message
problematic_json = """{"entities":[{"label":"Issue","attributes":{"id":"3371","title":"SDWAN Instance Power Options Visibility","status":"Resolved","priority":"P3","created_at":"2023-12-28T19:32:28Z","description":"Users were unable to see power on/off options for SDWAN instances in the Alkira Portal.","resolution":"The power off/on options for SDWAN instances eventually became visible to the user. The exact cause for the initial absence and subsequent appearance was not definitively identified, but it was suspected to be related to either a UI rendering issue (e.g., not scrolling down to reveal the options) or a delayed propagation of user permissions."}},{"label":"Person","attributes":{"name":"Deepak Muku"}},{"label":"Tenant","attributes":{"name":"splunk"}},{"label":"Service","attributes":{"name":"SDWAN instances"}},{"label":"Document","attributes":{"name":"Troubleshooting Dashboard","description":"Write permission for Troubleshooting Dashboard"}},{"label":"Role","attributes":{"name":"operator"}},{"label":"Role","attributes":{"name":"Netadmin Role"}},{"label":"Role","attributes":{"name":"sg-okta-alkira-netadmin"}},{"label":"Action","attributes":{"name":"inspect browser developer tools"}},{"label":"Action","attributes":{"name":"inspect API responses"}},{"label":"Action","attributes":{"name":"Verify User Role Permissions"}},{"label":"Action","attributes":{"name":"Inspect Browser Network Traffic for Permissions"}},{"label":"Action","attributes":{"name":"Inspect Browser Network Traffic for Instance State"}},{"label":"Document","attributes":{"url":"https://alkiranet.slack.com/archives/C032ENF5ZLH/p1703715791026759","type":"Slack thread"}},{"label":"Document","attributes":{"url":"https://status.alkira.com/","name":"Alkira Status Dashboard"}},{"label":"Document","attributes":{"url":"https://www.alkira.com/design-zone-videos/","name":"Alkira Design Zone Videos"}}],"relations":[{"label":"REPORTED_BY","source":{"label":"Issue","attributes":{"id":"3371"}},"target":{"label":"Person","attributes":{"name":"Deepak Muku"}},"attributes":{}},{"label":"AFFECTS","source":{"label":"Issue","attributes":{"id":"3371"}},"target":{"label":"Tenant","attributes":{"name":" splunk"}},"attributes":{}},{"label":"AFFECTS","source":{"label':"Issue","attributes":{"id":"3371"}},"target":{"label":"Service","attributes":{"name":"SDWAN instances"}},"attributes":{}},{"label":"HAS_ACTION","source":{"label":"Issue","attributes":{"id":"3371"}},"target":{"label":"Action","attributes":{"name":"Verify User Role Permissions"}},"attributes":{}},{"label":"HAS_ACTION","source":{"label":"Issue","attributes":{"id":"3371"}},"target":{"label":"Action","attributes":{"name":"Inspect Browser Network Traffic for Permissions"}},"attributes":{}},{"label":"HAS_ACTION","source":{"label":"Issue","attributes":{"id":"3371"}},"target":{"label":"Action","attributes":{"name":"Inspect Browser Network Traffic for Instance State"}},"attributes":{}},{"label":"HAS_POLICY","source":{"label":"Role","attributes":{"name":"operator"}},"target":{"label":"Document","attributes":{"name":"Troubleshooting Dashboard","description":"Write permission for Troubleshooting Dashboard"}},"attributes":{"effect":"lack"}},{"label":"HAS_POLICY","source":{"label":"Role","attributes":{"name":"sg-okta-alkira-netadmin"}},"target":{"label":"Document","attributes":{"name":"Troubleshooting Dashboard","description":"Write permission for Troubleshooting Dashboard"}},"attributes":{"effect":"has"}},{"label":"RELATED_TO","source":{"label":"Issue","attributes":{"id":"3371"}},"target":{"label":"Document","attributes":{"url":"https://alkiranet.slack.com/archives/C032ENF5ZLH/p1703715791026759","type":"Slack thread"}},"attributes":{}},{"label":"RELATED_TO","source":{"label":"Issue","attributes":{"id":"3371"}},"target":{"label":"Document","attributes":{"url":"https://status.alkira.com/","name":"Alkira Status Dashboard"}},"attributes":{}},{"label":"RELATED_TO","source":{"label":"Issue","attributes":{"id":"3371"}},"target":{"label":"Document","attributes":{"url":"https://www.alkira.com/design-zone-videos/","name":"Alkira Design Zone Videos"}},"attributes":{}},{"label":"USES","source":{"label":"Person","attributes":{"name":"Deepak Muku"}},"target":{"label":"Action","attributes":{"name":"inspect browser developer tools"}},"attributes":{}},{"label":"USES","source":{"label":"Person","attributes":{"name":"Deepak Muku"}},"target":{"label":"Action","attributes":{"name":"inspect API responses"}},"attributes":{}},{"label":"CAUSED_BY","source":{"label":"Issue","attributes":{"id":"3371"}},"target":{"label":"Action","attributes":{"name":"inspect browser developer tools"}},"attributes":{}},{"label":"CAUSED_BY","source":{"label":"Issue","attributes":{"id":"3371"}},"target":{"label":"Action","attributes":{"name":"inspect API responses"}},"attributes":{}},{"label":"AFFECTS","source":{"label":"Issue","attributes":{"id":"3371"}},"target":{"label":"Role","attributes":{"name":"sg-okta-alkira-netadmin"}},"attributes":{"description":"expected to have permission"}},{"label":"HAS_ACTION","source":{"label":"Issue","attributes":{"id":"3371"}},"target":{"label":"Action","attributes":{"name":"Inspect Browser Network Traffic for Permissions"}},"attributes":{"description":"Filter for \"Permissions\""}},{"label":"HAS_ACTION","source":{"label":"Issue","attributes":{"id":"3371"}},"target":{"label":"Action","attributes":{"name":"Inspect Browser Network Traffic for Instance State"}},"attributes":{"description":"Filter for \"instance-state\""}},{"label":"MANAGES","source":{"label":"Role","attributes":{"name":"sg-okta-alkira-netadmin"}},"target":{"label":"Issue","attributes":{"id":"3371"}},"attributes":{}}]}"""

def test_json_validation():
    """Test the JSON validation with the problematic JSON."""
    print("Testing JSON validation with problematic JSON...")
    
    is_valid, error_msg, line_num = validate_json_with_details(problematic_json)
    
    print(f"Is valid: {is_valid}")
    if not is_valid:
        print(f"Error: {error_msg}")
        if line_num:
            print(f"Error at line: {line_num}")
    
    return is_valid

def test_json_repair():
    """Test the JSON repair functionality."""
    print("\nTesting JSON repair...")
    
    try:
        repaired_json = extract_json(problematic_json)
        
        # Check if the repaired JSON is valid
        is_valid, error_msg, line_num = validate_json_with_details(repaired_json)
        
        print(f"Repair successful: {is_valid}")
        if not is_valid:
            print(f"Repair failed with error: {error_msg}")
            if line_num:
                print(f"Error at line: {line_num}")
        else:
            print("JSON was successfully repaired!")
            # Optionally print a snippet of the repaired JSON
            print(f"Repaired JSON (first 200 chars): {repaired_json[:200]}...")
        
        return is_valid
    except Exception as e:
        print(f"Exception during repair: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("JSON Repair Test")
    print("=" * 60)
    
    # Test validation first
    is_valid = test_json_validation()
    
    if not is_valid:
        # Test repair if validation failed
        repair_success = test_json_repair()
        
        if repair_success:
            print("\n✅ Test passed: JSON was successfully repaired!")
            sys.exit(0)
        else:
            print("\n❌ Test failed: JSON repair was unsuccessful.")
            sys.exit(1)
    else:
        print("\n✅ Test passed: JSON was already valid!")
        sys.exit(0)