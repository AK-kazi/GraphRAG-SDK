#!/usr/bin/env python3
"""
Test script to verify the JSON repair functionality with the problematic JSON from the error.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from graphrag_sdk.helpers import extract_json, validate_json_with_details

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