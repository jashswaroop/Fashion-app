#!/usr/bin/env python3
"""
Test the debug endpoint to see user state
"""

import requests
import json

def test_debug_endpoint():
    """Test the debug endpoint"""
    print("🔍 Testing Debug Endpoint...")
    
    base_url = "http://127.0.0.1:5000"
    session = requests.Session()
    
    # Login first
    login_data = {
        'username': 'testuser_debug',
        'password': 'testpass123'
    }
    
    try:
        response = session.post(f"{base_url}/login", data=login_data)
        print(f"   Login response: {response.status_code}")
        
        if response.status_code == 200:
            # Test debug endpoint
            debug_response = session.get(f"{base_url}/debug_user")
            print(f"   Debug response: {debug_response.status_code}")
            
            if debug_response.status_code == 200:
                user_info = debug_response.json()
                print(f"   User info: {json.dumps(user_info, indent=2)}")
                
                # Test recommendations access
                rec_response = session.get(f"{base_url}/style_recommendations")
                print(f"   Recommendations response: {rec_response.status_code}")
                print(f"   Final URL: {rec_response.url}")
                
            else:
                print(f"   Debug endpoint failed: {debug_response.text}")
        else:
            print(f"   Login failed")
            
    except Exception as e:
        print(f"   Error: {e}")

if __name__ == "__main__":
    test_debug_endpoint()
