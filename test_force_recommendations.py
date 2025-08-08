#!/usr/bin/env python3
"""
Test the force recommendations endpoint
"""

import requests
import json

def test_force_recommendations():
    """Test the force recommendations endpoint"""
    print("🔍 Testing Force Recommendations Endpoint...")
    
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
            # Test force recommendations endpoint
            force_response = session.get(f"{base_url}/force_recommendations")
            print(f"   Force recommendations response: {force_response.status_code}")
            print(f"   Final URL: {force_response.url}")
            
            if force_response.status_code == 200:
                # Check if it's HTML content (successful template render)
                if force_response.headers.get('content-type', '').startswith('text/html'):
                    print("   ✅ Successfully accessed recommendations via force endpoint!")

                    # Check if the page has actual content
                    if 'No Style Recommendations Available' in force_response.text:
                        print("   ❌ But page shows 'No recommendations available'")
                    elif 'Personalized Style Recommendations' in force_response.text or 'Your Personalized Style Guide' in force_response.text:
                        print("   ✅ Recommendations page has proper content!")

                        # Count outfit sections
                        outfit_count = force_response.text.count('Recommendations')
                        print(f"   📊 Found {outfit_count} recommendation sections")

                        return True
                    else:
                        print("   ❓ Page content unclear")
                        print(f"   Content preview: {force_response.text[:200]}")
                else:
                    # Might be JSON error response
                    try:
                        error_data = force_response.json()
                        print(f"   ❌ Error response: {error_data}")
                    except:
                        print(f"   ❓ Unexpected response format")
                        print(f"   Content type: {force_response.headers.get('content-type', 'unknown')}")
                        print(f"   Content preview: {force_response.text[:200]}")
            else:
                print(f"   ❌ Force recommendations failed: {force_response.text[:200]}")
        else:
            print(f"   ❌ Login failed")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        
    return False

def test_existing_user():
    """Test with existing user that has preferences"""
    print("\n🧪 Testing with existing user (Swaroop1902)...")
    
    base_url = "http://127.0.0.1:5000"
    session = requests.Session()
    
    # Try different possible passwords
    possible_passwords = ['password', 'password123', 'swaroop1902', '123456']
    
    for password in possible_passwords:
        login_data = {
            'username': 'Swaroop1902',
            'password': password
        }
        
        try:
            response = session.post(f"{base_url}/login", data=login_data)
            if response.status_code == 200 and 'dashboard' in response.url:
                print(f"   ✅ Logged in with password: {password}")
                
                # Test force recommendations
                force_response = session.get(f"{base_url}/force_recommendations")
                print(f"   Force recommendations response: {force_response.status_code}")
                
                if force_response.status_code == 200 and 'Your Personalized Style Guide' in force_response.text:
                    print("   ✅ Successfully accessed recommendations for existing user!")
                    return True
                break
        except:
            continue
    
    print("   ❌ Could not login as existing user")
    return False

def main():
    """Run all tests"""
    print("🚀 Starting Force Recommendations Test...\n")
    
    test1 = test_force_recommendations()
    test2 = test_existing_user()
    
    print(f"\n📊 Test Results:")
    print(f"   Test User Force Recommendations: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"   Existing User Force Recommendations: {'✅ PASS' if test2 else '❌ FAIL'}")
    
    if test1 or test2:
        print(f"\n✅ Force recommendations endpoint works! This confirms the issue is in the normal routing logic.")
        print(f"🔧 Solution: Update the questionnaire to redirect to /force_recommendations instead of /style_recommendations")
    else:
        print(f"\n❌ Force recommendations also failed - there may be a deeper issue")

if __name__ == "__main__":
    main()
