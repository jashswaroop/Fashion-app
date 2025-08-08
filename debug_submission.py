#!/usr/bin/env python3
"""
Debug script to check what happens during form submission
"""

import requests
import sqlite3
import json
import time

def check_user_before_and_after():
    """Check user state before and after submission"""
    print("🔍 Debugging Form Submission...")
    
    base_url = "http://127.0.0.1:5000"
    session = requests.Session()
    
    # Login first
    login_data = {
        'username': 'testuser_debug',
        'password': 'testpass123'
    }
    
    session.post(f"{base_url}/login", data=login_data)
    
    # Check user state before submission
    print("\n📊 User state BEFORE submission:")
    conn = sqlite3.connect('instance/fashion_app.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, style_analysis_complete, LENGTH(fashion_preferences) FROM user WHERE username = 'testuser_debug';")
    user_before = cursor.fetchone()
    print(f"   User: {user_before}")
    conn.close()
    
    # Submit form with minimal required data
    print("\n📝 Submitting minimal form data...")
    minimal_data = {
        'lifestyle': 'professional',
        'work_environment': 'corporate',
        'clothing_fit': 'fitted',
        'budget_per_item': '100_200',
        'fashion_risk': 'moderate'
    }
    
    response = session.post(f"{base_url}/submit_fashion_preferences", data=minimal_data)
    print(f"   Response status: {response.status_code}")
    print(f"   Final URL: {response.url}")
    
    # Check user state after submission
    print("\n📊 User state AFTER submission:")
    conn = sqlite3.connect('instance/fashion_app.db')
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, style_analysis_complete, LENGTH(fashion_preferences) FROM user WHERE username = 'testuser_debug';")
    user_after = cursor.fetchone()
    print(f"   User: {user_after}")
    
    # Check if preferences were updated
    if user_after and user_after[3]:  # If fashion_preferences length > 0
        cursor.execute("SELECT fashion_preferences FROM user WHERE username = 'testuser_debug';")
        prefs = cursor.fetchone()[0]
        try:
            parsed_prefs = json.loads(prefs)
            print(f"   Latest preferences keys: {list(parsed_prefs.keys())}")
        except:
            print(f"   Error parsing preferences")
    
    conn.close()
    
    # Try to access recommendations directly
    print("\n🎯 Testing direct recommendations access...")
    rec_response = session.get(f"{base_url}/style_recommendations")
    print(f"   Recommendations response: {rec_response.status_code}")
    print(f"   Final URL: {rec_response.url}")
    
    if 'style_recommendations' in rec_response.url:
        print("   ✅ Successfully accessed recommendations")
        if 'No Style Recommendations Available' in rec_response.text:
            print("   ❌ But page shows 'No recommendations available'")
        else:
            print("   ✅ Recommendations page has content")
    else:
        print("   ❌ Redirected away from recommendations")

def test_with_existing_user():
    """Test with a user that already has preferences"""
    print("\n🧪 Testing with existing user (swaroop1902)...")
    
    base_url = "http://127.0.0.1:5000"
    session = requests.Session()
    
    # Try to login as existing user
    login_data = {
        'username': 'swaroop1902',
        'password': 'password123'  # Assuming this is the password
    }
    
    response = session.post(f"{base_url}/login", data=login_data)
    if response.status_code == 200 and 'dashboard' in response.url:
        print("   ✅ Logged in as existing user")
        
        # Try to access recommendations
        rec_response = session.get(f"{base_url}/style_recommendations")
        print(f"   Recommendations response: {rec_response.status_code}")
        print(f"   Final URL: {rec_response.url}")
        
        if 'style_recommendations' in rec_response.url:
            print("   ✅ Successfully accessed recommendations")
        else:
            print("   ❌ Redirected away from recommendations")
    else:
        print("   ❌ Could not login as existing user")

def main():
    """Run debugging tests"""
    print("🚀 Starting Form Submission Debug...\n")
    
    check_user_before_and_after()
    test_with_existing_user()

if __name__ == "__main__":
    main()
