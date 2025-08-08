#!/usr/bin/env python3
"""
Test script to simulate the questionnaire submission flow
This will help debug the issue where recommendations aren't being generated
"""

import requests
import json
import sys
import os

# Test the questionnaire submission flow
def test_questionnaire_submission():
    """Test the complete questionnaire submission flow"""
    print("🧪 Testing Questionnaire Submission Flow...")
    
    base_url = "http://127.0.0.1:5000"
    session = requests.Session()
    
    # Step 1: Register a test user
    print("\n1. Registering test user...")
    register_data = {
        'username': 'testuser_debug',
        'email': 'testuser_debug@example.com',
        'password': 'testpass123'
    }
    
    try:
        response = session.post(f"{base_url}/register", data=register_data)
        print(f"   Register response: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Registration successful or user already exists")
        else:
            print(f"   ❌ Registration failed: {response.text[:200]}")
    except Exception as e:
        print(f"   ❌ Registration error: {e}")
        return False
    
    # Step 2: Login
    print("\n2. Logging in...")
    login_data = {
        'username': 'testuser_debug',
        'password': 'testpass123'
    }
    
    try:
        response = session.post(f"{base_url}/login", data=login_data)
        print(f"   Login response: {response.status_code}")
        if response.status_code == 200 and 'dashboard' in response.url:
            print("   ✅ Login successful")
        else:
            print(f"   ❌ Login failed: {response.url}")
    except Exception as e:
        print(f"   ❌ Login error: {e}")
        return False
    
    # Step 3: Submit questionnaire
    print("\n3. Submitting questionnaire...")
    # Send data as form would send it (lists as multiple values with same key)
    questionnaire_data = {
        'lifestyle': 'professional',
        'work_environment': 'corporate',
        'clothing_fit': 'fitted',
        'budget_per_item': '100_200',
        'fashion_risk': 'moderate',
        'shopping_frequency': 'seasonally',
        'brand_preference': 'mid_range',
        'sustainability_importance': 'important',
        'comfort_vs_style': 'balanced',
        'seasonal_preference': 'all_seasons'
    }

    # Add list fields as multiple form values
    list_data = [
        ('social_activities', 'networking'),
        ('social_activities', 'dining'),
        ('style_inspiration', 'classic'),
        ('style_inspiration', 'minimalist'),
        ('preferred_colors', 'navy'),
        ('preferred_colors', 'black'),
        ('preferred_colors', 'white'),
        ('avoided_colors', 'neon_colors'),
        ('pattern_preference', 'solid'),
        ('fabric_preference', 'cotton'),
        ('fabric_preference', 'wool'),
        ('body_concerns', 'enhance_curves'),
        ('style_goals', 'build_classic_wardrobe'),
        ('accessory_preference', 'jewelry'),
        ('accessory_preference', 'bags'),
        ('shoe_preference', 'heels'),
        ('shoe_preference', 'flats')
    ]
    
    try:
        # Combine regular data with list data for form submission
        form_data = list(questionnaire_data.items()) + list_data

        response = session.post(f"{base_url}/submit_fashion_preferences", data=form_data)
        print(f"   Questionnaire response: {response.status_code}")
        print(f"   Final URL: {response.url}")

        if 'style_recommendations' in response.url or 'force_recommendations' in response.url:
            print("   ✅ Successfully redirected to recommendations")
            return True
        elif 'fashion_questionnaire' in response.url:
            print("   ❌ Redirected back to questionnaire - there's an issue")
            print(f"   Response content preview: {response.text[:500]}")
            return False
        else:
            print(f"   ❓ Unexpected redirect: {response.url}")
            return False

    except Exception as e:
        print(f"   ❌ Questionnaire submission error: {e}")
        return False

def test_direct_recommendations_access():
    """Test direct access to recommendations page"""
    print("\n🔍 Testing Direct Recommendations Access...")
    
    base_url = "http://127.0.0.1:5000"
    session = requests.Session()
    
    # Login first
    login_data = {
        'username': 'testuser_debug',
        'password': 'testpass123'
    }
    
    try:
        session.post(f"{base_url}/login", data=login_data)
        
        # Try to access recommendations directly
        response = session.get(f"{base_url}/style_recommendations")
        print(f"   Direct access response: {response.status_code}")
        print(f"   Final URL: {response.url}")
        
        if 'style_recommendations' in response.url:
            print("   ✅ Successfully accessed recommendations")
            # Check if there's actual content
            if 'No Style Recommendations Available' in response.text:
                print("   ❌ Page shows 'No recommendations available'")
                return False
            else:
                print("   ✅ Recommendations page has content")
                return True
        else:
            print("   ❌ Redirected away from recommendations")
            return False
            
    except Exception as e:
        print(f"   ❌ Direct access error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Questionnaire Flow Debug Tests...\n")
    
    # Test 1: Complete flow
    flow_success = test_questionnaire_submission()
    
    # Test 2: Direct access
    direct_success = test_direct_recommendations_access()
    
    print(f"\n📊 Test Results:")
    print(f"   Questionnaire Flow: {'✅ PASS' if flow_success else '❌ FAIL'}")
    print(f"   Direct Access: {'✅ PASS' if direct_success else '❌ FAIL'}")
    
    if not flow_success:
        print(f"\n🔧 Debugging Suggestions:")
        print(f"   1. Check Flask app logs for errors")
        print(f"   2. Verify database schema is correct")
        print(f"   3. Check if recommendation generation is working")
        print(f"   4. Verify user session is maintained")

if __name__ == "__main__":
    main()
