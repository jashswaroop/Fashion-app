#!/usr/bin/env python3
"""
Test male gender recommendations
"""

import requests
import json

def test_male_questionnaire():
    """Test questionnaire submission with male gender"""
    print("🧪 Testing Male Gender Questionnaire Submission...")
    
    base_url = "http://127.0.0.1:5000"
    session = requests.Session()
    
    # Register and login a new test user
    username = 'testuser_male'
    password = 'testpass123'
    
    # Register
    register_data = {
        'username': username,
        'email': f'{username}@test.com',
        'password': password
    }
    
    try:
        register_response = session.post(f"{base_url}/register", data=register_data)
        print(f"   Register response: {register_response.status_code}")
        
        # Login
        login_data = {
            'username': username,
            'password': password
        }
        
        login_response = session.post(f"{base_url}/login", data=login_data)
        print(f"   Login response: {login_response.status_code}")
        
        if login_response.status_code == 200:
            # Submit questionnaire with male gender
            questionnaire_data = {
                'gender': 'male',
                'age_range': '25-34',
                'lifestyle': 'professional',
                'work_environment': 'corporate',
                'social_activities': ['networking', 'dining'],
                'style_inspiration': ['classic', 'modern'],
                'preferred_colors': ['navy', 'black', 'gray'],
                'avoided_colors': ['pink'],
                'clothing_fit': 'fitted',
                'pattern_preference': ['solid'],
                'fabric_preference': ['cotton', 'wool'],
                'shopping_frequency': 'monthly',
                'budget_per_item': '100_200',
                'brand_preference': 'quality_brands',
                'sustainability_importance': 'important',
                'comfort_vs_style': 'balanced',
                'seasonal_preference': 'all_seasons',
                'body_concerns': ['none'],
                'style_goals': ['professional_image'],
                'fashion_risk': 'moderate',
                'accessory_preference': ['watches', 'belts'],
                'shoe_preference': ['dress_shoes', 'loafers']
            }
            
            questionnaire_response = session.post(f"{base_url}/submit_fashion_preferences", data=questionnaire_data)
            print(f"   Questionnaire response: {questionnaire_response.status_code}")
            print(f"   Final URL: {questionnaire_response.url}")
            
            if questionnaire_response.status_code == 200:
                if 'force_recommendations' in questionnaire_response.url:
                    print("   ✅ Successfully redirected to recommendations")
                    
                    # Check if the page has male-specific content
                    content = questionnaire_response.text
                    
                    # Look for male-specific clothing items
                    male_items = ['suit_jacket', 'dress_shirt', 'tie', 'chinos', 'polo_shirt', 'briefcase', 'watch']
                    female_items = ['blouse', 'dress', 'heels', 'skirt', 'pumps']
                    
                    male_count = sum(1 for item in male_items if item.replace('_', ' ') in content.lower())
                    female_count = sum(1 for item in female_items if item.replace('_', ' ') in content.lower())
                    
                    print(f"   📊 Male-specific items found: {male_count}")
                    print(f"   📊 Female-specific items found: {female_count}")
                    
                    if male_count > female_count:
                        print("   ✅ Recommendations appear to be male-specific!")
                        return True
                    else:
                        print("   ❌ Recommendations still appear to be female-oriented")
                        return False
                else:
                    print("   ❌ Not redirected to recommendations")
                    return False
            else:
                print(f"   ❌ Questionnaire submission failed")
                return False
        else:
            print(f"   ❌ Login failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_debug_male_user():
    """Test debug endpoint for male user"""
    print(f"\n🔍 Testing Debug Endpoint for Male User...")
    
    base_url = "http://127.0.0.1:5000"
    session = requests.Session()
    
    # Login as the male test user
    login_data = {
        'username': 'testuser_male',
        'password': 'testpass123'
    }
    
    try:
        login_response = session.post(f"{base_url}/login", data=login_data)
        if login_response.status_code == 200:
            # Test debug endpoint
            debug_response = session.get(f"{base_url}/debug_user")
            if debug_response.status_code == 200:
                user_info = debug_response.json()
                print(f"   User info: {json.dumps(user_info, indent=2)}")
                
                if user_info.get('gender') == 'Male':
                    print("   ✅ Gender correctly set to Male")
                    return True
                else:
                    print(f"   ❌ Gender is: {user_info.get('gender')}")
                    return False
            else:
                print(f"   ❌ Debug endpoint failed")
                return False
        else:
            print(f"   ❌ Login failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Male Gender Recommendations Test...\n")
    
    test1 = test_male_questionnaire()
    test2 = test_debug_male_user()
    
    print(f"\n📊 Test Results:")
    print(f"   Male Questionnaire: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"   Debug Male User: {'✅ PASS' if test2 else '❌ FAIL'}")
    
    if test1 and test2:
        print(f"\n✅ Male gender recommendations are working correctly!")
    else:
        print(f"\n❌ There are still issues with male gender recommendations")

if __name__ == "__main__":
    main()
