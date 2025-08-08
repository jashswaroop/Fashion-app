#!/usr/bin/env python3
"""
Test male user debug
"""

import requests

def test_male_force_recommendations():
    """Test force recommendations for male user"""
    print("🔍 Testing Male User Force Recommendations...")
    
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
            print("   ✅ Logged in successfully")
            
            # Test force recommendations endpoint
            force_response = session.get(f"{base_url}/force_recommendations")
            print(f"   Force recommendations response: {force_response.status_code}")
            
            if force_response.status_code == 200:
                content = force_response.text
                
                # Look for male-specific clothing items in the content
                male_items = [
                    'suit jacket', 'dress shirt', 'tie', 'chinos', 'polo shirt', 
                    'briefcase', 'watch', 'dress shoes', 'loafers', 'blazer',
                    'button down', 'henley', 'baseball cap'
                ]
                
                female_items = [
                    'blouse', 'dress', 'heels', 'skirt', 'pumps', 'sheath dress',
                    'midi dress', 'clutch', 'statement earrings'
                ]
                
                print(f"\n   🔍 Searching for clothing items in recommendations...")
                
                male_found = []
                female_found = []
                
                for item in male_items:
                    if item.lower() in content.lower():
                        male_found.append(item)
                        
                for item in female_items:
                    if item.lower() in content.lower():
                        female_found.append(item)
                
                print(f"   📊 Male items found: {male_found}")
                print(f"   📊 Female items found: {female_found}")
                
                # Check for specific male outfit names
                male_outfits = [
                    'Classic Business Suit', 'Smart Casual Office Look', 
                    'Professional Polo Look', 'Date Night Smart', 'Formal Evening'
                ]
                
                male_outfit_found = []
                for outfit in male_outfits:
                    if outfit in content:
                        male_outfit_found.append(outfit)
                        
                print(f"   👔 Male outfit names found: {male_outfit_found}")
                
                if male_found or male_outfit_found:
                    print("   ✅ Male-specific recommendations detected!")
                    return True
                else:
                    print("   ❌ No male-specific recommendations found")
                    
                    # Print a sample of the content to debug
                    print(f"\n   📄 Content sample (first 500 chars):")
                    print(f"   {content[:500]}...")
                    
                    return False
            else:
                print(f"   ❌ Force recommendations failed: {force_response.status_code}")
                return False
        else:
            print(f"   ❌ Login failed: {login_response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == "__main__":
    test_male_force_recommendations()
