#!/usr/bin/env python3
"""
Test script to verify gender-specific recommendations are working correctly
"""

import requests
import json
from bs4 import BeautifulSoup

def test_male_recommendations():
    """Test that male users get male-specific recommendations"""
    
    # Create a session
    session = requests.Session()
    
    # Login as male user
    login_data = {
        'username': 'testuser',
        'password': 'testpass'
    }
    
    # First register if needed
    try:
        register_response = session.post('http://127.0.0.1:5000/register', data={
            'username': 'testuser',
            'email': 'test@example.com',
            'password': 'testpass',
            'confirm_password': 'testpass'
        })
    except:
        pass  # User might already exist
    
    # Login
    login_response = session.post('http://127.0.0.1:5000/login', data=login_data)
    
    if login_response.status_code != 200:
        print("❌ Failed to login")
        return False
    
    # Update profile to male
    profile_data = {
        'age': '25',
        'gender': 'male',
        'body_type': 'athletic',
        'style_preference': 'classic',
        'color_preference': 'neutral',
        'size_preference': 'M',
        'budget_range': 'moderate'
    }
    
    profile_response = session.post('http://127.0.0.1:5000/profile', data=profile_data)
    print(f"Profile update status: {profile_response.status_code}")
    
    # Get recommendations
    recommendations_response = session.get('http://127.0.0.1:5000/force_recommendations')
    
    if recommendations_response.status_code != 200:
        print(f"❌ Failed to get recommendations: {recommendations_response.status_code}")
        return False
    
    # Parse the HTML response
    soup = BeautifulSoup(recommendations_response.text, 'html.parser')
    
    # Check for male-specific content
    page_text = soup.get_text().lower()
    
    # Male-specific items that should be present
    male_items = ['suit', 'blazer', 'dress shirt', 'tie', 'chinos', 'polo', 'watch', 'loafers']
    
    # Female-specific items that should NOT be present
    female_items = ['dress', 'skirt', 'blouse', 'heels', 'purse', 'makeup']
    
    male_items_found = sum(1 for item in male_items if item in page_text)
    female_items_found = sum(1 for item in female_items if item in page_text)
    
    print(f"✅ Male items found: {male_items_found}/{len(male_items)}")
    print(f"❌ Female items found: {female_items_found}")
    
    # Check for outfit names
    outfit_names = soup.find_all('h4', class_='font-bold')
    outfit_names_text = [name.get_text() for name in outfit_names]
    print(f"📋 Outfit names: {outfit_names_text}")
    
    # Check images - look for different image URLs
    images = soup.find_all('img', class_='outfit-image')
    image_urls = [img.get('src') for img in images if img.get('src')]
    unique_images = len(set(image_urls))
    total_images = len(image_urls)
    
    print(f"🖼️  Images: {total_images} total, {unique_images} unique")
    
    # Success criteria
    success = (
        male_items_found >= 3 and  # At least 3 male items found
        female_items_found <= 2 and  # At most 2 female items (some might be generic)
        unique_images == total_images  # All images should be unique
    )
    
    if success:
        print("✅ GENDER FIX TEST PASSED!")
        print("   - Male-specific recommendations detected")
        print("   - Female-specific items minimized")
        print("   - Unique images for each outfit")
    else:
        print("❌ GENDER FIX TEST FAILED!")
        if male_items_found < 3:
            print(f"   - Not enough male items ({male_items_found}/3)")
        if female_items_found > 2:
            print(f"   - Too many female items ({female_items_found})")
        if unique_images != total_images:
            print(f"   - Duplicate images detected ({unique_images}/{total_images})")
    
    return success

if __name__ == "__main__":
    print("🧪 Testing Gender-Specific Recommendations Fix...")
    print("=" * 50)
    
    success = test_male_recommendations()
    
    print("=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! Gender fix is working correctly.")
    else:
        print("💥 TESTS FAILED! Gender fix needs more work.")
