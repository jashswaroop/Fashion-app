#!/usr/bin/env python3
"""
Test script to directly test database operations and recommendation generation
"""

import sqlite3
import json
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_user_recommendations():
    """Test recommendations for existing users"""
    print("🧪 Testing User Recommendations...")
    
    # Connect to database
    conn = sqlite3.connect('instance/fashion_app.db')
    cursor = conn.cursor()
    
    # Get users with completed style analysis
    cursor.execute("SELECT id, username, fashion_preferences FROM user WHERE style_analysis_complete = 1;")
    users = cursor.fetchall()
    
    print(f"📊 Found {len(users)} users with completed style analysis")
    
    for user_id, username, preferences_json in users:
        print(f"\n👤 Testing user: {username}")
        
        try:
            # Parse preferences
            preferences = json.loads(preferences_json)
            print(f"   ✅ Preferences loaded: {len(preferences)} fields")
            
            # Test recommendation generation
            from app import generate_style_recommendations_with_links
            
            # Create a mock user object
            class MockUser:
                def __init__(self, user_id, username):
                    self.id = user_id
                    self.username = username
                    self.gender = None
            
            mock_user = MockUser(user_id, username)
            
            # Generate recommendations
            recommendations = generate_style_recommendations_with_links(preferences, mock_user)
            
            if recommendations and recommendations.get('outfits'):
                outfit_count = sum(len(outfits) for outfits in recommendations['outfits'].values())
                print(f"   ✅ Generated {outfit_count} outfits successfully")
                
                # Print outfit categories
                for category, outfits in recommendations['outfits'].items():
                    if outfits:
                        print(f"      {category}: {len(outfits)} outfits")
                        
                return True
            else:
                print(f"   ❌ No recommendations generated")
                return False
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    conn.close()

def test_minimal_preferences():
    """Test with minimal required preferences"""
    print(f"\n🧪 Testing Minimal Preferences...")
    
    minimal_preferences = {
        'lifestyle': 'professional',
        'work_environment': 'corporate',
        'clothing_fit': 'fitted',
        'budget_per_item': '100_200',
        'fashion_risk': 'moderate'
    }
    
    try:
        from app import generate_style_recommendations_with_links, analyze_fashion_preferences
        
        # Test analysis
        style_profile = analyze_fashion_preferences(minimal_preferences, None)
        print(f"   ✅ Style profile: {list(style_profile.keys())}")
        
        # Test recommendations
        recommendations = generate_style_recommendations_with_links(minimal_preferences, None)
        
        if recommendations and recommendations.get('outfits'):
            outfit_count = sum(len(outfits) for outfits in recommendations['outfits'].values())
            print(f"   ✅ Generated {outfit_count} outfits with minimal data")
            return True
        else:
            print(f"   ❌ No recommendations with minimal data")
            return False
            
    except Exception as e:
        print(f"   ❌ Error with minimal preferences: {e}")
        import traceback
        traceback.print_exc()
        return False

def simulate_form_processing():
    """Simulate the exact form processing logic"""
    print(f"\n🧪 Simulating Form Processing...")
    
    # Simulate form data as it would come from the web form
    form_data = {
        'lifestyle': 'professional',
        'work_environment': 'corporate',
        'clothing_fit': 'fitted',
        'budget_per_item': '100_200',
        'fashion_risk': 'moderate',
        'preferred_colors': 'navy',  # Single value as form would send
        'style_inspiration': 'classic'
    }
    
    try:
        # Simulate the preference collection logic from the Flask app
        preferences = {}
        
        # Single value fields
        single_fields = ['lifestyle', 'work_environment', 'clothing_fit', 'budget_per_item', 'fashion_risk']
        for field in single_fields:
            if field in form_data:
                preferences[field] = form_data[field]
        
        # List fields (in real form, these would be getlist())
        list_fields = ['social_activities', 'style_inspiration', 'preferred_colors', 'avoided_colors']
        for field in list_fields:
            if field in form_data:
                # Simulate getlist() behavior
                preferences[field] = [form_data[field]] if isinstance(form_data[field], str) else form_data[field]
        
        print(f"   ✅ Processed preferences: {preferences}")
        
        # Check required fields
        required_fields = ['lifestyle', 'work_environment', 'clothing_fit', 'budget_per_item', 'fashion_risk']
        missing_fields = [field for field in required_fields if not preferences.get(field)]
        
        if missing_fields:
            print(f"   ❌ Missing required fields: {missing_fields}")
            return False
        else:
            print(f"   ✅ All required fields present")
        
        # Test recommendation generation
        from app import generate_style_recommendations_with_links
        recommendations = generate_style_recommendations_with_links(preferences, None)
        
        if recommendations and recommendations.get('outfits'):
            print(f"   ✅ Recommendations generated successfully")
            return True
        else:
            print(f"   ❌ No recommendations generated")
            return False
            
    except Exception as e:
        print(f"   ❌ Error in form processing simulation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Direct Database Tests...\n")
    
    test1 = test_user_recommendations()
    test2 = test_minimal_preferences()
    test3 = simulate_form_processing()
    
    print(f"\n📊 Test Results:")
    print(f"   User Recommendations: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"   Minimal Preferences: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"   Form Processing: {'✅ PASS' if test3 else '❌ FAIL'}")
    
    if all([test1, test2, test3]):
        print(f"\n✅ All tests passed - the issue is likely in the Flask app session/routing logic")
    else:
        print(f"\n❌ Some tests failed - there may be an issue with the recommendation engine")

if __name__ == "__main__":
    main()
