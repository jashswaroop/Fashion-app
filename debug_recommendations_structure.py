#!/usr/bin/env python3
"""
Debug the recommendations data structure
"""

import json
import sys
import os

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_recommendations_structure():
    """Test the structure of recommendations"""
    print("🔍 Testing Recommendations Data Structure...")
    
    try:
        from app import generate_style_recommendations_with_links, analyze_fashion_preferences
        
        # Sample preferences
        sample_preferences = {
            'lifestyle': 'professional',
            'work_environment': 'corporate',
            'clothing_fit': 'fitted',
            'budget_per_item': '100_200',
            'fashion_risk': 'moderate',
            'preferred_colors': ['navy', 'black'],
            'style_inspiration': ['classic']
        }
        
        print("   Generating recommendations...")
        recommendations = generate_style_recommendations_with_links(sample_preferences, None)
        
        print(f"   ✅ Recommendations generated")
        print(f"   Top-level keys: {list(recommendations.keys())}")
        
        # Check seasonal recommendations structure
        if 'seasonal_recommendations' in recommendations:
            seasonal = recommendations['seasonal_recommendations']
            print(f"   Seasonal recommendations type: {type(seasonal)}")
            print(f"   Seasonal keys: {list(seasonal.keys())}")
            
            # Check first season
            first_season = list(seasonal.keys())[0]
            first_details = seasonal[first_season]
            print(f"   First season ({first_season}) type: {type(first_details)}")
            print(f"   First season keys: {list(first_details.keys())}")
            
            # Check items specifically
            if 'items' in first_details:
                items = first_details['items']
                print(f"   Items type: {type(items)}")
                print(f"   Items content: {items}")
                
                # Check if items has an 'items' method (which would cause the error)
                if hasattr(items, 'items'):
                    print(f"   ❌ Items has 'items' method - this is the problem!")
                    print(f"   Items methods: {[method for method in dir(items) if not method.startswith('_')]}")
                else:
                    print(f"   ✅ Items is a simple list")
            else:
                print(f"   ❌ No 'items' key in season details")
        else:
            print(f"   ❌ No 'seasonal_recommendations' in recommendations")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_seasonal_function_directly():
    """Test the seasonal recommendations function directly"""
    print(f"\n🧪 Testing Seasonal Function Directly...")
    
    try:
        from app import generate_seasonal_recommendations, analyze_fashion_preferences
        
        # Sample data
        sample_preferences = {
            'lifestyle': 'professional',
            'seasonal_preference': 'all_seasons'
        }
        
        style_profile = analyze_fashion_preferences(sample_preferences, None)
        
        seasonal = generate_seasonal_recommendations(style_profile, sample_preferences)
        
        print(f"   ✅ Seasonal function returned: {type(seasonal)}")
        print(f"   Keys: {list(seasonal.keys())}")
        
        # Check structure
        for season, details in seasonal.items():
            print(f"   {season}: {type(details)}")
            print(f"     Keys: {list(details.keys())}")
            print(f"     Items: {details['items']} (type: {type(details['items'])})")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Recommendations Structure Debug...\n")
    
    test1 = test_recommendations_structure()
    test2 = test_seasonal_function_directly()
    
    print(f"\n📊 Debug Results:")
    print(f"   Full Recommendations: {'✅ OK' if test1 else '❌ ERROR'}")
    print(f"   Seasonal Function: {'✅ OK' if test2 else '❌ ERROR'}")

if __name__ == "__main__":
    main()
