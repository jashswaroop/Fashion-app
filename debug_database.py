#!/usr/bin/env python3
"""
Debug script to check the database state
"""

import sqlite3
import json
import sys
import os

def check_database():
    """Check the current state of the database"""
    print("🔍 Checking Database State...")
    
    try:
        # Connect to the database (Flask uses instance folder)
        conn = sqlite3.connect('instance/fashion_app.db')
        cursor = conn.cursor()
        
        # Check if the database exists and has tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"📋 Tables in database: {[table[0] for table in tables]}")
        
        # Check User table structure
        cursor.execute("PRAGMA table_info(user);")
        columns = cursor.fetchall()
        print(f"\n👤 User table columns:")
        for col in columns:
            print(f"   {col[1]} ({col[2]}) - {'NOT NULL' if col[3] else 'NULL'}")
        
        # Check if our test user exists
        cursor.execute("SELECT id, username, email, style_analysis_complete, fashion_preferences FROM user WHERE username = 'testuser_debug';")
        user = cursor.fetchone()
        
        if user:
            print(f"\n✅ Test user found:")
            print(f"   ID: {user[0]}")
            print(f"   Username: {user[1]}")
            print(f"   Email: {user[2]}")
            print(f"   Style Analysis Complete: {user[3]}")
            print(f"   Fashion Preferences: {user[4][:100] if user[4] else 'None'}...")
            
            if user[4]:  # If fashion_preferences exists
                try:
                    preferences = json.loads(user[4])
                    print(f"   Preferences Keys: {list(preferences.keys())}")
                except json.JSONDecodeError as e:
                    print(f"   ❌ Error parsing preferences JSON: {e}")
        else:
            print(f"\n❌ Test user not found")
        
        # Check all users with style analysis
        cursor.execute("SELECT username, style_analysis_complete, LENGTH(fashion_preferences) FROM user WHERE style_analysis_complete = 1;")
        users_with_analysis = cursor.fetchall()
        
        print(f"\n📊 Users with completed style analysis: {len(users_with_analysis)}")
        for user in users_with_analysis:
            print(f"   {user[0]}: {user[2]} chars of preferences")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_recommendation_generation():
    """Test if recommendation generation works with sample data"""
    print(f"\n🧪 Testing Recommendation Generation...")
    
    # Add the project directory to Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
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
        
        print(f"   Testing with sample preferences...")
        
        # Test analysis
        style_profile = analyze_fashion_preferences(sample_preferences, None)
        print(f"   ✅ Style profile generated: {list(style_profile.keys())}")
        
        # Test recommendations
        recommendations = generate_style_recommendations_with_links(sample_preferences, None)
        print(f"   ✅ Recommendations generated: {list(recommendations.keys())}")
        
        if recommendations.get('outfits'):
            outfit_count = sum(len(outfits) for outfits in recommendations['outfits'].values())
            print(f"   ✅ Total outfits generated: {outfit_count}")
            return True
        else:
            print(f"   ❌ No outfits in recommendations")
            return False
            
    except Exception as e:
        print(f"   ❌ Recommendation generation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all database checks"""
    print("🚀 Starting Database Debug...\n")
    
    db_ok = check_database()
    rec_ok = test_recommendation_generation()
    
    print(f"\n📊 Debug Results:")
    print(f"   Database State: {'✅ OK' if db_ok else '❌ ERROR'}")
    print(f"   Recommendation Generation: {'✅ OK' if rec_ok else '❌ ERROR'}")
    
    if not db_ok:
        print(f"\n🔧 Database Issues Found:")
        print(f"   - Check if database file exists")
        print(f"   - Verify table schema is correct")
        print(f"   - Check if new columns were added properly")
    
    if not rec_ok:
        print(f"\n🔧 Recommendation Issues Found:")
        print(f"   - Check if all required functions are defined")
        print(f"   - Verify recommendation logic is working")
        print(f"   - Check for missing dependencies")

if __name__ == "__main__":
    main()
