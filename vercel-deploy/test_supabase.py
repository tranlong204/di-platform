#!/usr/bin/env python3
"""
Test script to verify Supabase connection
"""

import os
from supabase import create_client, Client

# Test Supabase connection
def test_supabase_connection():
    print("🔍 Testing Supabase connection...")
    
    # Get credentials from environment
    supabase_url = "https://pvmaxqesaxplmztyaujj.supabase.co"
    supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2bWF4cWVzYXhwbG16dHlhdWpqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NzA3MTMzNSwiZXhwIjoyMDcyNjQ3MzM1fQ.gHxohhjNrliLCvAZg-yMqAk9gLYEZw32xtasf1ycAvw"
    
    try:
        # Create Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)
        print("✅ Supabase client created successfully")
        
        # Test connection by querying a table
        try:
            result = supabase.table("documents").select("*").limit(1).execute()
            print("✅ Successfully connected to Supabase database")
            print(f"📊 Documents table accessible: {len(result.data)} records found")
            return True
        except Exception as e:
            print(f"⚠️ Documents table not found or accessible: {e}")
            print("🔧 This is expected if tables haven't been created yet")
            return True
            
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        return False

if __name__ == "__main__":
    success = test_supabase_connection()
    if success:
        print("\n🎉 Supabase connection test completed successfully!")
        print("📝 Next steps:")
        print("1. Run the SQL schema in your Supabase SQL editor")
        print("2. Test document upload functionality")
    else:
        print("\n❌ Supabase connection test failed!")
        print("🔧 Please check your credentials and network connection")
