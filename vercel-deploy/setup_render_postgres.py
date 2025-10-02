#!/usr/bin/env python3
"""
Script to test Render PostgreSQL connection and create tables
"""

import subprocess
import json
import time

# Render PostgreSQL configuration
DATABASE_URL = "postgresql://longtran:5sxTnyazQz3UZl0sepb0zxtnx2vkJlr0@dpg-d3feq0fdiees73fp1dq0-a/di_platform"

def test_postgres_connection():
    """Test PostgreSQL connection"""
    print("🔍 Testing Render PostgreSQL connection...")
    
    try:
        # Try to connect using psql if available
        result = subprocess.run([
            "psql", DATABASE_URL, "-c", "SELECT version();"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ PostgreSQL connection successful")
            print(f"Response: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ PostgreSQL connection failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("⚠️ psql not found, trying alternative method")
        return test_postgres_via_python()
    except subprocess.TimeoutExpired:
        print("❌ Connection timeout")
        return False
    except Exception as e:
        print(f"❌ Connection error: {str(e)}")
        return False

def test_postgres_via_python():
    """Test PostgreSQL connection using Python"""
    print("🔍 Testing PostgreSQL connection via Python...")
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        version = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        print("✅ PostgreSQL connection successful via Python")
        print(f"Database version: {version}")
        return True
        
    except ImportError:
        print("❌ psycopg2 not available")
        return False
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {str(e)}")
        return False

def create_tables_via_python():
    """Create tables using Python psycopg2"""
    print("🚀 Creating tables via Python...")
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Create documents table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                doc_id VARCHAR(50) UNIQUE NOT NULL,
                filename VARCHAR(255) NOT NULL,
                file_type VARCHAR(100) NOT NULL,
                file_size INTEGER NOT NULL,
                status VARCHAR(50) DEFAULT 'processed',
                chunks_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'::jsonb
            )
        """)
        print("✅ Documents table created")
        
        # Create chunks table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id SERIAL PRIMARY KEY,
                chunk_id VARCHAR(100) UNIQUE NOT NULL,
                document_id VARCHAR(50) NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                metadata JSONB DEFAULT '{}'::jsonb,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✅ Document chunks table created")
        
        # Create queries table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id SERIAL PRIMARY KEY,
                query_id VARCHAR(50) UNIQUE NOT NULL,
                query_text TEXT NOT NULL,
                response TEXT,
                processing_time FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("✅ Queries table created")
        
        conn.commit()
        cur.close()
        conn.close()
        
        print("🎉 All tables created successfully!")
        return True
        
    except ImportError:
        print("❌ psycopg2 not available")
        return False
    except Exception as e:
        print(f"❌ Error creating tables: {str(e)}")
        return False

def create_tables_via_psql():
    """Create tables using psql command"""
    print("🚀 Creating tables via psql...")
    
    sql_statements = [
        """
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            doc_id VARCHAR(50) UNIQUE NOT NULL,
            filename VARCHAR(255) NOT NULL,
            file_type VARCHAR(100) NOT NULL,
            file_size INTEGER NOT NULL,
            status VARCHAR(50) DEFAULT 'processed',
            chunks_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB DEFAULT '{}'::jsonb
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id SERIAL PRIMARY KEY,
            chunk_id VARCHAR(100) UNIQUE NOT NULL,
            document_id VARCHAR(50) NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT,
            metadata JSONB DEFAULT '{}'::jsonb,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS queries (
            id SERIAL PRIMARY KEY,
            query_id VARCHAR(50) UNIQUE NOT NULL,
            query_text TEXT NOT NULL,
            response TEXT,
            processing_time FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    ]
    
    success_count = 0
    
    for i, sql in enumerate(sql_statements, 1):
        print(f"🔧 Creating table {i}/3...")
        
        try:
            result = subprocess.run([
                "psql", DATABASE_URL, "-c", sql
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✅ Table {i} created successfully")
                success_count += 1
            else:
                print(f"❌ Table {i} creation failed: {result.stderr}")
            
        except Exception as e:
            print(f"❌ Error creating table {i}: {str(e)}")
        
        time.sleep(1)
    
    print(f"\n📊 Table Creation Summary:")
    print(f"✅ Successful: {success_count}/3")
    print(f"❌ Failed: {3 - success_count}/3")
    
    return success_count == 3

def test_tables_exist():
    """Test if tables exist"""
    print("\n🔍 Testing if tables exist...")
    
    try:
        import psycopg2
        
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        tables = ["documents", "document_chunks", "queries"]
        existing_tables = []
        
        for table in tables:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = %s
                );
            """, (table,))
            
            exists = cur.fetchone()[0]
            
            if exists:
                print(f"✅ Table '{table}' exists")
                existing_tables.append(table)
            else:
                print(f"❌ Table '{table}' not found")
        
        cur.close()
        conn.close()
        
        return existing_tables
        
    except Exception as e:
        print(f"❌ Error testing tables: {str(e)}")
        return []

def main():
    """Main function"""
    print("🚀 Render PostgreSQL Database Setup Script")
    print("=" * 50)
    print(f"📊 Database URL: {DATABASE_URL[:30]}...")
    
    # Test connection
    if not test_postgres_connection():
        print("\n❌ Cannot connect to Render PostgreSQL. Please check your configuration.")
        return
    
    # Try to create tables
    print("\n📋 Creating database tables...")
    
    # Method 1: Try Python
    if create_tables_via_python():
        print("✅ Tables created via Python")
    else:
        # Method 2: Try psql
        print("\n📋 Trying psql method...")
        if create_tables_via_psql():
            print("✅ Tables created via psql")
        else:
            print("❌ All methods failed")
    
    # Test if tables exist
    existing_tables = test_tables_exist()
    
    print(f"\n📊 Final Summary:")
    print(f"✅ Tables created: {len(existing_tables)}/3")
    print(f"📋 Existing tables: {existing_tables}")
    
    if len(existing_tables) == 3:
        print("\n🎉 All tables created successfully!")
        print("✅ Your application should now work properly")
        print("\n🎯 Next steps:")
        print("1. Deploy the updated application to Vercel")
        print("2. Test document upload functionality")
        print("3. Test chatbot functionality")
    else:
        print("\n⚠️ Some tables were not created")
        print("🔧 You may need to:")
        print("1. Check Render PostgreSQL permissions")
        print("2. Verify the database URL is correct")
        print("3. Try creating tables manually")

if __name__ == "__main__":
    main()
