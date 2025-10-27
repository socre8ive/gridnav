#!/usr/bin/env python3
"""Check what tables exist in the database"""
import asyncio
from database import D1Database
from dotenv import load_dotenv

load_dotenv()

async def check_tables():
    db = D1Database()

    print("=" * 60)
    print("DATABASE TABLES")
    print("=" * 60)

    # Get all tables
    sql = "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    result = await db.execute(sql)
    tables = result.get('results', [])

    if not tables:
        print("\nNo tables found in database.")
        return

    print(f"\nFound {len(tables)} table(s):\n")
    for table in tables:
        table_name = table['name']
        print(f"\nTable: {table_name}")

        # Get table schema
        schema_sql = f"PRAGMA table_info({table_name})"
        schema_result = await db.execute(schema_sql)
        columns = schema_result.get('results', [])

        print("  Columns:")
        for col in columns:
            print(f"    - {col['name']} ({col['type']})")

        # Get row count
        count_sql = f"SELECT COUNT(*) as count FROM {table_name}"
        count_result = await db.execute(count_sql)
        count = count_result.get('results', [{}])[0].get('count', 0)
        print(f"  Rows: {count}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(check_tables())
