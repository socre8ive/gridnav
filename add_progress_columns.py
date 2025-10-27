#!/usr/bin/env python3
"""
Migration script to add distance_miles and elapsed_minutes columns to search_progress table
"""
import asyncio
from database import D1Database
import os
from dotenv import load_dotenv

load_dotenv()

async def migrate():
    # Initialize database client
    db = D1Database()

    print("Adding distance_miles column to search_progress table...")
    try:
        await db.execute("ALTER TABLE search_progress ADD COLUMN distance_miles REAL")
        print("✓ Added distance_miles column")
    except Exception as e:
        print(f"Note: {e} (column may already exist)")

    print("Adding elapsed_minutes column to search_progress table...")
    try:
        await db.execute("ALTER TABLE search_progress ADD COLUMN elapsed_minutes INTEGER")
        print("✓ Added elapsed_minutes column")
    except Exception as e:
        print(f"Note: {e} (column may already exist)")

    print("\nMigration complete!")

if __name__ == "__main__":
    asyncio.run(migrate())
