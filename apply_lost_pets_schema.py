#!/usr/bin/env python3
"""
Apply lost_pets table schema to Cloudflare D1 database
"""

import asyncio
from database import db

async def apply_schema():
    """Apply the lost_pets table schema"""
    print("Applying lost_pets table schema...")

    # Read schema file
    with open('/opt/petsearch/schema_lost_pets.sql', 'r') as f:
        schema_sql = f.read()

    # Split by semicolons to execute each statement separately
    statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]

    for i, statement in enumerate(statements, 1):
        print(f"\nExecuting statement {i}...")
        print(f"{statement[:100]}...")
        try:
            await db.execute(statement)
            print(f"✓ Statement {i} executed successfully")
        except Exception as e:
            print(f"✗ Error executing statement {i}: {e}")
            # Continue anyway - table might already exist

    print("\n✓ Schema application complete!")

    # Verify table exists
    print("\nVerifying lost_pets table...")
    result = await db.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='lost_pets'
    """)

    if result.get('results'):
        print("✓ lost_pets table exists")
    else:
        print("✗ lost_pets table NOT found")

if __name__ == '__main__':
    asyncio.run(apply_schema())
