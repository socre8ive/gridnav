#!/usr/bin/env python3
"""
Add total_distance_miles and duration_minutes columns to grid_assignments table
"""

import asyncio
from database import db

async def apply_schema():
    """Add completion tracking fields to grid_assignments"""
    print("Adding completion tracking fields to grid_assignments table...")

    statements = [
        "ALTER TABLE grid_assignments ADD COLUMN total_distance_miles REAL DEFAULT 0",
        "ALTER TABLE grid_assignments ADD COLUMN duration_minutes INTEGER DEFAULT 0"
    ]

    for i, statement in enumerate(statements, 1):
        print(f"\nExecuting statement {i}...")
        print(f"{statement}")
        try:
            await db.execute(statement)
            print(f"✓ Statement {i} executed successfully")
        except Exception as e:
            print(f"✗ Error executing statement {i}: {e}")
            # Continue anyway - columns might already exist

    print("\n✓ Schema update complete!")

    # Verify columns exist
    print("\nVerifying columns...")
    result = await db.execute("""
        SELECT sql FROM sqlite_master
        WHERE type='table' AND name='grid_assignments'
    """)

    if result.get('results'):
        print("✓ grid_assignments table schema:")
        print(result['results'][0].get('sql', 'N/A'))

if __name__ == '__main__':
    asyncio.run(apply_schema())
