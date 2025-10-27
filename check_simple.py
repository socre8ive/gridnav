#!/usr/bin/env python3
"""Simple check of searches table"""
import asyncio
from database import D1Database
from dotenv import load_dotenv

load_dotenv()

async def check():
    db = D1Database()

    print("=" * 60)
    print("CHECKING SEARCHES")
    print("=" * 60)

    # Try the grid-based schema first (search_id column)
    try:
        sql = "SELECT * FROM searches LIMIT 1"
        result = await db.execute(sql)
        print("\nSearches table exists!")

        searches_result = result.get('results', [])
        if searches_result:
            print("\nSample row columns:")
            for key in searches_result[0].keys():
                print(f"  - {key}")

        # Get all searches
        sql2 = "SELECT * FROM searches"
        result2 = await db.execute(sql2)
        all_searches = result2.get('results', [])

        print(f"\nTotal searches: {len(all_searches)}")

        for search in all_searches:
            print(f"\nSearch data:")
            for key, value in search.items():
                print(f"  {key}: {value}")

    except Exception as e:
        print(f"\nError: {e}")

    # Check grid_assignments table
    try:
        sql = "SELECT * FROM grid_assignments LIMIT 5"
        result = await db.execute(sql)
        assignments = result.get('results', [])
        print(f"\n\nGrid assignments found: {len(assignments)}")
        for assign in assignments:
            print(f"\n  Assignment:")
            for key, value in assign.items():
                print(f"    {key}: {value}")
    except Exception as e:
        print(f"\nGrid assignments error: {e}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(check())
