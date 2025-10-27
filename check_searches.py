#!/usr/bin/env python3
"""Check active searches in the database"""
import asyncio
from database import D1Database
import os
from dotenv import load_dotenv

load_dotenv()

async def check_searches():
    db = D1Database()

    print("=" * 60)
    print("CHECKING ACTIVE SEARCHES")
    print("=" * 60)

    # Get all searches
    sql = """
        SELECT search_id, pet_id, address, center_lat, center_lon,
               radius_miles, total_grids, created_at, status
        FROM searches
        ORDER BY created_at DESC
        LIMIT 10
    """
    result = await db.execute(sql)
    searches = result.get('results', [])

    if not searches:
        print("\nNo searches found in database.")
        return

    print(f"\nFound {len(searches)} search(es):\n")

    for search in searches:
        print(f"Search ID: {search['search_id']}")
        print(f"  Pet ID: {search['pet_id']}")
        print(f"  Address: {search['address']}")
        print(f"  Location: ({search['center_lat']}, {search['center_lon']})")
        print(f"  Radius: {search['radius_miles']} miles")
        print(f"  Total Grids: {search['total_grids']}")
        print(f"  Status: {search['status']}")
        print(f"  Created: {search['created_at']}")

        # Check for grid assignments
        assign_sql = """
            SELECT COUNT(*) as count,
                   COUNT(CASE WHEN status = 'active' THEN 1 END) as active_count
            FROM grid_assignments
            WHERE search_id = ?
        """
        assign_result = await db.execute(assign_sql, [search['search_id']])
        assign_data = assign_result.get('results', [{}])[0]

        print(f"  Grid Assignments: {assign_data.get('count', 0)} total, {assign_data.get('active_count', 0)} active")

        # Check for progress updates
        progress_sql = """
            SELECT COUNT(*) as count, COUNT(DISTINCT searcher_id) as searcher_count
            FROM search_progress
            WHERE search_id = ?
        """
        progress_result = await db.execute(progress_sql, [search['search_id']])
        progress_data = progress_result.get('results', [{}])[0]

        print(f"  Progress Updates: {progress_data.get('count', 0)} total, {progress_data.get('searcher_count', 0)} unique searchers")
        print()

    print("=" * 60)
    print("\nTo view a search on the map, visit:")
    if searches:
        print(f"  http://YOUR_DOMAIN:8106/track?search_id={searches[0]['search_id']}")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(check_searches())
