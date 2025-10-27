#!/usr/bin/env python3
"""Check current active searchers"""
import asyncio
from database import D1Database
from dotenv import load_dotenv
import time

load_dotenv()

async def check():
    db = D1Database()

    print("=" * 60)
    print("ACTIVE SEARCHERS")
    print("=" * 60)

    # Get all active searchers (those with recent progress updates)
    # Active = updated in last 5 minutes
    cutoff_time = int(time.time()) - 300  # 5 minutes ago

    sql = """
        SELECT
            sp.searcher_id,
            sp.search_id,
            ga.searcher_name,
            ga.grid_id,
            sp.lat,
            sp.lon,
            sp.timestamp,
            sp.distance_miles,
            sp.elapsed_minutes,
            (? - sp.timestamp) as seconds_ago
        FROM search_progress sp
        LEFT JOIN grid_assignments ga ON sp.assignment_id = ga.assignment_id
        WHERE sp.timestamp > ?
        ORDER BY sp.timestamp DESC
    """

    result = await db.execute(sql, [int(time.time()), cutoff_time])
    active = result.get('results', [])

    if not active:
        print("\n❌ No active searchers (no GPS updates in last 5 minutes)")

        # Check for any searchers with assignments
        sql2 = """
            SELECT
                assignment_id,
                search_id,
                searcher_id,
                searcher_name,
                grid_id,
                status,
                assigned_at
            FROM grid_assignments
            WHERE status = 'active'
            ORDER BY assigned_at DESC
        """
        result2 = await db.execute(sql2)
        assignments = result2.get('results', [])

        if assignments:
            print(f"\n📋 But there are {len(assignments)} searcher(s) with active assignments:")
            print("   (They haven't sent GPS updates yet)")
            for a in assignments:
                print(f"\n   • {a['searcher_name']} ({a['searcher_id']})")
                print(f"     Search: {a['search_id']}")
                print(f"     Grid: {a['grid_id']}")
                print(f"     Assigned: {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(a['assigned_at']))} UTC")
        else:
            print("\n📋 No active grid assignments either")
    else:
        print(f"\n✅ Found {len(active)} active searcher update(s):\n")

        for searcher in active:
            print(f"\n🔵 Searcher: {searcher.get('searcher_name', 'Unknown')} ({searcher['searcher_id']})")
            print(f"   Search: {searcher['search_id']}")
            print(f"   Grid: {searcher.get('grid_id', 'N/A')}")
            print(f"   Location: ({searcher['lat']}, {searcher['lon']})")
            print(f"   Distance: {searcher.get('distance_miles', 0)} miles")
            print(f"   Time: {searcher.get('elapsed_minutes', 0)} minutes")
            print(f"   Last Update: {searcher['seconds_ago']} seconds ago")

    print("\n" + "=" * 60)
    print("\nℹ️  An 'active searcher' is someone who:")
    print("   1. Has been assigned a grid to search")
    print("   2. Has sent GPS updates in the last 5 minutes")
    print("   3. Shows as a moving marker on the tracking map")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(check())
