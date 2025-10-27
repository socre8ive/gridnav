#!/usr/bin/env python3
"""Check recent progress updates"""
import asyncio
from database import D1Database
from dotenv import load_dotenv
import time

load_dotenv()

async def check():
    db = D1Database()

    print("=" * 60)
    print("RECENT PROGRESS UPDATES (Last 30 minutes)")
    print("=" * 60)

    # Check for recent progress updates
    cutoff_time = int(time.time()) - 1800  # Last 30 minutes

    sql = """
        SELECT
            progress_id,
            search_id,
            searcher_id,
            lat,
            lon,
            timestamp,
            distance_miles,
            elapsed_minutes,
            accuracy_meters
        FROM search_progress
        WHERE timestamp > ?
        ORDER BY timestamp DESC
        LIMIT 50
    """

    result = await db.execute(sql, [cutoff_time])
    updates = result.get('results', [])

    if not updates:
        print("\nNo progress updates in the last 30 minutes.")
        print("\nChecking ALL progress updates...")

        sql_all = """
            SELECT
                progress_id,
                search_id,
                searcher_id,
                lat,
                lon,
                timestamp,
                distance_miles,
                elapsed_minutes,
                accuracy_meters
            FROM search_progress
            ORDER BY timestamp DESC
            LIMIT 10
        """
        result_all = await db.execute(sql_all)
        all_updates = result_all.get('results', [])

        if not all_updates:
            print("No progress updates found at all in the database.")
        else:
            print(f"\nFound {len(all_updates)} total progress updates (showing last 10):")
            for update in all_updates:
                print(f"\n  Search: {update['search_id']}")
                print(f"  Searcher: {update['searcher_id']}")
                print(f"  Location: ({update['lat']}, {update['lon']})")
                print(f"  Distance: {update.get('distance_miles', 'N/A')} miles")
                print(f"  Time: {update.get('elapsed_minutes', 'N/A')} minutes")
                print(f"  Timestamp: {update['timestamp']} ({time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(update['timestamp']))} UTC)")
    else:
        print(f"\nFound {len(updates)} progress update(s):\n")

        for update in updates:
            print(f"\n  Search: {update['search_id']}")
            print(f"  Searcher: {update['searcher_id']}")
            print(f"  Location: ({update['lat']}, {update['lon']})")
            print(f"  Distance: {update.get('distance_miles', 'N/A')} miles")
            print(f"  Time: {update.get('elapsed_minutes', 'N/A')} minutes")
            print(f"  Timestamp: {update['timestamp']} ({time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(update['timestamp']))} UTC)")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    asyncio.run(check())
