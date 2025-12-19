#!/usr/bin/env python3
"""
Grid Verification Worker

Runs every 30 minutes to ensure all pet searches have grids generated.
Automatically retries grid generation for any searches that are missing grids.
"""

import asyncio
import time
import signal
import sys
import os
import httpx

# Add the petsearch directory to path
sys.path.insert(0, '/opt/petsearch')

from database import D1Database

# Configuration
CHECK_INTERVAL_SECONDS = 30 * 60  # 30 minutes
API_BASE_URL = "https://localhost:8443"
API_KEY = os.getenv("API_KEY", "petsearch_2024_secure_key_f8d92a1b3c4e5f67")

# Graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print(f"[WORKER] Received signal {signum}, shutting down gracefully...")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


async def get_searches_needing_grids(db: D1Database) -> list:
    """Find all pet searches that need grid generation"""

    # Find searches where:
    # 1. total_grids = 0 (no grids generated)
    # 2. status is 'active' or 'pending' but has no grids (broken state)
    # 3. status is 'failed' (needs retry)
    # 4. status is 'processing' for more than 30 minutes (stuck)
    import time
    thirty_mins_ago = int(time.time()) - (30 * 60)

    sql = """
        SELECT
            ps.search_id,
            ps.pet_id,
            ps.total_grids,
            ps.status,
            ps.center_lat,
            ps.center_lon,
            ps.radius_miles,
            ps.created_at
        FROM pet_searches ps
        WHERE ps.total_grids = 0
           OR ps.status IN ('failed', 'pending')
           OR (ps.status = 'processing' AND ps.created_at < ?)
        ORDER BY ps.created_at DESC
    """

    result = await db.execute(sql, [thirty_mins_ago])
    return result.get('results', [])


async def get_searches_with_mismatches(db: D1Database) -> list:
    """Find active searches where total_grids doesn't match actual grid count"""

    # Get all active searches
    sql = """
        SELECT
            ps.search_id,
            ps.pet_id,
            ps.total_grids,
            ps.status,
            ps.center_lat,
            ps.center_lon,
            ps.radius_miles,
            ps.created_at
        FROM pet_searches ps
        WHERE ps.status = 'active' AND ps.total_grids > 0
        ORDER BY ps.created_at DESC
    """

    result = await db.execute(sql)
    searches = result.get('results', [])

    mismatches = []
    for search in searches:
        actual_count = await get_actual_grid_count(db, search['search_id'])
        if actual_count != search['total_grids']:
            search['actual_grids'] = actual_count
            mismatches.append(search)

    return mismatches


async def get_actual_grid_count(db: D1Database, search_id: str) -> int:
    """Get the actual count of grids in search_grids table"""
    sql = "SELECT COUNT(*) as count FROM search_grids WHERE search_id = ?"
    result = await db.execute(sql, [search_id])
    if result.get('results'):
        return result['results'][0].get('count', 0)
    return 0


async def trigger_grid_regeneration(search_id: str, pet_id: str) -> bool:
    """Trigger grid regeneration via the API"""
    try:
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.post(
                f"{API_BASE_URL}/api/retry-search",
                params={"search_id": search_id},
                headers={"X-API-Key": API_KEY},
                timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                print(f"[WORKER] ✓ Triggered regeneration for pet_id={pet_id}, search_id={search_id}")
                return True
            elif response.status_code == 400:
                # Status might not allow retry, try updating status first
                return False
            else:
                print(f"[WORKER] ✗ Failed to trigger regeneration: {response.status_code} - {response.text}")
                return False

    except Exception as e:
        print(f"[WORKER] ✗ Error triggering regeneration for {search_id}: {str(e)}")
        return False


async def fix_and_retry(db: D1Database, search: dict) -> bool:
    """Fix status if needed and retry grid generation"""
    search_id = search['search_id']
    pet_id = search['pet_id']
    status = search['status']
    total_grids = search['total_grids']

    # Check actual grid count in database
    actual_grids = await get_actual_grid_count(db, search_id)

    # If grids exist but total_grids is wrong, just update the count
    if actual_grids > 0 and total_grids == 0:
        print(f"[WORKER] Fixing grid count for pet_id={pet_id}: {total_grids} -> {actual_grids}")
        await db.execute(
            "UPDATE pet_searches SET total_grids = ?, status = 'active' WHERE search_id = ?",
            [actual_grids, search_id]
        )
        return True

    # If no grids exist, need to regenerate
    if actual_grids == 0:
        # If status is 'active' or 'processing' with 0 grids, it's a broken state - set to 'failed'
        if status in ('active', 'processing'):
            print(f"[WORKER] Fixing broken state for pet_id={pet_id}: status='{status}' but 0 grids")
            await db.execute(
                "UPDATE pet_searches SET status = 'failed' WHERE search_id = ?",
                [search_id]
            )

        # Check if we have coordinates to regenerate
        if search.get('center_lat') and search.get('center_lon'):
            return await trigger_grid_regeneration(search_id, pet_id)
        else:
            print(f"[WORKER] ✗ Cannot regenerate pet_id={pet_id}: missing coordinates")
            return False

    return True


async def fix_mismatch(db: D1Database, search: dict) -> bool:
    """Fix a search where total_grids doesn't match actual grid count"""
    search_id = search['search_id']
    pet_id = search['pet_id']
    total_grids = search['total_grids']
    actual_grids = search['actual_grids']

    # If we have some grids but fewer than expected, just update the count
    # (Grid consolidation may have reduced the count, or some failed to save)
    if actual_grids > 0:
        print(f"[WORKER] Fixing grid count mismatch for pet_id={pet_id}: {total_grids} -> {actual_grids}")
        await db.execute(
            "UPDATE pet_searches SET total_grids = ? WHERE search_id = ?",
            [actual_grids, search_id]
        )
        return True

    # If no grids at all, trigger regeneration
    print(f"[WORKER] No grids found for pet_id={pet_id} (expected {total_grids}), triggering regeneration")
    await db.execute(
        "UPDATE pet_searches SET status = 'failed', total_grids = 0 WHERE search_id = ?",
        [search_id]
    )
    return await trigger_grid_regeneration(search_id, pet_id)


async def verify_all_searches(db: D1Database) -> dict:
    """Main verification loop - check all searches and fix issues"""
    stats = {
        'checked': 0,
        'healthy': 0,
        'fixed': 0,
        'fixed_mismatches': 0,
        'regenerating': 0,
        'failed': 0
    }

    # PART 1: Check for searches with total_grids = 0 or bad status
    searches = await get_searches_needing_grids(db)
    stats['checked'] = len(searches)

    if not searches:
        print("[WORKER] ✓ No searches with missing grids")
    else:
        print(f"[WORKER] Found {len(searches)} searches needing attention")

    for search in searches:
        if shutdown_requested:
            print("[WORKER] Shutdown requested, stopping verification")
            break

        pet_id = search['pet_id']
        search_id = search['search_id']
        status = search['status']
        total_grids = search['total_grids']

        print(f"[WORKER] Checking pet_id={pet_id}: status={status}, total_grids={total_grids}")

        try:
            result = await fix_and_retry(db, search)
            if result:
                # Check if it was just a fix or needs regeneration
                actual_grids = await get_actual_grid_count(db, search_id)
                if actual_grids > 0:
                    stats['fixed'] += 1
                else:
                    stats['regenerating'] += 1
            else:
                stats['failed'] += 1
        except Exception as e:
            print(f"[WORKER] ✗ Error processing pet_id={pet_id}: {str(e)}")
            stats['failed'] += 1

        # Small delay between processing to avoid overwhelming the API
        await asyncio.sleep(2)

    # PART 2: Check for mismatches between total_grids and actual grid count
    if not shutdown_requested:
        print("[WORKER] Checking for grid count mismatches...")
        mismatches = await get_searches_with_mismatches(db)

        if not mismatches:
            print("[WORKER] ✓ No grid count mismatches found")
        else:
            print(f"[WORKER] Found {len(mismatches)} searches with grid count mismatches")

            for search in mismatches:
                if shutdown_requested:
                    break

                pet_id = search['pet_id']
                total_grids = search['total_grids']
                actual_grids = search['actual_grids']

                print(f"[WORKER] Mismatch: pet_id={pet_id}, reported={total_grids}, actual={actual_grids}")

                try:
                    if await fix_mismatch(db, search):
                        stats['fixed_mismatches'] += 1
                    else:
                        stats['failed'] += 1
                except Exception as e:
                    print(f"[WORKER] ✗ Error fixing mismatch for pet_id={pet_id}: {str(e)}")
                    stats['failed'] += 1

                await asyncio.sleep(1)

    return stats


async def run_worker():
    """Main worker loop"""
    print("[WORKER] Grid Verification Worker starting...")
    print(f"[WORKER] Check interval: {CHECK_INTERVAL_SECONDS // 60} minutes")

    db = D1Database()

    while not shutdown_requested:
        try:
            print(f"\n[WORKER] === Starting verification run at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

            stats = await verify_all_searches(db)

            print(f"[WORKER] === Verification complete ===")
            print(f"[WORKER] Checked: {stats['checked']}, Fixed: {stats['fixed']}, "
                  f"Mismatches fixed: {stats['fixed_mismatches']}, "
                  f"Regenerating: {stats['regenerating']}, Failed: {stats['failed']}")

        except Exception as e:
            print(f"[WORKER] ✗ Error in verification run: {str(e)}")
            import traceback
            traceback.print_exc()

        # Wait for next check interval
        if not shutdown_requested:
            print(f"[WORKER] Next check in {CHECK_INTERVAL_SECONDS // 60} minutes...")

            # Sleep in small increments to allow graceful shutdown
            sleep_remaining = CHECK_INTERVAL_SECONDS
            while sleep_remaining > 0 and not shutdown_requested:
                sleep_time = min(10, sleep_remaining)
                await asyncio.sleep(sleep_time)
                sleep_remaining -= sleep_time

    print("[WORKER] Grid Verification Worker stopped")


if __name__ == "__main__":
    print("=" * 60)
    print("Grid Verification Worker")
    print("=" * 60)
    asyncio.run(run_worker())
