#!/usr/bin/env python3
"""
Test the complete-search endpoint end-to-end
"""

import asyncio
from database import db
import time
import uuid

async def test_complete_search():
    """Test the complete search workflow"""

    # Test data
    test_pet_id = f"test-pet-{uuid.uuid4()}"
    test_search_id = f"test-search-{uuid.uuid4()}"
    test_searcher_id = f"test-searcher-{uuid.uuid4()}"

    print("=" * 60)
    print("Testing Complete Search Workflow")
    print("=" * 60)

    # Step 1: Create a test pet
    print("\n1. Creating test pet...")
    await db.save_lost_pet(
        test_pet_id,
        "Test Dog Max",
        "https://example.com/max.jpg"
    )
    print(f"   ✓ Created pet: {test_pet_id}")

    # Step 2: Create a test search
    print("\n2. Creating test search...")
    await db.create_search_tracking(
        test_search_id,
        test_pet_id,
        "123 Test Street",
        37.7749,
        -122.4194,
        0.5,
        4
    )
    print(f"   ✓ Created search: {test_search_id}")

    # Step 3: Create a grid assignment
    print("\n3. Creating grid assignment...")
    assignment = await db.assign_grid(
        test_search_id,
        test_pet_id,
        1,  # grid_id
        test_searcher_id,
        "Test User",
        30  # timeframe_minutes
    )
    assignment_id = assignment['assignment_id']
    print(f"   ✓ Created assignment: {assignment_id}")

    # Step 4: Add some GPS breadcrumbs
    print("\n4. Adding GPS breadcrumbs...")
    gps_points = [
        (37.7749, -122.4194),
        (37.7750, -122.4195),
        (37.7751, -122.4196),
        (37.7752, -122.4197),
    ]

    for i, (lat, lon) in enumerate(gps_points):
        await db.update_search_progress(
            assignment_id,
            test_search_id,
            test_pet_id,
            1,  # grid_id
            test_searcher_id,
            lat,
            lon,
            10.0,  # accuracy_meters
            0.5 + (i * 0.3),  # cumulative distance_miles
            5 + (i * 5)  # cumulative elapsed_minutes
        )
    print(f"   ✓ Added {len(gps_points)} GPS points")

    # Step 5: Complete the search using the new method
    print("\n5. Completing search...")
    result = await db.complete_search(
        test_search_id,
        test_pet_id,
        test_searcher_id,
        2.5,  # total_distance_miles
        45    # duration_minutes
    )
    print(f"   ✓ Search completed:")
    print(f"     - Assignment ID: {result['assignment_id']}")
    print(f"     - Distance: {result['total_distance_miles']} miles")
    print(f"     - Duration: {result['duration_minutes']} minutes")

    # Step 6: Verify it appears in search history
    print("\n6. Fetching search history...")
    history = await db.get_search_history(test_searcher_id)
    print(f"   ✓ Found {len(history)} completed searches")

    if len(history) > 0:
        search = history[0]
        print(f"\n   Search Details:")
        print(f"     - Pet Name: {search['pet_name']}")
        print(f"     - Distance: {search['total_distance_miles']} miles")
        print(f"     - Duration: {search['duration_minutes']} minutes")
        print(f"     - GPS Points: {len(search['route'])}")

        # Verify the values match
        if search['total_distance_miles'] == 2.5 and search['duration_minutes'] == 45:
            print(f"\n   ✅ SUCCESS! Values match expected results")
        else:
            print(f"\n   ⚠️  WARNING: Values don't match")
            print(f"     Expected: 2.5 miles, 45 minutes")
            print(f"     Got: {search['total_distance_miles']} miles, {search['duration_minutes']} minutes")

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == '__main__':
    asyncio.run(test_complete_search())
