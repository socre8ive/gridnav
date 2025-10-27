#!/usr/bin/env python3
"""
Connectivity Analysis Background Worker

Continuously monitors for new searches and runs connectivity analysis in the background.
"""

import asyncio
import json
import time
from typing import List, Dict
from database import db

async def normalize_point(lat, lon):
    """Round to 6 decimals (~0.1 meter precision)"""
    return (round(lat, 6), round(lon, 6))

async def get_adjacent_grids(grid, all_grids, threshold_miles=0.5):
    """Find grids that are adjacent or nearby"""
    threshold_deg = threshold_miles / 69.0
    grid_center_lat = (grid['min_lat'] + grid['max_lat']) / 2
    grid_center_lon = (grid['min_lon'] + grid['max_lon']) / 2

    adjacent = []
    for other_grid in all_grids:
        if other_grid['id'] == grid['id']:
            continue
        other_center_lat = (other_grid['min_lat'] + other_grid['max_lat']) / 2
        other_center_lon = (other_grid['min_lon'] + other_grid['max_lon']) / 2
        lat_diff = abs(grid_center_lat - other_center_lat)
        lon_diff = abs(grid_center_lon - other_center_lon)
        if lat_diff <= threshold_deg and lon_diff <= threshold_deg:
            adjacent.append(other_grid)
    return adjacent

async def find_road_in_grids(road_id, grid_list):
    """Find which grids contain a specific road ID"""
    found_in = []
    for grid in grid_list:
        for road in grid.get('roads', []):
            if road['id'] == road_id:
                found_in.append(grid['id'])
                break
    return found_in

async def analyze_search_connectivity(search_id: str):
    """
    Analyze connectivity for a specific search.
    Loads grid data from database and runs BFS connectivity analysis.
    """
    try:
        print(f"\n[WORKER] Starting connectivity analysis for {search_id}")

        # Ensure table exists
        await db.create_connectivity_table()

        # Update status to running
        await db.update_connectivity_status(search_id, 'running')

        # Load grid data from database
        sql = """
            SELECT grid_id, grid_data FROM search_grids
            WHERE search_id = ?
            ORDER BY grid_id
        """
        result = await db.execute(sql, [search_id])
        grid_rows = result.get('results', [])

        if not grid_rows:
            print(f"[WORKER] No grids found for {search_id}")
            await db.update_connectivity_status(search_id, 'failed')
            return

        # Parse grid data
        grids = []
        for row in grid_rows:
            grid_data = json.loads(row['grid_data'])
            grids.append({
                'id': row['grid_id'],
                'roads': grid_data.get('road_details', []),
                'min_lat': grid_data['bounds']['min_lat'],
                'max_lat': grid_data['bounds']['max_lat'],
                'min_lon': grid_data['bounds']['min_lon'],
                'max_lon': grid_data['bounds']['max_lon']
            })

        print(f"[WORKER] Loaded {len(grids)} grids for analysis")

        total_analyzed = 0
        total_needing_extension = 0

        for grid in grids:
            roads = grid.get('roads', [])
            grid_id = grid['id']

            if len(roads) <= 1:
                # Single road or empty, mark as complete
                await db.save_connectivity_analysis(search_id, grid_id, {
                    'status': 'complete',
                    'total_roads': len(roads),
                    'connected_roads': len(roads),
                    'disconnected_roads': 0,
                    'components_count': 1 if len(roads) > 0 else 0,
                    'needs_extension': False
                })
                continue

            # Skip very large grids
            if len(roads) > 2000:
                await db.save_connectivity_analysis(search_id, grid_id, {
                    'status': 'skipped',
                    'total_roads': len(roads),
                    'connected_roads': 0,
                    'disconnected_roads': 0,
                    'components_count': 0,
                    'needs_extension': False,
                    'details': {'reason': 'too_many_roads'}
                })
                continue

            # Build connectivity graph
            endpoint_to_roads = {}
            for idx, road in enumerate(roads):
                waypoints = road.get('waypoints', [])
                if len(waypoints) < 2:
                    continue
                start = await normalize_point(waypoints[0]['lat'], waypoints[0]['lon'])
                end = await normalize_point(waypoints[-1]['lat'], waypoints[-1]['lon'])
                if start not in endpoint_to_roads:
                    endpoint_to_roads[start] = set()
                if end not in endpoint_to_roads:
                    endpoint_to_roads[end] = set()
                endpoint_to_roads[start].add(idx)
                endpoint_to_roads[end].add(idx)

            # Find connected components using BFS
            visited = set()
            components = []

            for start_idx in range(len(roads)):
                if start_idx in visited:
                    continue
                component = set()
                queue = [start_idx]
                while queue:
                    idx = queue.pop(0)
                    if idx in visited:
                        continue
                    visited.add(idx)
                    component.add(idx)
                    road = roads[idx]
                    waypoints = road.get('waypoints', [])
                    if len(waypoints) < 2:
                        continue
                    start = await normalize_point(waypoints[0]['lat'], waypoints[0]['lon'])
                    end = await normalize_point(waypoints[-1]['lat'], waypoints[-1]['lon'])
                    for endpoint in [start, end]:
                        for connected_idx in endpoint_to_roads.get(endpoint, set()):
                            if connected_idx not in visited:
                                queue.append(connected_idx)
                if component:
                    components.append(component)

            # Analyze results
            needs_extension = False
            if len(components) > 1:
                largest = max(components, key=len)
                # Check disconnected components
                for component in components:
                    if component == largest:
                        continue
                    component_road_ids = [roads[idx]['id'] for idx in component]
                    # Check if roads appear in adjacent grids
                    adjacent_grids = await get_adjacent_grids(grid, grids)
                    roads_found_elsewhere = {}
                    for road_id in component_road_ids:
                        other_grids = await find_road_in_grids(road_id, adjacent_grids)
                        if other_grids:
                            roads_found_elsewhere[road_id] = other_grids

                    # If roads NOT in adjacent grids, this grid needs extension
                    if not roads_found_elsewhere:
                        needs_extension = True
                        break

            if needs_extension:
                total_needing_extension += 1

            # Save analysis result
            await db.save_connectivity_analysis(search_id, grid_id, {
                'status': 'complete',
                'total_roads': len(roads),
                'connected_roads': len(max(components, key=len)) if components else 0,
                'disconnected_roads': len(roads) - (len(max(components, key=len)) if components else 0),
                'components_count': len(components),
                'needs_extension': needs_extension
            })

            total_analyzed += 1

        # Update final status
        await db.update_connectivity_status(search_id, 'complete')

        print(f"[WORKER] Analysis complete for {search_id}")
        print(f"[WORKER] Analyzed: {total_analyzed} grids")
        print(f"[WORKER] Needing extension: {total_needing_extension} grids")

    except Exception as e:
        print(f"[WORKER] ERROR analyzing {search_id}: {str(e)}")
        import traceback
        traceback.print_exc()
        await db.update_connectivity_status(search_id, 'failed')

async def poll_for_pending_searches():
    """
    Continuously poll database for searches needing connectivity analysis.
    """
    print("[WORKER] Connectivity worker started, polling for pending searches...")

    while True:
        try:
            # Find searches with pending connectivity status
            sql = """
                SELECT search_id, created_at
                FROM pet_searches
                WHERE connectivity_status = 'pending'
                ORDER BY created_at DESC
                LIMIT 5
            """
            result = await db.execute(sql)
            searches = result.get('results', [])

            if searches:
                print(f"[WORKER] Found {len(searches)} pending searches")
                for search in searches:
                    search_id = search['search_id']
                    await analyze_search_connectivity(search_id)

            # Wait 5 seconds before next poll
            await asyncio.sleep(5)

        except Exception as e:
            print(f"[WORKER] ERROR in poll loop: {str(e)}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(10)  # Wait longer on error

if __name__ == "__main__":
    print("[WORKER] Starting connectivity analysis worker...")
    asyncio.run(poll_for_pending_searches())
