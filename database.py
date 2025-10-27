#!/usr/bin/env python3
"""
Cloudflare D1 Database Helper Module
Handles all database operations for pet search roads tracking
"""

import hashlib
import json
from typing import List, Dict, Optional
from datetime import datetime
import httpx
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
DATABASE_ID = os.getenv("CLOUDFLARE_DATABASE_ID")
API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")

class D1Database:
    """Cloudflare D1 Database client"""

    def __init__(self):
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/d1/database/{DATABASE_ID}"
        self.headers = {
            "Authorization": f"Bearer {API_TOKEN}",
            "Content-Type": "application/json"
        }

    async def execute(self, sql: str, params: Optional[List] = None) -> Dict:
        """Execute a SQL query against D1 database"""
        async with httpx.AsyncClient() as client:
            payload = {"sql": sql}
            if params:
                payload["params"] = params

            response = await client.post(
                f"{self.base_url}/query",
                headers=self.headers,
                json=payload,
                timeout=30.0
            )

            result = response.json()
            if not result.get('success'):
                import sys
                print(f"SQL Error - Query: {sql[:100]}", file=sys.stderr)
                print(f"SQL Error - Params: {params}", file=sys.stderr)
                print(f"SQL Error - Response: {result}", file=sys.stderr)
                raise Exception(f"Database error: {result.get('errors')}")

            return result.get('result', [{}])[0]

    async def execute_batch(self, queries: List[str]) -> List[Dict]:
        """Execute multiple SQL queries in a batch"""
        async with httpx.AsyncClient() as client:
            payload = [{"sql": sql} for sql in queries]

            response = await client.post(
                f"{self.base_url}/query",
                headers=self.headers,
                json=payload,
                timeout=30.0
            )

            result = response.json()
            if not result.get('success'):
                raise Exception(f"Database error: {result.get('errors')}")

            return result.get('result', [])

    def generate_road_id(self, waypoints: List[Dict], name: str) -> str:
        """Generate unique ID for road based on geometry and name"""
        # Create a string representation of the road
        coords_str = ",".join([f"{w['lat']:.6f},{w['lon']:.6f}" for w in waypoints])
        road_signature = f"{name}:{coords_str}"

        # Generate SHA256 hash
        return hashlib.sha256(road_signature.encode()).hexdigest()[:32]

    async def save_search(self, search_data: Dict) -> str:
        """Save a search operation to the database"""
        sql = """
            INSERT INTO searches (id, center_lat, center_lon, radius_miles, address, total_roads, filtered_roads)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        await self.execute(sql, [
            search_data['search_id'],
            search_data['center']['lat'],
            search_data['center']['lon'],
            search_data.get('radius_miles', 0.25),
            search_data.get('address', ''),
            search_data['total_tiles'],
            search_data['filtered_count']
        ])

        return search_data['search_id']

    async def save_road(self, road_id: str, road_data: Dict) -> bool:
        """Save or update a road in the database"""
        import math

        # Clean road_data to remove NaN values
        name = road_data['name']
        if not isinstance(name, str):
            # Handle NaN or other non-string values
            try:
                if math.isnan(float(name)):
                    name = f"Unnamed {road_data.get('highway_type', 'road')}"
            except (ValueError, TypeError):
                name = str(name) if name is not None else "Unnamed road"

        highway_type = road_data.get('highway_type', 'unknown')
        if not isinstance(highway_type, str):
            highway_type = str(highway_type) if highway_type is not None else 'unknown'

        length_meters = road_data.get('length_meters', 0)
        if not isinstance(length_meters, (int, float)) or math.isnan(length_meters) or math.isinf(length_meters):
            length_meters = 0

        # Check if road already exists
        check_sql = "SELECT id, times_seen FROM roads WHERE id = ?"
        result = await self.execute(check_sql, [road_id])

        if result.get('results') and len(result['results']) > 0:
            # Road exists, update last_seen_at and increment times_seen
            update_sql = """
                UPDATE roads
                SET last_seen_at = CURRENT_TIMESTAMP,
                    times_seen = times_seen + 1
                WHERE id = ?
            """
            await self.execute(update_sql, [road_id])
            return False  # Not a new road
        else:
            # New road, insert it
            insert_sql = """
                INSERT INTO roads (id, name, highway_type, length_meters)
                VALUES (?, ?, ?, ?)
            """
            await self.execute(insert_sql, [
                road_id,
                name,
                highway_type,
                length_meters
            ])
            return True  # New road

    async def save_waypoints(self, road_id: str, waypoints: List[Dict]):
        """Save waypoints for a road"""
        # First check if waypoints already exist for this road
        check_sql = "SELECT COUNT(*) as count FROM road_waypoints WHERE road_id = ?"
        result = await self.execute(check_sql, [road_id])

        # If waypoints already exist, skip (they don't change)
        if result.get('results') and result['results'][0]['count'] > 0:
            return

        # Insert waypoints one at a time (simpler and more reliable)
        # For most roads, this is only 2-10 waypoints so performance is acceptable
        insert_sql = "INSERT INTO road_waypoints (road_id, sequence_order, latitude, longitude) VALUES (?, ?, ?, ?)"
        for idx, waypoint in enumerate(waypoints):
            await self.execute(insert_sql, [road_id, idx, waypoint['lat'], waypoint['lon']])

    async def link_search_to_road(self, search_id: str, road_id: str):
        """Link a search to a road (many-to-many relationship)"""
        sql = """
            INSERT OR IGNORE INTO search_roads (search_id, road_id)
            VALUES (?, ?)
        """
        await self.execute(sql, [search_id, road_id])

    async def save_search_results(self, search_data: Dict, tiles: List[Dict]) -> Dict:
        """
        Save complete search results including all roads
        Returns statistics about the save operation
        """
        stats = {
            'search_id': search_data['search_id'],
            'total_roads': len(tiles),
            'new_roads': 0,
            'existing_roads': 0,
            'total_waypoints': 0
        }

        # Save the search
        await self.save_search(search_data)

        # Save each road
        for tile in tiles:
            # Generate unique road ID
            road_id = self.generate_road_id(tile['waypoints'], tile['roads'][0])

            # Save road (returns True if new)
            is_new = await self.save_road(road_id, {
                'name': tile['roads'][0],
                'highway_type': tile['highway_type'],
                'length_meters': tile['length_meters']
            })

            if is_new:
                stats['new_roads'] += 1
                # Skip waypoints for now - too slow with individual inserts
                # TODO: Optimize waypoint saving with proper batch API
                # await self.save_waypoints(road_id, tile['waypoints'])
                # stats['total_waypoints'] += len(tile['waypoints'])
            else:
                stats['existing_roads'] += 1

            # Link search to road
            await self.link_search_to_road(search_data['search_id'], road_id)

        return stats

    async def get_search_statistics(self, search_id: str) -> Dict:
        """Get statistics for a specific search"""
        sql = """
            SELECT
                s.id,
                s.center_lat,
                s.center_lon,
                s.radius_miles,
                s.total_roads,
                s.filtered_roads,
                s.created_at,
                COUNT(DISTINCT sr.road_id) as unique_roads
            FROM searches s
            LEFT JOIN search_roads sr ON s.id = sr.search_id
            WHERE s.id = ?
            GROUP BY s.id
        """
        result = await self.execute(sql, [search_id])
        return result.get('results', [{}])[0] if result.get('results') else {}

    async def get_database_statistics(self) -> Dict:
        """Get overall database statistics"""
        sql = """
            SELECT
                (SELECT COUNT(*) FROM searches) as total_searches,
                (SELECT COUNT(*) FROM roads) as total_roads,
                (SELECT COUNT(*) FROM road_waypoints) as total_waypoints,
                (SELECT AVG(times_seen) FROM roads) as avg_road_appearances
        """
        result = await self.execute(sql)
        return result.get('results', [{}])[0] if result.get('results') else {}

    async def get_roads_in_area(self, center_lat: float, center_lon: float, radius_miles: float, limit: int = 100) -> List[Dict]:
        """
        Get roads near a specific location from database
        Uses search history to find roads discovered in nearby searches
        """
        # Find searches near this location
        # Note: This is approximate distance calculation (good enough for small areas)
        lat_delta = radius_miles / 69.0  # 1 degree lat ≈ 69 miles
        lon_delta = radius_miles / (69.0 * 0.8)  # Approximate at mid-latitudes

        sql = """
            SELECT DISTINCT r.id, r.name, r.highway_type, r.length_meters,
                   r.times_seen, r.first_seen_at, r.last_seen_at
            FROM roads r
            INNER JOIN search_roads sr ON r.id = sr.road_id
            INNER JOIN searches s ON sr.search_id = s.id
            WHERE s.center_lat BETWEEN ? AND ?
              AND s.center_lon BETWEEN ? AND ?
            ORDER BY r.name, r.length_meters DESC
            LIMIT ?
        """

        result = await self.execute(sql, [
            center_lat - lat_delta,
            center_lat + lat_delta,
            center_lon - lon_delta,
            center_lon + lon_delta,
            limit
        ])

        return result.get('results', [])

    async def get_road_with_waypoints(self, road_id: str) -> Dict:
        """Get a specific road with all its waypoints"""
        # Get road details
        road_sql = "SELECT * FROM roads WHERE id = ?"
        road_result = await self.execute(road_sql, [road_id])

        if not road_result.get('results'):
            return None

        road = road_result['results'][0]

        # Get waypoints
        waypoints_sql = """
            SELECT latitude, longitude, sequence_order
            FROM road_waypoints
            WHERE road_id = ?
            ORDER BY sequence_order
        """
        waypoints_result = await self.execute(waypoints_sql, [road_id])

        road['waypoints'] = [
            {'lat': w['latitude'], 'lon': w['longitude']}
            for w in waypoints_result.get('results', [])
        ]

        return road

    # ========================================
    # SEARCH TRACKING SYSTEM METHODS
    # ========================================

    async def create_search_tracking(self, search_id: str, pet_id: str, address: str,
                                     center_lat: float, center_lon: float,
                                     radius_miles: float, total_grids: int) -> str:
        """Create a new tracked search for a lost pet"""
        import time
        sql = """
            INSERT INTO pet_searches (search_id, pet_id, address, center_lat, center_lon,
                                 radius_miles, grid_size_miles, total_grids, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, 0.3, ?, ?, 'active')
        """
        await self.execute(sql, [
            search_id, pet_id, address, center_lat, center_lon,
            radius_miles, total_grids, int(time.time())
        ])
        return search_id

    async def assign_grid(self, search_id: str, pet_id: str, grid_id: int,
                         searcher_id: str, searcher_name: str, timeframe_minutes: int) -> Dict:
        """Assign a grid to a searcher"""
        import time
        import uuid

        assignment_id = f"assign-{uuid.uuid4()}"
        now = int(time.time())
        expires_at = now + (timeframe_minutes * 60)
        grace_expires_at = expires_at + (10 * 60)  # 10 minute grace period

        sql = """
            INSERT INTO grid_assignments (assignment_id, search_id, pet_id, grid_id,
                                         searcher_id, searcher_name, assigned_at,
                                         timeframe_minutes, expires_at, grace_expires_at,
                                         status, completion_percentage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', 0)
        """
        await self.execute(sql, [
            assignment_id, search_id, pet_id, grid_id, searcher_id,
            searcher_name, now, timeframe_minutes, expires_at, grace_expires_at
        ])

        return {
            'assignment_id': assignment_id,
            'grid_id': grid_id,
            'expires_at': expires_at,
            'grace_expires_at': grace_expires_at
        }

    async def get_available_grids(self, search_id: str, total_grids: int) -> List[int]:
        """
        Get list of available grid IDs for assignment
        Returns grids sorted by distance from center (closest first)
        Excludes grids that are:
        - Currently assigned (active)
        - Completed within last 12 hours
        """
        import time
        import json
        import math

        twelve_hours_ago = int(time.time()) - (12 * 60 * 60)

        sql = """
            SELECT grid_id FROM grid_assignments
            WHERE search_id = ?
            AND (
                (status = 'active' AND grace_expires_at > ?)
                OR (status = 'completed' AND completed_at > ?)
            )
        """
        result = await self.execute(sql, [search_id, int(time.time()), twelve_hours_ago])

        unavailable_grids = set()
        if result.get('results'):
            unavailable_grids = {row['grid_id'] for row in result['results']}

        # Get search center point
        search_sql = "SELECT center_lat, center_lon FROM pet_searches WHERE search_id = ?"
        search_result = await self.execute(search_sql, [search_id])

        if not search_result.get('results') or len(search_result['results']) == 0:
            # Fallback: return grids in creation order if no center found
            available = [gid for gid in range(1, total_grids + 1) if gid not in unavailable_grids]
            return available

        center_lat = search_result['results'][0]['center_lat']
        center_lon = search_result['results'][0]['center_lon']

        # Get all grids with their center points and calculate distances
        grids_sql = "SELECT grid_id, grid_data FROM search_grids WHERE search_id = ?"
        grids_result = await self.execute(grids_sql, [search_id])

        grid_distances = []
        if grids_result.get('results'):
            for row in grids_result['results']:
                grid_id = row['grid_id']

                # Skip unavailable grids
                if grid_id in unavailable_grids:
                    continue

                grid_data = json.loads(row['grid_data'])
                grid_center = grid_data.get('center', {})
                grid_lat = grid_center.get('lat', 0)
                grid_lon = grid_center.get('lon', 0)

                # Calculate distance from search center to grid center
                # Simple Euclidean distance (good enough for small areas)
                distance = math.sqrt((grid_lat - center_lat)**2 + (grid_lon - center_lon)**2)

                grid_distances.append((grid_id, distance))

        # Sort by distance (closest first) and return grid IDs
        grid_distances.sort(key=lambda x: x[1])
        available = [gid for gid, _ in grid_distances]

        return available

    async def update_search_progress(self, assignment_id: str, search_id: str, pet_id: str,
                                     grid_id: int, searcher_id: str, lat: float, lon: float,
                                     accuracy_meters: Optional[float] = None,
                                     distance_miles: Optional[float] = None,
                                     elapsed_minutes: Optional[int] = None) -> str:
        """Record GPS tracking point"""
        import time
        import uuid

        progress_id = f"progress-{uuid.uuid4()}"

        sql = """
            INSERT INTO search_progress (progress_id, assignment_id, search_id, pet_id,
                                         grid_id, searcher_id, lat, lon, timestamp, accuracy_meters,
                                         distance_miles, elapsed_minutes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        await self.execute(sql, [
            progress_id, assignment_id, search_id, pet_id, grid_id,
            searcher_id, lat, lon, int(time.time()), accuracy_meters,
            distance_miles, elapsed_minutes
        ])

        return progress_id

    async def mark_road_searched(self, assignment_id: str, search_id: str, pet_id: str,
                                 grid_id: int, searcher_id: str, road_id: str, road_name: str) -> str:
        """Mark a road as searched"""
        import time
        import uuid

        road_search_id = f"roadsearch-{uuid.uuid4()}"
        now = int(time.time())

        # Check if this road was searched before (for re-search tracking)
        check_sql = """
            SELECT road_search_id, searched_at, search_count
            FROM roads_searched
            WHERE search_id = ? AND road_id = ?
            ORDER BY searched_at DESC LIMIT 1
        """
        existing = await self.execute(check_sql, [search_id, road_id])

        if existing.get('results') and len(existing['results']) > 0:
            # Road was searched before - check if it's a re-search (12+ hours ago)
            last_searched = existing['results'][0]['searched_at']
            search_count = existing['results'][0]['search_count']

            if now - last_searched >= (12 * 60 * 60):
                # Re-search after 12 hours
                search_count += 1
        else:
            search_count = 1

        sql = """
            INSERT INTO roads_searched (road_search_id, assignment_id, search_id, pet_id,
                                       grid_id, searcher_id, road_id, road_name,
                                       searched_at, last_searched_at, search_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        await self.execute(sql, [
            road_search_id, assignment_id, search_id, pet_id, grid_id,
            searcher_id, road_id, road_name, now, now, search_count
        ])

        return road_search_id

    async def calculate_grid_completion(self, search_id: str, grid_id: int,
                                       total_roads_in_grid: int) -> float:
        """Calculate what percentage of a grid has been searched"""
        if total_roads_in_grid == 0:
            return 0.0

        sql = """
            SELECT COUNT(DISTINCT road_id) as searched_count
            FROM roads_searched
            WHERE search_id = ? AND grid_id = ?
        """
        result = await self.execute(sql, [search_id, grid_id])

        searched_count = 0
        if result.get('results') and len(result['results']) > 0:
            searched_count = result['results'][0]['searched_count']

        percentage = (searched_count / total_roads_in_grid) * 100.0
        return round(percentage, 2)

    async def update_assignment_completion(self, assignment_id: str, percentage: float):
        """Update the completion percentage of an assignment"""
        import time

        if percentage >= 85.0:
            # Mark as completed
            sql = """
                UPDATE grid_assignments
                SET completion_percentage = ?,
                    status = 'completed',
                    completed_at = ?
                WHERE assignment_id = ?
            """
            await self.execute(sql, [percentage, int(time.time()), assignment_id])
        else:
            sql = """
                UPDATE grid_assignments
                SET completion_percentage = ?
                WHERE assignment_id = ?
            """
            await self.execute(sql, [percentage, assignment_id])

    async def expire_old_assignments(self, search_id: str = None):
        """Mark assignments as expired if grace period has passed"""
        import time

        if search_id and search_id != 'all':
            sql = """
                UPDATE grid_assignments
                SET status = 'expired'
                WHERE search_id = ?
                AND status = 'active'
                AND grace_expires_at < ?
            """
            await self.execute(sql, [search_id, int(time.time())])
        else:
            # Update all searches when search_id is 'all' or None
            sql = """
                UPDATE grid_assignments
                SET status = 'expired'
                WHERE status = 'active'
                AND grace_expires_at < ?
            """
            await self.execute(sql, [int(time.time())])

    async def get_grid_status(self, search_id: str) -> List[Dict]:
        """Get status of all grids for a search"""
        import time

        # First expire old assignments
        await self.expire_old_assignments(search_id)

        sql = """
            SELECT
                grid_id,
                searcher_id,
                searcher_name,
                assigned_at,
                expires_at,
                grace_expires_at,
                status,
                completion_percentage,
                completed_at,
                timeframe_minutes
            FROM grid_assignments
            WHERE search_id = ?
            AND status IN ('active', 'completed')
            ORDER BY grid_id
        """
        result = await self.execute(sql, [search_id])

        return result.get('results', [])

    async def get_searcher_assignment(self, assignment_id: str) -> Optional[Dict]:
        """Get details of a specific assignment"""
        sql = """
            SELECT * FROM grid_assignments WHERE assignment_id = ?
        """
        result = await self.execute(sql, [assignment_id])

        if result.get('results') and len(result['results']) > 0:
            return result['results'][0]
        return None

    async def get_active_searchers_with_positions(self, search_id: str = None) -> List[Dict]:
        """
        Get all active searchers with their latest GPS positions
        Returns searcher info, assignment details, and most recent location
        If search_id is 'all' or None, returns all active searchers across all searches
        """
        import time

        # First expire old assignments
        await self.expire_old_assignments(search_id)

        if search_id and search_id != 'all':
            sql = """
                SELECT
                    ga.assignment_id,
                    ga.searcher_id,
                    ga.searcher_name,
                    ga.grid_id,
                    ga.assigned_at,
                    ga.expires_at,
                    ga.completion_percentage,
                    ga.timeframe_minutes,
                    sp.lat,
                    sp.lon,
                    sp.timestamp,
                    sp.accuracy_meters
                FROM grid_assignments ga
                LEFT JOIN (
                    SELECT assignment_id, lat, lon, timestamp, accuracy_meters,
                           ROW_NUMBER() OVER (PARTITION BY assignment_id ORDER BY timestamp DESC) as rn
                    FROM search_progress
                ) sp ON ga.assignment_id = sp.assignment_id AND sp.rn = 1
                WHERE ga.search_id = ?
                AND ga.status = 'active'
                AND ga.expires_at > ?
                ORDER BY ga.grid_id
            """
            result = await self.execute(sql, [search_id, int(time.time())])
        else:
            # Return all active searchers across all searches
            sql = """
                SELECT
                    ga.assignment_id,
                    ga.searcher_id,
                    ga.searcher_name,
                    ga.grid_id,
                    ga.assigned_at,
                    ga.expires_at,
                    ga.completion_percentage,
                    ga.timeframe_minutes,
                    sp.lat,
                    sp.lon,
                    sp.timestamp,
                    sp.accuracy_meters
                FROM grid_assignments ga
                LEFT JOIN (
                    SELECT assignment_id, lat, lon, timestamp, accuracy_meters,
                           ROW_NUMBER() OVER (PARTITION BY assignment_id ORDER BY timestamp DESC) as rn
                    FROM search_progress
                ) sp ON ga.assignment_id = sp.assignment_id AND sp.rn = 1
                WHERE ga.status = 'active'
                AND ga.expires_at > ?
                ORDER BY ga.grid_id
            """
            result = await self.execute(sql, [int(time.time())])

        searchers = []
        if result.get('results'):
            for row in result['results']:
                searchers.append({
                    'assignment_id': row['assignment_id'],
                    'searcher_id': row['searcher_id'],
                    'searcher_name': row['searcher_name'],
                    'grid_id': row['grid_id'],
                    'assigned_at': row['assigned_at'],
                    'expires_at': row['expires_at'],
                    'completion_percentage': row['completion_percentage'],
                    'timeframe_minutes': row['timeframe_minutes'],
                    'location': {
                        'lat': row.get('lat'),
                        'lon': row.get('lon'),
                        'timestamp': row.get('timestamp'),
                        'accuracy_meters': row.get('accuracy_meters')
                    } if row.get('lat') else None
                })

        return searchers

    async def save_grid_data(self, search_id: str, grid_id: int, grid_data: Dict) -> str:
        """Save grid tile data for later retrieval"""
        import time
        import json
        import uuid

        grid_record_id = f"grid-{uuid.uuid4()}"

        sql = """
            INSERT OR REPLACE INTO search_grids (id, search_id, grid_id, grid_data, created_at)
            VALUES (?, ?, ?, ?, ?)
        """

        await self.execute(sql, [
            grid_record_id,
            search_id,
            grid_id,
            json.dumps(grid_data),
            int(time.time())
        ])

        return grid_record_id

    async def get_grid_data(self, search_id: str, grid_id: int) -> Optional[Dict]:
        """Retrieve grid tile data by search_id and grid_id"""
        import json

        sql = """
            SELECT grid_data FROM search_grids
            WHERE search_id = ? AND grid_id = ?
        """

        result = await self.execute(sql, [search_id, grid_id])

        if result.get('results') and len(result['results']) > 0:
            grid_data_json = result['results'][0]['grid_data']
            return json.loads(grid_data_json)

        return None

    async def get_search_stats(self, search_id: str) -> Dict:
        """
        Get aggregate statistics for all searchers who searched for this pet.
        Returns total searchers, distance, hours, grids assigned, and grids completed.

        Note: elapsed_minutes and distance_miles are cumulative values sent by iOS,
        so we take MAX per searcher (their latest cumulative value), then SUM across all searchers.
        """
        # Get unique searchers count and sum of distance/time from progress updates
        # Since iOS sends cumulative values, we need MAX per searcher to avoid double-counting
        progress_sql = """
            SELECT
                COUNT(DISTINCT searcher_id) as total_searchers,
                COALESCE(SUM(max_distance), 0) as total_distance_miles,
                COALESCE(SUM(max_elapsed), 0) as total_elapsed_minutes
            FROM (
                SELECT
                    searcher_id,
                    MAX(distance_miles) as max_distance,
                    MAX(elapsed_minutes) as max_elapsed
                FROM search_progress
                WHERE search_id = ?
                GROUP BY searcher_id
            )
        """
        progress_result = await self.execute(progress_sql, [search_id])
        progress_data = progress_result.get('results', [{}])[0] if progress_result.get('results') else {}

        # Get total grids assigned and completed for this search
        grids_sql = """
            SELECT
                COUNT(*) as total_grids_assigned,
                COUNT(CASE WHEN status = 'completed' THEN 1 END) as total_grids_completed
            FROM grid_assignments
            WHERE search_id = ?
        """
        grids_result = await self.execute(grids_sql, [search_id])
        grids_data = grids_result.get('results', [{}])[0] if grids_result.get('results') else {}

        # Calculate total hours from minutes
        total_elapsed_minutes = progress_data.get('total_elapsed_minutes', 0) or 0
        total_hours = round(total_elapsed_minutes / 60, 1) if total_elapsed_minutes else 0

        return {
            'search_id': search_id,
            'total_searchers': progress_data.get('total_searchers', 0) or 0,
            'total_distance_miles': round(progress_data.get('total_distance_miles', 0) or 0, 1),
            'total_hours': total_hours,
            'total_grids_assigned': grids_data.get('total_grids_assigned', 0) or 0,
            'total_grids_completed': grids_data.get('total_grids_completed', 0) or 0
        }
    async def create_connectivity_table(self):
        """Create connectivity analysis table if it doesn't exist"""
        sql = """
            CREATE TABLE IF NOT EXISTS connectivity_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_id TEXT NOT NULL,
                grid_id INTEGER NOT NULL,
                status TEXT DEFAULT 'pending',
                total_roads INTEGER DEFAULT 0,
                connected_roads INTEGER DEFAULT 0,
                disconnected_roads INTEGER DEFAULT 0,
                components_count INTEGER DEFAULT 0,
                needs_extension BOOLEAN DEFAULT 0,
                analysis_data TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(search_id, grid_id)
            )
        """
        await self.execute(sql)

        # Create index for fast lookups
        index_sql = """
            CREATE INDEX IF NOT EXISTS idx_connectivity_search
            ON connectivity_analysis(search_id)
        """
        await self.execute(index_sql)

        # Add connectivity_status column to pet_searches if it doesn't exist
        alter_sql = """
            ALTER TABLE pet_searches
            ADD COLUMN connectivity_status TEXT DEFAULT 'pending'
        """
        try:
            await self.execute(alter_sql)
        except:
            # Column probably already exists, ignore error
            pass

    async def save_connectivity_analysis(self, search_id: str, grid_id: int, analysis_result: Dict):
        """Save connectivity analysis results for a grid"""
        sql = """
            INSERT INTO connectivity_analysis
            (search_id, grid_id, status, total_roads, connected_roads, disconnected_roads,
             components_count, needs_extension, analysis_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(search_id, grid_id) DO UPDATE SET
                status = excluded.status,
                total_roads = excluded.total_roads,
                connected_roads = excluded.connected_roads,
                disconnected_roads = excluded.disconnected_roads,
                components_count = excluded.components_count,
                needs_extension = excluded.needs_extension,
                analysis_data = excluded.analysis_data,
                updated_at = CURRENT_TIMESTAMP
        """

        await self.execute(sql, [
            search_id,
            grid_id,
            analysis_result.get('status', 'complete'),
            analysis_result.get('total_roads', 0),
            analysis_result.get('connected_roads', 0),
            analysis_result.get('disconnected_roads', 0),
            analysis_result.get('components_count', 0),
            1 if analysis_result.get('needs_extension', False) else 0,
            json.dumps(analysis_result.get('details', {}))
        ])

    async def update_connectivity_status(self, search_id: str, status: str):
        """Update overall connectivity analysis status for a search"""
        sql = """
            UPDATE pet_searches
            SET connectivity_status = ?
            WHERE search_id = ?
        """
        await self.execute(sql, [status, search_id])

    async def get_connectivity_status(self, search_id: str) -> Dict:
        """Get connectivity analysis status for a search"""
        sql = """
            SELECT
                connectivity_status,
                (SELECT COUNT(*) FROM connectivity_analysis WHERE search_id = ?) as analyzed_grids,
                (SELECT COUNT(*) FROM connectivity_analysis WHERE search_id = ? AND needs_extension = 1) as grids_needing_extension
            FROM pet_searches
            WHERE search_id = ?
        """
        result = await self.execute(sql, [search_id, search_id, search_id])
        return result.get('results', [{}])[0] if result.get('results') else {}


# Global database instance
db = D1Database()
