#!/usr/bin/env python3
"""
GEOGRAPHIC GRID SYSTEM - Creates fixed-size geographic grids for volunteer search teams
Uses proven anti-diagonal filtering from server_final.py
Creates grids of specific geographic dimensions (e.g., 0.5 x 0.5 miles) instead of grouping roads
"""

from fastapi import FastAPI, HTTPException, Header, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import osmnx as ox  # Keep for potential fallback
import overpy  # Overpass API for querying OSM data
from geopy.distance import geodesic  # For calculating road lengths
import json
from typing import List, Dict, Optional
from pydantic import BaseModel
import uuid
import math
from shapely.geometry import LineString, Point, box
from database import db
import os
import asyncio
import threading
import networkx as nx

# API Key for authentication
API_KEY = os.getenv("API_KEY", "petsearch_2024_secure_key_f8d92a1b3c4e5f67")  # Change this!

# Configure Overpass API (no local files needed!)
OVERPASS_TIMEOUT = 120  # seconds

# Keep OSMnx settings for potential fallback
ox.settings.use_cache = True
ox.settings.log_console = True

app = FastAPI(title="Pet Search Geographic Grid API")

# API Key verification dependency
async def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key from X-API-Key header"""
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key. Include X-API-Key header."
        )
    return x_api_key

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    lat: float
    lon: float
    radius_miles: float = 0.25
    grid_size_miles: float = 0.5  # Size of each grid square
    address: Optional[str] = None
    pet_id: Optional[str] = None  # For iOS app tracking
    force_regenerate: bool = False  # If True, delete old grids and regenerate at new location

def is_definitely_fake(edge, coords, edge_data):
    """
    Identify fake diagonal connections with high accuracy
    These are edges added by OSMNX for graph connectivity
    COPIED FROM server_final.py - PROVEN WORKING LOGIC
    """
    # Get properties
    street_name = edge_data.get('name', None)
    if isinstance(street_name, list):
        street_name = street_name[0] if street_name else None

    highway_type = edge_data.get('highway', 'unknown')
    if isinstance(highway_type, list):
        highway_type = highway_type[0] if highway_type else 'unknown'

    length = edge_data.get('length', 0)

    # Real roads have names OR are short service roads
    if street_name:
        return False  # Named roads are always real

    # Check if it's a straight line (only 2 points)
    if len(coords) != 2:
        return False  # Multi-point segments are usually real

    # Calculate angle
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angle_mod = abs(angle % 90)

    # Check if diagonal AND unnamed AND long
    if angle_mod > 30 and angle_mod < 60:  # Diagonal range
        if length > 50:  # Longer than 50 meters
            # This is very likely a fake connection
            return True

    # Also filter very long unnamed segments
    if not street_name and length > 150:
        return True

    return False

def create_geographic_grids(center_lat, center_lon, radius_miles, grid_size_miles, overlap_pct=10):
    """
    Create geographic grid squares covering the search area

    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        radius_miles: Total search radius
        grid_size_miles: Size of each grid square (e.g., 0.5 for 0.5x0.5 mile grids)
        overlap_pct: Percentage overlap between adjacent grids (default 10%, range 3-12%)

    Returns:
        List of grid dictionaries with bounds
    """
    # Convert miles to degrees (approximate)
    # 1 degree latitude ≈ 69 miles
    # 1 degree longitude varies by latitude
    lat_miles_per_degree = 69.0
    lon_miles_per_degree = 69.0 * math.cos(math.radians(center_lat))

    grid_size_lat = grid_size_miles / lat_miles_per_degree
    grid_size_lon = grid_size_miles / lon_miles_per_degree

    # Calculate overlap in degrees
    overlap_lat = grid_size_lat * (overlap_pct / 100.0)
    overlap_lon = grid_size_lon * (overlap_pct / 100.0)

    # Calculate step size (grid size minus overlap)
    step_lat = grid_size_lat - overlap_lat
    step_lon = grid_size_lon - overlap_lon

    # Calculate bounds of total area
    total_radius_lat = radius_miles / lat_miles_per_degree
    total_radius_lon = radius_miles / lon_miles_per_degree

    min_lat = center_lat - total_radius_lat
    max_lat = center_lat + total_radius_lat
    min_lon = center_lon - total_radius_lon
    max_lon = center_lon + total_radius_lon

    grids = []
    grid_id = 1

    # Scan from NW to SE (top-left to bottom-right)
    current_lat = max_lat  # Start from north (top)

    while current_lat > min_lat:
        current_lon = min_lon  # Start from west (left)

        while current_lon < max_lon:
            # Create grid bounds
            grid_min_lat = current_lat - grid_size_lat
            grid_max_lat = current_lat
            grid_min_lon = current_lon
            grid_max_lon = current_lon + grid_size_lon

            # Create bounding box for this grid
            bbox = box(grid_min_lon, grid_min_lat, grid_max_lon, grid_max_lat)

            grids.append({
                'id': grid_id,
                'min_lat': grid_min_lat,
                'max_lat': grid_max_lat,
                'min_lon': grid_min_lon,
                'max_lon': grid_max_lon,
                'bbox': bbox,
                'center_lat': (grid_min_lat + grid_max_lat) / 2,
                'center_lon': (grid_min_lon + grid_max_lon) / 2,
                'roads': []
            })

            grid_id += 1
            current_lon += step_lon

        current_lat -= step_lat

    # Grid creation is scan-based (NW to SE) for complete coverage
    # Now renumber by distance from center so Grid 1-4 are closest to pet location
    def distance_from_center(grid):
        lat_diff = grid['center_lat'] - center_lat
        lon_diff = grid['center_lon'] - center_lon
        return math.sqrt(lat_diff**2 + lon_diff**2)

    # Sort by distance (closest first)
    grids.sort(key=distance_from_center)

    # Renumber grids 1, 2, 3, 4... from center outward
    for i, grid in enumerate(grids, 1):
        grid['id'] = i

    return grids

def filter_connected_roads_sequential(grids, center_lat, center_lon):
    """
    Filter grids to only include roads connected to the main road network,
    building connectivity sequentially from the center outward.

    Args:
        grids: List of grid dictionaries (already sorted by distance from center)
        center_lat: Search center latitude
        center_lon: Search center longitude

    Returns:
        Filtered grids with only connected roads, empty grids removed
    """
    if not grids:
        return []

    # Track all connected road endpoints globally
    # Use a set of tuples (lat, lon) for fast lookups
    connected_endpoints = set()

    # Helper function to normalize coordinates for matching
    def normalize_coord(lat, lon, precision=6):
        """Round to 6 decimal places (~0.1 meter precision)"""
        return (round(lat, precision), round(lon, precision))

    # Helper function to get road endpoints
    def get_road_endpoints(road):
        """Extract start and end points from road waypoints"""
        waypoints = road.get('waypoints', [])
        if len(waypoints) < 2:
            return None, None

        start = normalize_coord(waypoints[0]['lat'], waypoints[0]['lon'])
        end = normalize_coord(waypoints[-1]['lat'], waypoints[-1]['lon'])
        return start, end

    # Process grids sequentially from center outward
    filtered_grids = []
    total_roads_before = 0
    total_roads_after = 0

    for grid_idx, grid in enumerate(grids):
        is_first_grid = (grid_idx == 0)
        grid_roads_before = len(grid.get('roads', []))
        total_roads_before += grid_roads_before

        if is_first_grid:
            # First grid (center): keep all roads as seed of connectivity
            filtered_roads = grid['roads']

            # Add all endpoints from first grid to connected set
            for road in filtered_roads:
                start, end = get_road_endpoints(road)
                if start and end:
                    connected_endpoints.add(start)
                    connected_endpoints.add(end)

            print(f"Grid {grid['id']} (CENTER): Kept all {len(filtered_roads)} roads as connectivity seed")
        else:
            # Subsequent grids: only keep roads that connect to existing network
            filtered_roads = []
            new_endpoints = set()

            for road in grid['roads']:
                start, end = get_road_endpoints(road)
                if not start or not end:
                    continue

                # Check if this road connects to the existing network
                if start in connected_endpoints or end in connected_endpoints:
                    filtered_roads.append(road)
                    # Add this road's endpoints to the connected set
                    new_endpoints.add(start)
                    new_endpoints.add(end)

            # Add new endpoints to global connected set
            connected_endpoints.update(new_endpoints)

            roads_removed = grid_roads_before - len(filtered_roads)
            if roads_removed > 0:
                print(f"Grid {grid['id']}: Kept {len(filtered_roads)}/{grid_roads_before} roads (removed {roads_removed} disconnected)")
            else:
                print(f"Grid {grid['id']}: Kept all {len(filtered_roads)} roads (all connected)")

        # Update grid with filtered roads
        grid['roads'] = filtered_roads
        total_roads_after += len(filtered_roads)

        # Only keep grids that have roads
        if len(filtered_roads) > 0:
            filtered_grids.append(grid)
        else:
            print(f"Grid {grid['id']}: REMOVED (no connected roads)")

    print(f"\nConnectivity filtering summary:")
    print(f"  Total roads before: {total_roads_before}")
    print(f"  Total roads after: {total_roads_after}")
    print(f"  Roads removed: {total_roads_before - total_roads_after}")
    print(f"  Grids before: {len(grids)}")
    print(f"  Grids after: {len(filtered_grids)}")
    print(f"  Grids removed: {len(grids) - len(filtered_grids)}")

    return filtered_grids

def filter_grid_connectivity(grids):
    """
    For each grid, keep only roads that are part of the largest connected component.

    This ensures all roads WITHIN A GRID connect to each other.

    Args:
        grids: List of grid dictionaries with roads assigned

    Returns:
        Grids with disconnected roads filtered out
    """
    def normalize_point(lat, lon):
        """Round to 6 decimals (~0.1 meter precision)"""
        return (round(lat, 6), round(lon, 6))

    total_roads_removed = 0
    grids_processed = 0

    for grid in grids:
        try:
            roads = grid.get('roads', [])

            if len(roads) <= 1:
                # Single road or empty grid, nothing to filter
                continue

            # Skip very large grids to avoid timeout (process later if needed)
            if len(roads) > 2000:
                print(f"  Grid {grid['id']}: Skipping (too many roads: {len(roads)})")
                continue

            # Build connectivity graph for THIS GRID ONLY
            endpoint_to_roads = {}

            for idx, road in enumerate(roads):
                waypoints = road.get('waypoints', [])
                if len(waypoints) < 2:
                    continue

                start = normalize_point(waypoints[0]['lat'], waypoints[0]['lon'])
                end = normalize_point(waypoints[-1]['lat'], waypoints[-1]['lon'])

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

                # BFS to find all roads connected to this one
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

                    start = normalize_point(waypoints[0]['lat'], waypoints[0]['lon'])
                    end = normalize_point(waypoints[-1]['lat'], waypoints[-1]['lon'])

                    # Find connected roads
                    for endpoint in [start, end]:
                        for connected_idx in endpoint_to_roads.get(endpoint, set()):
                            if connected_idx not in visited:
                                queue.append(connected_idx)

                if component:
                    components.append(component)

            # Keep only largest component in this grid
            if components:
                largest = max(components, key=len)
                removed = len(roads) - len(largest)

                if removed > 0:
                    print(f"  Grid {grid['id']}: Removed {removed} disconnected roads (kept {len(largest)}/{len(roads)})")
                    total_roads_removed += removed

                    # Update grid with only connected roads
                    grid['roads'] = [roads[idx] for idx in sorted(largest)]

                grids_processed += 1

        except Exception as e:
            print(f"  Grid {grid['id']}: Error during connectivity check: {str(e)[:100]}")
            # Keep grid as-is if error occurs
            continue

    print(f"Processed {grids_processed} grids, removed {total_roads_removed} disconnected roads")

    # Remove grids that have no roads left
    grids = [g for g in grids if len(g.get('roads', [])) > 0]

    return grids

def connect_grid_components(grids, all_roads):
    """
    Fix disconnected components within each grid by finding and adding bridge roads.
    Iteratively connects components until grid is fully connected or no more bridges found.

    Args:
        grids: List of grid dictionaries with 'roads' list
        all_roads: Complete list of all roads (used to find bridges)

    Returns:
        Updated grids with bridge roads added to connect components
    """
    import networkx as nx

    def normalize_point(lat, lon):
        """Round coordinates to 5 decimal places (~1.1m precision)"""
        return (round(lat, 5), round(lon, 5))

    # Build lookup of all roads by ID for fast access
    road_lookup = {r['id']: r for r in all_roads}

    # Build full graph from ALL roads once (reused for all grids)
    print(f"  [CONNECTIVITY] Building full road network graph from {len(all_roads)} roads...")
    full_graph = nx.Graph()
    for road in all_roads:
        waypoints = road.get('waypoints', [])
        if len(waypoints) < 2:
            continue
        start = normalize_point(waypoints[0]['lat'], waypoints[0]['lon'])
        end = normalize_point(waypoints[-1]['lat'], waypoints[-1]['lon'])
        full_graph.add_edge(start, end, road_id=road['id'], length=road.get('length_meters', 0))

    print(f"  [CONNECTIVITY] Full graph built: {full_graph.number_of_nodes()} nodes, {full_graph.number_of_edges()} edges")

    grids_fixed = 0
    total_bridges_added = 0

    for grid_idx, grid in enumerate(grids):
        grid_roads = grid.get('roads', [])
        if len(grid_roads) < 2:
            continue  # Can't have disconnected components with <2 roads

        # Iteratively fix disconnected components until fully connected
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Build graph from current roads in this grid
            G = nx.Graph()
            road_to_idx = {road['id']: idx for idx, road in enumerate(grid_roads)}

            for road in grid_roads:
                waypoints = road.get('waypoints', [])
                if len(waypoints) < 2:
                    continue

                start = normalize_point(waypoints[0]['lat'], waypoints[0]['lon'])
                end = normalize_point(waypoints[-1]['lat'], waypoints[-1]['lon'])
                G.add_edge(start, end, road_id=road['id'])

            if G.number_of_nodes() == 0:
                break

            # Find connected components
            components = list(nx.connected_components(G))

            if len(components) <= 1:
                # Grid is fully connected!
                if iteration > 1:
                    print(f"    Grid {grid['id']}: ✓ Fully connected after {iteration-1} iterations")
                break

            if iteration == 1:
                print(f"    Grid {grid['id']}: {len(components)} disconnected components ({len(grid_roads)} roads, {G.number_of_nodes()} nodes)")

            # Find bridges between components
            bridges_added = []
            components_list = sorted(components, key=len, reverse=True)

            # Try to connect largest component to all others
            largest_component = components_list[0]

            for component_idx, component in enumerate(components_list[1:], 1):
                # Find shortest path between this component and largest component
                min_path_length = float('inf')
                best_path = None

                # Use all nodes for small components, sample for large ones
                component_sample_size = min(20, len(component)) if len(component) > 20 else len(component)
                largest_sample_size = min(20, len(largest_component)) if len(largest_component) > 20 else len(largest_component)

                component_samples = list(component)[:component_sample_size]
                largest_samples = list(largest_component)[:largest_sample_size]

                for node_a in component_samples:
                    for node_b in largest_samples:
                        try:
                            path = nx.shortest_path(full_graph, node_a, node_b, weight='length')
                            path_length = nx.shortest_path_length(full_graph, node_a, node_b, weight='length')

                            if path_length < min_path_length:
                                min_path_length = path_length
                                best_path = path
                        except nx.NetworkXNoPath:
                            continue

                if best_path and len(best_path) > 1:
                    # Extract bridge roads from path
                    for i in range(len(best_path) - 1):
                        edge_data = full_graph.get_edge_data(best_path[i], best_path[i+1])
                        if edge_data:
                            bridge_road_id = edge_data.get('road_id')
                            if bridge_road_id and bridge_road_id not in road_to_idx:
                                # This road is not in the grid, add it as a bridge
                                bridge_road = road_lookup.get(bridge_road_id)
                                if bridge_road:
                                    bridges_added.append(bridge_road)
                                    road_to_idx[bridge_road_id] = len(grid_roads) + len(bridges_added) - 1
                else:
                    print(f"    Grid {grid['id']}: WARNING - No path found to connect component {component_idx} ({len(component)} nodes)")

            if bridges_added:
                grid['roads'].extend(bridges_added)
                grid_roads = grid['roads']  # Update local reference
                total_bridges_added += len(bridges_added)

                bridge_miles = sum(r['length_meters'] for r in bridges_added) / 1609.34
                print(f"    Grid {grid['id']}: Iteration {iteration}: Added {len(bridges_added)} bridge roads ({bridge_miles:.2f} mi)")

                if iteration == 1:
                    grids_fixed += 1
            else:
                # No bridges found, can't connect further
                print(f"    Grid {grid['id']}: WARNING - Could not connect all components after {iteration} iterations ({len(components)} components remain)")
                break

        # Recalculate grid totals after all iterations
        if grid.get('roads'):
            grid['total_miles'] = sum(r['length_meters'] for r in grid['roads']) / 1609.34
            grid['roads_count'] = len(grid['roads'])

    if grids_fixed > 0:
        print(f"  [CONNECTIVITY] Fixed {grids_fixed} grids by adding {total_bridges_added} bridge roads")
    else:
        print(f"  [CONNECTIVITY] All grids already fully connected")

    return grids

def create_grids_from_roads(roads, center_lat, center_lon, target_miles=6, min_miles=4, max_miles=8):
    """
    Build grids by grouping roads to reach target mileage (~6 miles per grid).

    Uses GRID PATTERN starting from NW corner (NOT radial from center).

    Algorithm:
    1. Find bounding box of all roads
    2. Start at NW corner (max_lat, min_lon)
    3. Create grids in pattern: West to East, then North to South
    4. For each grid position, accumulate nearby roads until target mileage
    5. Continue until all roads assigned

    Args:
        roads: List of road dictionaries with geometry
        center_lat: Center latitude of search area (used for bounding calc)
        center_lon: Center longitude of search area (used for bounding calc)
        target_miles: Target miles per grid (default 6)
        min_miles: Minimum miles before creating new grid (default 4)
        max_miles: Maximum miles before forcing new grid (default 8)

    Returns:
        List of grids with roads grouped by mileage
    """
    from shapely.geometry import Point, box
    from shapely.strtree import STRtree
    import math

    if not roads:
        return []

    print(f"Building grids from {len(roads)} roads (target: {target_miles} miles per grid)...")

    # Find bounding box of all roads
    all_lats = []
    all_lons = []
    for road in roads:
        for wp in road.get('waypoints', []):
            all_lats.append(wp['lat'])
            all_lons.append(wp['lon'])

    min_lat = min(all_lats)
    max_lat = max(all_lats)
    min_lon = min(all_lons)
    max_lon = max(all_lons)

    print(f"  Road area: {min_lat:.4f} to {max_lat:.4f} lat, {min_lon:.4f} to {max_lon:.4f} lon")

    # Build spatial index for FAST road lookups (critical optimization!)
    print(f"  Building spatial index for {len(roads)} roads...")
    # Use index-based approach to avoid object identity issues with spatial index
    roads_with_geom = [road for road in roads if road.get('geometry')]
    road_geometries = [road['geometry'] for road in roads_with_geom]
    spatial_index = STRtree(road_geometries)
    print(f"  Spatial index built successfully with {len(road_geometries)} geometries")

    # Test spatial index with overall bounding box
    test_bbox = box(min_lon, min_lat, max_lon, max_lat)
    test_results = spatial_index.query(test_bbox)
    print(f"  [DEBUG] Test query with full bbox found {len(test_results)} geometries")

    # Start building grids from NW corner
    # Use small initial grid size (0.3 x 0.3 miles) and expand as needed
    grid_size_degrees = 0.00435  # ~0.3 miles in degrees at this latitude

    grids = []
    grid_id = 1
    assigned_road_ids = set()

    def normalize_point(lat, lon):
        """Round to 6 decimals (~0.1 meter precision)"""
        return (round(lat, 6), round(lon, 6))

    # Traverse from North to South, West to East (like reading a book)
    current_lat = max_lat
    grid_iterations = 0

    while current_lat > min_lat:
        current_lon = min_lon

        while current_lon < max_lon:
            grid_iterations += 1
            if grid_iterations % 100 == 1:
                print(f"  [DEBUG] Grid iteration {grid_iterations}: lat={current_lat:.4f}, lon={current_lon:.4f}")
            # Define initial grid boundaries
            grid_min_lat = current_lat - grid_size_degrees
            grid_max_lat = current_lat
            grid_min_lon = current_lon
            grid_max_lon = current_lon + grid_size_degrees

            # Find candidate roads in this area (may expand later)
            grid_bbox = box(grid_min_lon, grid_min_lat, grid_max_lon, grid_max_lat)

            # Get all unassigned roads in expanded area using FAST spatial index
            expansion_factor = 10.0  # LARGE expansion to ensure we capture roads in sparse areas
            expanded_bbox = box(
                grid_min_lon - (grid_size_degrees * expansion_factor),
                grid_min_lat - (grid_size_degrees * expansion_factor),
                grid_max_lon + (grid_size_degrees * expansion_factor),
                grid_max_lat + (grid_size_degrees * expansion_factor)
            )

            if grid_iterations == 1:
                print(f"  [DEBUG] First bbox: grid({grid_min_lon:.4f},{grid_min_lat:.4f} to {grid_max_lon:.4f},{grid_max_lat:.4f})")
                print(f"  [DEBUG] Expanded: ({expanded_bbox.bounds})")

            # Use spatial index for FAST lookup instead of checking all roads
            candidate_indices = spatial_index.query(expanded_bbox, predicate='intersects')
            candidate_roads = []
            for idx in candidate_indices:
                road = roads_with_geom[idx]
                if road['id'] not in assigned_road_ids:
                    candidate_roads.append(road)

            if grid_iterations == 1:
                print(f"  [DEBUG] Query returned {len(candidate_indices)} roads, {len(candidate_roads)} unassigned candidates")

            if not candidate_roads:
                # Move to next position
                current_lon += grid_size_degrees
                continue
            else:
                print(f"  [DEBUG] Found {len(candidate_roads)} candidate roads at ({current_lat:.4f}, {current_lon:.4f})")

            # Build connectivity graph from candidate roads
            endpoint_to_roads = {}
            road_index = {}

            for idx, road in enumerate(candidate_roads):
                waypoints = road.get('waypoints', [])
                if len(waypoints) < 2:
                    continue

                road_index[idx] = road
                start = normalize_point(waypoints[0]['lat'], waypoints[0]['lon'])
                end = normalize_point(waypoints[-1]['lat'], waypoints[-1]['lon'])

                if start not in endpoint_to_roads:
                    endpoint_to_roads[start] = set()
                if end not in endpoint_to_roads:
                    endpoint_to_roads[end] = set()

                endpoint_to_roads[start].add(idx)
                endpoint_to_roads[end].add(idx)

            # Find seed road closest to grid center
            grid_center_lat = (grid_min_lat + grid_max_lat) / 2
            grid_center_lon = (grid_min_lon + grid_max_lon) / 2

            best_seed_idx = None
            best_seed_dist = float('inf')
            for idx, road in road_index.items():
                wp = road.get('waypoints', [{}])[0]
                dist = math.sqrt((wp['lat'] - grid_center_lat)**2 + (wp['lon'] - grid_center_lon)**2)
                if dist < best_seed_dist:
                    best_seed_dist = dist
                    best_seed_idx = idx

            if best_seed_idx is None:
                current_lon += grid_size_degrees
                continue

            # BFS to build connected component starting from seed, up to target miles
            visited = set()
            grid_roads = []
            grid_miles = 0
            queue = [best_seed_idx]

            while queue and grid_miles < max_miles:
                idx = queue.pop(0)
                if idx in visited:
                    continue

                visited.add(idx)
                road = road_index[idx]
                road_miles = road.get('length_meters', 0) / 1609.34

                # Add road if it won't massively exceed max
                if grid_miles + road_miles <= max_miles or grid_miles < min_miles:
                    grid_roads.append(road)
                    grid_miles += road_miles
                    assigned_road_ids.add(road['id'])

                    # Find connected roads and add to queue
                    waypoints = road.get('waypoints', [])
                    if len(waypoints) >= 2:
                        start = normalize_point(waypoints[0]['lat'], waypoints[0]['lon'])
                        end = normalize_point(waypoints[-1]['lat'], waypoints[-1]['lon'])

                        for endpoint in [start, end]:
                            for connected_idx in endpoint_to_roads.get(endpoint, set()):
                                if connected_idx not in visited:
                                    queue.append(connected_idx)

                # Stop if we hit target
                if grid_miles >= target_miles:
                    break

            # Create grid if we have roads - NO MINIMUM REQUIRED, capture ALL roads
            if grid_roads:
                grids.append(create_grid_from_roads(grid_roads, grid_id))
                if grid_miles >= min_miles:
                    print(f"  Grid {grid_id}: {len(grid_roads)} roads, {grid_miles:.2f} miles (connected)")
                else:
                    print(f"  Grid {grid_id}: {len(grid_roads)} roads, {grid_miles:.2f} miles (small isolated cluster)")
                grid_id += 1

            # Move east
            current_lon += grid_size_degrees

        # Move south
        current_lat -= grid_size_degrees

    print(f"Created {len(grids)} grids from roads")

    # CONSOLIDATION STEP: Merge tiny grids into large grids
    MIN_GRID_SIZE = 1.5  # Grids smaller than 1.5 miles are "tiny" and should be merged

    large_grids = []
    tiny_grids = []

    for grid in grids:
        grid_miles = sum(r['length_meters'] for r in grid['roads']) / 1609.34
        if grid_miles >= MIN_GRID_SIZE:
            large_grids.append(grid)
        else:
            tiny_grids.append(grid)

    print(f"  [CONSOLIDATE] {len(large_grids)} large grids (>={MIN_GRID_SIZE} mi), {len(tiny_grids)} tiny grids to merge")

    # Merge tiny grids into nearest large grids
    roads_merged = 0
    for tiny_grid in tiny_grids:
        if not large_grids:
            print(f"  [WARNING] No large grids to merge into!")
            break

        # Find nearest large grid to this tiny grid's center
        tiny_center_lat = tiny_grid.get('center_lat', (tiny_grid['min_lat'] + tiny_grid['max_lat']) / 2)
        tiny_center_lon = tiny_grid.get('center_lon', (tiny_grid['min_lon'] + tiny_grid['max_lon']) / 2)

        nearest_large = min(large_grids, key=lambda g:
            ((tiny_center_lat - g.get('center_lat', (g['min_lat'] + g['max_lat'])/2))**2 +
             (tiny_center_lon - g.get('center_lon', (g['min_lon'] + g['max_lon'])/2))**2)**0.5)

        # Merge all roads from tiny grid into nearest large grid
        nearest_large['roads'].extend(tiny_grid['roads'])
        roads_merged += len(tiny_grid['roads'])

    print(f"  [CONSOLIDATE] Merged {roads_merged} roads from {len(tiny_grids)} tiny grids into {len(large_grids)} large grids")

    # Replace grids list with only large grids
    grids = large_grids
    print(f"  [CONSOLIDATE] Reduced from {len(large_grids) + len(tiny_grids)} to {len(grids)} grids")

    # Recalculate grid totals after consolidation
    for grid in grids:
        grid['total_miles'] = sum(r['length_meters'] for r in grid['roads']) / 1609.34
        grid['roads_count'] = len(grid['roads'])

    # CLEANUP STEP: Assign orphaned roads to nearest existing grids
    unassigned_roads = [r for r in roads if r['id'] not in assigned_road_ids]
    print(f"  [DEBUG] Checking cleanup: {len(roads)} total roads, {len(assigned_road_ids)} marked as assigned, {len(unassigned_roads)} unassigned")

    if unassigned_roads:
        total_unassigned_miles = sum(r['length_meters'] for r in unassigned_roads) / 1609.34
        print(f"  [CLEANUP] {len(unassigned_roads)} orphaned roads ({total_unassigned_miles:.2f} miles) need assignment...")

        added_to_existing = 0
        bonus_grids_created = 0
        still_orphaned = []

        # Step 1: Add ALL orphaned roads to nearest existing grid (no distance limit)
        for road in unassigned_roads:
            road_center = road['waypoints'][0] if road.get('waypoints') else None
            if not road_center:
                print(f"  [CLEANUP] Skipping road {road['id']} - no waypoints")
                still_orphaned.append(road)
                continue

            # Find nearest grid (NO distance restriction)
            nearest_grid = None
            min_distance = float('inf')

            for grid in grids:
                # Check distance to grid center (use flat structure)
                grid_lat = grid.get('center_lat', (grid['min_lat'] + grid['max_lat']) / 2)
                grid_lon = grid.get('center_lon', (grid['min_lon'] + grid['max_lon']) / 2)

                dist = ((road_center['lat'] - grid_lat)**2 + (road_center['lon'] - grid_lon)**2)**0.5

                if dist < min_distance:
                    min_distance = dist
                    nearest_grid = grid

            if nearest_grid:
                # Add road to nearest grid
                nearest_grid['roads'].append(road)
                added_to_existing += 1
                assigned_road_ids.add(road['id'])
            else:
                print(f"  [CLEANUP] ERROR: No grid found for road {road['id']}")
                still_orphaned.append(road)

        print(f"  [CLEANUP] Added {added_to_existing} orphaned roads to nearest grids")

        # Step 2: Recalculate grid totals after adding orphaned roads
        for grid in grids:
            grid['total_miles'] = sum(r['length_meters'] for r in grid['roads']) / 1609.34
            grid['roads_count'] = len(grid['roads'])

        final_unassigned = [r for r in roads if r['id'] not in assigned_road_ids]
        if final_unassigned:
            final_missing_miles = sum(r['length_meters'] for r in final_unassigned) / 1609.34
            print(f"  [WARNING] {len(final_unassigned)} roads ({final_missing_miles:.2f} miles) still unassigned")
        else:
            print(f"  [SUCCESS] 100% of roads assigned to grids!")

    # CONNECTIVITY STEP: Fix disconnected components within grids
    # TEMPORARILY DISABLED due to memory issues with large searches (OOM killer)
    # TODO: Optimize memory usage or make this optional for large searches
    print(f"\n  [CONNECTIVITY] Skipping connectivity checking (disabled to prevent OOM)")
    # grids = connect_grid_components(grids, roads)

    return grids

def create_grid_from_roads(roads, grid_id):
    """
    Create a grid object from a list of roads.
    Calculate bounding box and center from the roads.
    """
    if not roads:
        return None

    # Calculate bounding box from all road waypoints
    all_lats = []
    all_lons = []

    for road in roads:
        for wp in road.get('waypoints', []):
            all_lats.append(wp['lat'])
            all_lons.append(wp['lon'])

    min_lat = min(all_lats)
    max_lat = max(all_lats)
    min_lon = min(all_lons)
    max_lon = max(all_lons)

    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    return {
        'id': grid_id,
        'min_lat': min_lat,
        'max_lat': max_lat,
        'min_lon': min_lon,
        'max_lon': max_lon,
        'center_lat': center_lat,
        'center_lon': center_lon,
        'roads': roads
    }

def split_oversized_grids(grids, target_miles=6, max_miles=8):
    """
    Split grids that have too many road miles into smaller sub-grids.

    Goal: Keep each grid close to target_miles (30 min at 12mph)

    Args:
        grids: List of grids with roads assigned
        target_miles: Target miles per grid (default 6)
        max_miles: Max miles before splitting (default 8)

    Returns:
        New list of grids with oversized grids split
    """
    from shapely.geometry import box, Point, LineString

    new_grids = []
    next_grid_id = max([g['id'] for g in grids]) + 1 if grids else 1

    for grid in grids:
        # Calculate total miles in this grid
        total_miles = sum(r.get('length_meters', 0) for r in grid['roads']) / 1609.34

        # If grid is acceptable size, keep as is
        if total_miles <= max_miles:
            new_grids.append(grid)
            continue

        # Grid is too large - split it
        # Determine split factor (2x2, 3x3, 4x4, etc.)
        split_factor = max(2, int((total_miles / target_miles) ** 0.5) + 1)
        split_factor = min(split_factor, 4)  # Max 4x4 = 16 sub-grids

        print(f"  Grid {grid['id']}: {total_miles:.1f} miles - splitting into {split_factor}x{split_factor} sub-grids")

        # Create sub-grids
        lat_range = grid['max_lat'] - grid['min_lat']
        lon_range = grid['max_lon'] - grid['min_lon']
        lat_step = lat_range / split_factor
        lon_step = lon_range / split_factor

        for row in range(split_factor):
            for col in range(split_factor):
                min_lat = grid['min_lat'] + (row * lat_step)
                max_lat = grid['min_lat'] + ((row + 1) * lat_step)
                min_lon = grid['min_lon'] + (col * lon_step)
                max_lon = grid['min_lon'] + ((col + 1) * lon_step)

                sub_grid = {
                    'id': next_grid_id,
                    'min_lat': min_lat,
                    'max_lat': max_lat,
                    'min_lon': min_lon,
                    'max_lon': max_lon,
                    'center_lat': (min_lat + max_lat) / 2,
                    'center_lon': (min_lon + max_lon) / 2,
                    'roads': []
                }

                # Assign roads to this sub-grid
                sub_bbox = box(sub_grid['min_lon'], sub_grid['min_lat'],
                              sub_grid['max_lon'], sub_grid['max_lat'])

                for road in grid['roads']:
                    if hasattr(road.get('geometry'), 'intersects'):
                        if road['geometry'].intersects(sub_bbox):
                            sub_grid['roads'].append(road)
                    else:
                        # Fallback: check if any waypoint is in the sub-grid
                        for wp in road.get('waypoints', []):
                            if (sub_grid['min_lat'] <= wp['lat'] <= sub_grid['max_lat'] and
                                sub_grid['min_lon'] <= wp['lon'] <= sub_grid['max_lon']):
                                sub_grid['roads'].append(road)
                                break

                # Only add sub-grid if it has roads
                if sub_grid['roads']:
                    new_grids.append(sub_grid)
                    next_grid_id += 1

    return new_grids

def assign_roads_to_grids(roads, grids):
    """
    Assign roads to grids with extended boundaries to ensure connectivity.

    Uses 2500ft buffer around each grid to capture connecting road segments.

    Args:
        roads: List of road dictionaries with geometry
        grids: List of grid dictionaries with bounding boxes

    Returns:
        Grids with roads assigned (including connecting segments)
    """
    # 500 feet = 0.00137 degrees (just enough to catch boundary-crossing roads)
    # Goal: Each grid should contain ~6 miles of roads (30 min at 12mph)
    buffer = 0.00137

    # Pre-create extended bboxes ONCE for all grids (not in road loop)
    print(f"Creating extended boundaries for {len(grids)} grids...")
    for grid in grids:
        grid['extended_bbox'] = box(
            grid['min_lon'] - buffer,
            grid['min_lat'] - buffer,
            grid['max_lon'] + buffer,
            grid['max_lat'] + buffer
        )

    # Assign roads using extended bbox
    for road in roads:
        road_geom = road['geometry']

        for grid in grids:
            if road_geom.intersects(grid['extended_bbox']):
                clean_road = {
                    'id': road['id'],
                    'name': road['name'],
                    'waypoints': road['waypoints'],
                    'highway_type': road['highway_type'],
                    'length_meters': road['length_meters'],
                    'has_name': road['has_name']
                }
                grid['roads'].append(clean_road)

    # Clean up extended bboxes
    for grid in grids:
        del grid['extended_bbox']

    return grids

def analyze_grid_connectivity(grids, roads_list):
    """
    Analyze each grid for road connectivity and determine if roads need to be
    moved to adjacent grids or if the grid needs to be extended.

    For each grid:
    1. Check if all roads connect to each other
    2. If disconnected roads found, check if they appear in adjacent grids
    3. If not in adjacent grids, recommend grid extension

    Args:
        grids: List of grid dictionaries with roads assigned
        roads_list: Original list of all roads with geometry data

    Returns:
        Updated grids with connectivity analysis
    """
    def normalize_point(lat, lon):
        """Round to 6 decimals (~0.1 meter precision)"""
        return (round(lat, 6), round(lon, 6))

    def get_adjacent_grids(grid, all_grids, threshold_miles=0.5):
        """Find grids that are adjacent or nearby (within threshold distance)"""
        # Convert threshold to degrees (approximate)
        threshold_deg = threshold_miles / 69.0

        grid_center_lat = (grid['min_lat'] + grid['max_lat']) / 2
        grid_center_lon = (grid['min_lon'] + grid['max_lon']) / 2

        adjacent = []
        for other_grid in all_grids:
            if other_grid['id'] == grid['id']:
                continue

            other_center_lat = (other_grid['min_lat'] + other_grid['max_lat']) / 2
            other_center_lon = (other_grid['min_lon'] + other_grid['max_lon']) / 2

            # Calculate distance between grid centers
            lat_diff = abs(grid_center_lat - other_center_lat)
            lon_diff = abs(grid_center_lon - other_center_lon)

            # Check if grids overlap or are within threshold
            if lat_diff <= threshold_deg and lon_diff <= threshold_deg:
                adjacent.append(other_grid)

        return adjacent

    def find_road_in_grids(road_id, grid_list):
        """Find which grids contain a specific road ID"""
        found_in = []
        for grid in grid_list:
            for road in grid.get('roads', []):
                if road['id'] == road_id:
                    found_in.append(grid['id'])
                    break
        return found_in

    print(f"\n=== Analyzing Grid Connectivity ===")
    total_grids_analyzed = 0
    total_grids_with_issues = 0

    for grid in grids:
        roads = grid.get('roads', [])
        grid_id = grid['id']

        if len(roads) <= 1:
            # Single road or empty grid, nothing to analyze
            continue

        # Skip very large grids to avoid timeout
        if len(roads) > 2000:
            print(f"Grid {grid_id}: Skipping connectivity analysis (too many roads: {len(roads)})")
            continue

        # Build connectivity graph for THIS GRID ONLY
        endpoint_to_roads = {}

        for idx, road in enumerate(roads):
            waypoints = road.get('waypoints', [])
            if len(waypoints) < 2:
                continue

            start = normalize_point(waypoints[0]['lat'], waypoints[0]['lon'])
            end = normalize_point(waypoints[-1]['lat'], waypoints[-1]['lon'])

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

            # BFS to find all roads connected to this one
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

                start = normalize_point(waypoints[0]['lat'], waypoints[0]['lon'])
                end = normalize_point(waypoints[-1]['lat'], waypoints[-1]['lon'])

                # Find connected roads
                for endpoint in [start, end]:
                    for connected_idx in endpoint_to_roads.get(endpoint, set()):
                        if connected_idx not in visited:
                            queue.append(connected_idx)

            if component:
                components.append(component)

        # Analyze connectivity
        total_grids_analyzed += 1

        if len(components) > 1:
            # Multiple components = disconnected roads
            largest = max(components, key=len)
            largest_size = len(largest)

            print(f"\nGrid {grid_id}: Found {len(components)} disconnected components:")
            print(f"  Largest component: {largest_size} roads")

            # Analyze each smaller component
            for comp_idx, component in enumerate(components):
                if component == largest:
                    continue

                component_size = len(component)
                component_road_ids = [roads[idx]['id'] for idx in component]
                component_road_names = set([str(roads[idx]['name']) for idx in component if roads[idx].get('name')])

                print(f"  Component {comp_idx + 1}: {component_size} roads - {', '.join(list(component_road_names)[:3]) if component_road_names else 'unnamed'}")

                # Check if these roads appear in adjacent grids
                adjacent_grids = get_adjacent_grids(grid, grids)
                roads_found_elsewhere = {}

                for road_id in component_road_ids:
                    other_grids = find_road_in_grids(road_id, adjacent_grids)
                    if other_grids:
                        roads_found_elsewhere[road_id] = other_grids

                if roads_found_elsewhere:
                    print(f"    → {len(roads_found_elsewhere)} roads also appear in adjacent grids {set([g for grids_list in roads_found_elsewhere.values() for g in grids_list])}")
                    print(f"    → These roads may be connected in adjacent grids (check overlap)")
                else:
                    print(f"    → Roads NOT found in adjacent grids - grid extension may be needed")
                    total_grids_with_issues += 1
        else:
            # All roads connected!
            print(f"Grid {grid_id}: All {len(roads)} roads connected ✓")

    print(f"\n=== Connectivity Analysis Summary ===")
    print(f"Grids analyzed: {total_grids_analyzed}")
    print(f"Grids with connectivity issues: {total_grids_with_issues}")

    return grids

async def run_connectivity_analysis_background(search_id: str, grids: List[Dict], roads_list: List[Dict]):
    """
    Background task to analyze connectivity for all grids.
    Checks each grid, identifies disconnected roads, checks adjacent grids, and extends if needed.
    """
    try:
        print(f"\n=== Background Connectivity Analysis Started for {search_id} ===")

        # Create table if doesn't exist
        await db.create_connectivity_table()

        # Update status to running
        await db.update_connectivity_status(search_id, 'running')

        def normalize_point(lat, lon):
            """Round to 6 decimals (~0.1 meter precision)"""
            return (round(lat, 6), round(lon, 6))

        def get_adjacent_grids(grid, all_grids, threshold_miles=0.5):
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

        def find_road_in_grids(road_id, grid_list):
            """Find which grids contain a specific road ID"""
            found_in = []
            for grid in grid_list:
                for road in grid.get('roads', []):
                    if road['id'] == road_id:
                        found_in.append(grid['id'])
                        break
            return found_in

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
                start = normalize_point(waypoints[0]['lat'], waypoints[0]['lon'])
                end = normalize_point(waypoints[-1]['lat'], waypoints[-1]['lon'])
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
                    start = normalize_point(waypoints[0]['lat'], waypoints[0]['lon'])
                    end = normalize_point(waypoints[-1]['lat'], waypoints[-1]['lon'])
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
                    adjacent_grids = get_adjacent_grids(grid, grids)
                    roads_found_elsewhere = {}
                    for road_id in component_road_ids:
                        other_grids = find_road_in_grids(road_id, adjacent_grids)
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

        print(f"=== Background Analysis Complete for {search_id} ===")
        print(f"Analyzed: {total_analyzed} grids")
        print(f"Needing extension: {total_needing_extension} grids")

    except Exception as e:
        print(f"ERROR in background connectivity analysis: {str(e)}")
        await db.update_connectivity_status(search_id, 'failed')

async def process_search_grids(search_id: str, request: SearchRequest):
    """Background task to process grid generation (runs after response sent)"""
    # ADD FILE LOGGING since systemd journal doesn't capture prints
    import sys
    log_file = open(f'/tmp/grid_gen_{search_id}.log', 'w', buffering=1)
    sys.stdout = log_file
    sys.stderr = log_file

    try:
        print(f"[{search_id}] Starting grid generation...")

        # Update status to processing
        await db.execute(
            "UPDATE pet_searches SET status = ? WHERE search_id = ?",
            ['processing', search_id]
        )

        print(f"Downloading OSM data for {request.lat}, {request.lon}")
        print(f"Grid size: {request.grid_size_miles} miles")

        # Smart completion: Add 500ft (0.095 miles) buffer to download radius
        # This ensures we get connecting road segments for completion
        completion_buffer_miles = 0.095
        download_radius_miles = request.radius_miles + completion_buffer_miles

        # Use OVERPASS API to get comprehensive road network (FAST & RELIABLE!)
        print(f"[OVERPASS] Querying OSM data near ({request.lat}, {request.lon})...")
        print(f"Download radius: {download_radius_miles:.2f} miles (includes 500ft completion buffer)")

        # Convert miles to meters for Overpass API
        radius_meters = int(download_radius_miles * 1609.34)

        # Try multiple Overpass servers with fallback to OSMnx
        overpass_servers = [
            "https://overpass-api.de/api/interpreter",  # Default (main)
            "https://overpass.kumi.systems/api/interpreter",  # Alternative 1
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",  # Alternative 2
        ]

        result = None
        last_error = None

        for server_url in overpass_servers:
            try:
                print(f"[OVERPASS] Trying server: {server_url}")
                api = overpy.Overpass(url=server_url)
                query = f"""
[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  way["highway"](around:{radius_meters},{request.lat},{request.lon});
  >;
);
out body;
"""
                print(f"[OVERPASS] Querying roads within {radius_meters}m radius...")
                result = api.query(query)
                print(f"[OVERPASS] SUCCESS! Got {len(result.ways)} ways and {len(result.nodes)} nodes")
                break  # Success, exit loop
            except Exception as e:
                last_error = e
                print(f"[OVERPASS] Server {server_url} failed: {str(e)}")
                continue  # Try next server

        # If all Overpass servers failed, fall back to OSMnx
        if result is None:
            print(f"[FALLBACK] All Overpass servers failed, falling back to OSMnx...")
            print(f"[FALLBACK] Last error was: {str(last_error)}")

            try:
                # Use OSMnx to get the road network
                print(f"[OSMnx] Downloading road network from OpenStreetMap...")
                G = ox.graph_from_point(
                    (request.lat, request.lon),
                    dist=download_radius_miles * 1609.34,  # Convert miles to meters
                    network_type='drive',
                    simplify=False
                )
                print(f"[OSMnx] SUCCESS! Got graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

                # Convert OSMnx graph format to match Overpass result format for consistency
                # We'll skip the Overpass result processing and go straight to using the graph
                result = None  # Signal to skip Overpass processing

            except Exception as osmx_error:
                print(f"[OSMnx] FAILED: {str(osmx_error)}")
                raise Exception(f"All data sources failed. Overpass: {str(last_error)}, OSMnx: {str(osmx_error)}")

        # Only process Overpass result if we got one (otherwise G is already set by OSMnx)
        if result is not None:
            print(f"[OVERPASS] Processing {len(result.ways)} ways and {len(result.nodes)} nodes")

            # Build NetworkX graph from Overpass result
            G = nx.MultiDiGraph()

            # Create node lookup
            node_coords = {}
            for node in result.nodes:
                node_coords[node.id] = (node.lat, node.lon)
                G.add_node(node.id, y=node.lat, x=node.lon)

            # Add edges from ways
            for way in result.ways:
                way_nodes = [node.id for node in way.nodes]
                highway_type = way.tags.get('highway', 'unknown')

                # Check if road is one-way (default is bidirectional)
                oneway = way.tags.get('oneway', 'no')
                is_oneway = oneway in ['yes', 'true', '1', '-1']

                # Add edges between consecutive nodes
                for i in range(len(way_nodes) - 1):
                    u, v = way_nodes[i], way_nodes[i+1]
                    if u in node_coords and v in node_coords:
                        # Calculate edge geometry and length
                        u_coord = node_coords[u]
                        v_coord = node_coords[v]
                        line = LineString([
                            (u_coord[1], u_coord[0]),  # (lon, lat)
                            (v_coord[1], v_coord[0])
                        ])
                        # Calculate length in meters using geopy (lat/lon degrees to meters)
                        length_meters = geodesic(u_coord, v_coord).meters

                        # Add forward edge
                        G.add_edge(u, v, highway=highway_type, geometry=line, length=length_meters, name=way.tags.get('name', ''))

                        # Add reverse edge for bidirectional roads (search both sides)
                        if not is_oneway:
                            reverse_line = LineString([
                                (v_coord[1], v_coord[0]),  # (lon, lat)
                                (u_coord[1], u_coord[0])
                            ])
                            G.add_edge(v, u, highway=highway_type, geometry=reverse_line, length=length_meters, name=way.tags.get('name', ''))

            print(f"[OVERPASS] Created NetworkX graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        else:
            print(f"[OSMnx] Using graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # DEBUG: Check a sample edge for length
        sample_edges = list(G.edges(data=True))[:3]
        for u, v, data in sample_edges:
            print(f"[DEBUG] Sample edge {u}->{v}: length={data.get('length', 'MISSING')}, highway={data.get('highway', 'MISSING')}")

        # OPTIMIZED: Extract edges directly from graph instead of slow graph_to_gdfs()
        # This bypasses OSMnx's comprehensive (but slow) GeoDataFrame conversion
        print(f"Extracting {G.number_of_edges()} edges from graph...")

        # Filter fake diagonals and prepare roads
        roads = []
        filtered_count = 0
        kept_count = 0
        edge_idx = 0
        seen_edges = set()  # Track unique edge pairs to avoid duplicates from bidirectional edges

        # Iterate directly over graph edges (much faster than graph_to_gdfs)
        print(f"[DEBUG] Starting edge iteration loop...")
        for u, v, key, data in G.edges(keys=True, data=True):
            # Skip duplicate edges (we added bidirectional edges for connectivity, but only want unique roads)
            edge_pair = tuple(sorted([u, v]))  # Normalize (u,v) and (v,u) to same key
            if edge_pair in seen_edges:
                continue
            seen_edges.add(edge_pair)

            # Progress tracking every 5000 edges
            if edge_idx > 0 and edge_idx % 5000 == 0:
                print(f"[DEBUG] Processed {edge_idx} edges so far, kept {kept_count}, filtered {filtered_count}")

            # Get or create geometry
            if 'geometry' in data:
                geom = data['geometry']
            else:
                # Create LineString from node coordinates (using top-level import)
                u_coords = (G.nodes[u]['x'], G.nodes[u]['y'])
                v_coords = (G.nodes[v]['x'], G.nodes[v]['y'])
                geom = LineString([u_coords, v_coords])

            coords = list(geom.coords)

            # Check if this is a fake diagonal (PROVEN LOGIC)
            if is_definitely_fake(data, coords, data):
                filtered_count += 1
                continue

            kept_count += 1

            # Get street name
            street_name = data.get('name', None)
            if isinstance(street_name, list):
                street_name = street_name[0] if street_name else None
            if not street_name:
                street_name = f"Road_{kept_count}"

            # Create waypoints
            waypoints = []
            for lon, lat in coords:
                if not (math.isnan(lat) or math.isnan(lon) or math.isinf(lat) or math.isinf(lon)):
                    waypoints.append({'lat': lat, 'lon': lon})

            if len(waypoints) < 2:
                continue

            # Get properties
            highway_type = data.get('highway', 'unknown')
            if isinstance(highway_type, list):
                highway_type = highway_type[0] if highway_type else 'unknown'

            length = data.get('length', 0)
            if not isinstance(length, (int, float)) or math.isnan(length) or math.isinf(length):
                length = 0

            roads.append({
                'id': f"{search_id}-{edge_idx}",
                'name': street_name,
                'waypoints': waypoints,
                'geometry': geom,
                'highway_type': highway_type,
                'length_meters': round(length, 1),
                'has_name': bool(data.get('name'))
            })
            edge_idx += 1

        print(f"Extracted {len(roads)} edges from graph (filtered {filtered_count} fake diagonals)")

        # DEBUG: Check road lengths
        total_meters = sum(r['length_meters'] for r in roads)
        avg_meters = total_meters / len(roads) if roads else 0
        zero_length_roads = [r for r in roads if r['length_meters'] == 0]
        print(f"[DEBUG] Total road length: {total_meters:.1f}m ({total_meters/1609.34:.2f} miles), avg per road: {avg_meters:.1f}m")
        if zero_length_roads:
            print(f"[DEBUG] WARNING: {len(zero_length_roads)} roads have ZERO length!")
        sample_roads = roads[:3]
        for road in sample_roads:
            print(f"[DEBUG] Sample road: name={road['name']}, length={road['length_meters']}m")

        # NEW APPROACH: Build grids based on road mileage (not fixed geographic size)
        # This ensures each grid has ~6 miles of roads (30 min at 12mph)
        # regardless of area density
        grids = create_grids_from_roads(
            roads,
            request.lat,
            request.lon,
            target_miles=6,
            min_miles=2,  # Lowered from 4 to capture urban areas with disconnected segments
            max_miles=8
        )

        # DISABLED: Connectivity analysis - causes worker timeout on large searches
        # TODO: Optimize or move to background task
        # grids = analyze_grid_connectivity(grids, roads)

        # DISABLED: Connectivity filtering - has critical flaw with grid processing order
        # TODO: Fix to process adjacent grids correctly, not just by distance
        # print("\nApplying connectivity filtering...")
        # grids = filter_connected_roads_sequential(grids, request.lat, request.lon)

        # Prepare response
        tiles = []
        for grid in grids:
            total_miles = sum(r['length_meters'] for r in grid['roads']) / 1609.34

            # Keep roads separate with their own waypoints
            road_details = []
            for road in grid['roads']:
                road_details.append({
                    'id': road['id'],
                    'name': road['name'],
                    'waypoints': road['waypoints'],
                    'highway_type': road['highway_type'],
                    'length_meters': road['length_meters'],
                    'has_name': road['has_name']
                })

            tiles.append({
                'id': f"{search_id}-grid-{grid['id']}",
                'grid_id': grid['id'],
                'road_count': len(grid['roads']),
                'total_distance_miles': round(total_miles, 3),
                'estimated_minutes': max(5, int(total_miles * 15)),  # ~4 mph walking speed
                'bounds': {
                    'min_lat': grid['min_lat'],
                    'max_lat': grid['max_lat'],
                    'min_lon': grid['min_lon'],
                    'max_lon': grid['max_lon']
                },
                'center': {
                    'lat': grid['center_lat'],
                    'lon': grid['center_lon']
                },
                'road_details': road_details,
                'grid_size_miles': request.grid_size_miles
            })

        # Filter out empty grids
        tiles = [t for t in tiles if t['road_count'] > 0]

        # Clean any NaN/Infinity values
        def clean_nan(obj):
            if isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(item) for item in obj]
            elif isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            return obj

        result = {
            'search_id': search_id,
            'center': {'lat': request.lat, 'lon': request.lon},
            'tiles': tiles,
            'total_tiles': len(tiles),
            'total_roads': kept_count,
            'filtered_count': filtered_count,
            'grid_size_miles': request.grid_size_miles,
            'message': f'Created {len(tiles)} geographic grids with {kept_count} real roads'
        }

        # Clean the result to remove any NaN/Infinity values
        cleaned_result = clean_nan(result)

        # Update tracking database with total_grids and status='active'
        if request.pet_id:
            try:
                # Update the existing record (created with status='pending' in endpoint)
                await db.execute(
                    "UPDATE pet_searches SET total_grids = ?, status = ? WHERE search_id = ?",
                    [len(tiles), 'active', search_id]
                )
                print(f"[{search_id}] Updated search tracking: {len(tiles)} grids, status=active")

                # Save grid data for later retrieval by iOS app
                for tile in tiles:
                    await db.save_grid_data(search_id, tile['grid_id'], tile)
                print(f"[{search_id}] Saved {len(tiles)} grid tiles for retrieval")

            except Exception as tracking_error:
                print(f"[{search_id}] Error updating search tracking: {str(tracking_error)}")
                # Don't fail the whole request if tracking fails

        # Save to database in background (non-blocking)
        import asyncio

        async def save_to_database_background():
            """Save to database without blocking the response"""
            try:
                search_data = {
                    'search_id': search_id,
                    'center': {'lat': request.lat, 'lon': request.lon},
                    'radius_miles': request.radius_miles,
                    'address': request.address or '',
                    'total_tiles': len(tiles),
                    'filtered_count': filtered_count
                }

                # Convert roads to tile format for database (each road as a tile)
                road_tiles = []
                for road in roads:
                    road_tiles.append({
                        'id': road['id'],
                        'roads': [road['name']],
                        'waypoints': road['waypoints'],
                        'highway_type': road['highway_type'],
                        'length_meters': road['length_meters'],
                        'point_count': len(road['waypoints']),
                        'has_name': road['has_name'],
                        'total_distance_miles': round(road['length_meters'] / 1609.34, 3),
                        'estimated_minutes': max(1, int(road['length_meters'] / 50)) if road['length_meters'] > 0 else 1
                    })

                db_stats = await db.save_search_results(search_data, road_tiles)
                print(f"Database: Saved {db_stats['new_roads']} new roads, {db_stats['existing_roads']} existing roads")
            except Exception as db_error:
                print(f"Database error: {str(db_error)}")

        # Update status to active (grid generation complete)
        await db.execute(
            "UPDATE pet_searches SET status = ?, total_grids = ? WHERE search_id = ?",
            ['active', len(tiles), search_id]
        )

        print(f"[{search_id}] Grid generation COMPLETE - {len(tiles)} grids created, status=active")

        # Background tasks are handled by separate worker process (connectivity_worker.py)
        # The connectivity worker will pick up searches with connectivity_status='pending'

        return {
            'search_id': search_id,
            'total_grids': len(tiles),
            'status': 'active',
            'message': f'Grid generation complete - {len(tiles)} grids created'
        }

    except Exception as e:
        print(f"[{search_id}] Grid generation FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        # Update status to failed
        await db.execute(
            "UPDATE pet_searches SET status = ? WHERE search_id = ?",
            ['failed', search_id]
        )

@app.post("/api/create-search", dependencies=[Depends(verify_api_key)])
async def create_search(request: SearchRequest):
    """Accept search request and return immediately, process grids in background.

    If force_regenerate=True and pet_id already exists:
    - Updates the search center to new coordinates
    - Deletes all old grids
    - Regenerates grids at new location (e.g., after confirmed sighting)
    """
    try:
        import asyncio

        # Check if search already exists for this pet_id
        existing_search = None
        if request.pet_id:
            existing_search = await db.get_search_by_pet_id(request.pet_id)

        if existing_search and request.force_regenerate:
            # FORCE REGENERATE: Update existing search with new location
            search_id = existing_search['search_id']
            old_lat = existing_search.get('center_lat')
            old_lon = existing_search.get('center_lon')

            print(f"[{search_id}] FORCE REGENERATE for pet_id={request.pet_id}")
            print(f"[{search_id}] Old location: ({old_lat}, {old_lon})")
            print(f"[{search_id}] New location: ({request.lat}, {request.lon})")

            # Delete old grids
            delete_result = await db.execute(
                "DELETE FROM search_grids WHERE search_id = ?",
                [search_id]
            )
            print(f"[{search_id}] Deleted old grids")

            # Update search record with new center coordinates and reset status
            await db.execute(
                """UPDATE pet_searches
                   SET center_lat = ?, center_lon = ?, address = ?,
                       radius_miles = ?, status = 'pending', total_grids = 0
                   WHERE search_id = ?""",
                [request.lat, request.lon,
                 request.address or f"{request.lat}, {request.lon}",
                 request.radius_miles, search_id]
            )
            print(f"[{search_id}] Updated search record with new coordinates")

            # Spawn background task to regenerate grids
            asyncio.create_task(process_search_grids(search_id, request))

            print(f"[{search_id}] Background grid regeneration started")

            return {
                'success': True,
                'search_id': search_id,
                'status': 'pending',
                'force_regenerated': True,
                'old_location': {'lat': old_lat, 'lon': old_lon},
                'new_location': {'lat': request.lat, 'lon': request.lon},
                'message': 'Search recentered, grids are being regenerated at new location'
            }

        elif existing_search and not request.force_regenerate:
            # Search exists but force_regenerate not set - return existing search info
            print(f"[{existing_search['search_id']}] Search already exists for pet_id={request.pet_id}, force_regenerate=False")
            return {
                'success': True,
                'search_id': existing_search['search_id'],
                'status': existing_search.get('status', 'unknown'),
                'already_exists': True,
                'message': 'Search already exists for this pet. Use force_regenerate=true to recenter at new location.'
            }

        else:
            # NEW SEARCH: Create new search record
            search_id = f"search-{uuid.uuid4()}"

            print(f"[{search_id}] Received search request for pet_id={request.pet_id}")

            # Save basic search record immediately with status='pending'
            await db.create_search_tracking(
                search_id,
                request.pet_id,
                request.address or f"{request.lat}, {request.lon}",
                request.lat,
                request.lon,
                request.radius_miles,
                0  # total_grids will be updated when processing completes
            )

            print(f"[{search_id}] Saved to database with status='pending'")

            # Spawn background task to process grids (fire and forget)
            asyncio.create_task(process_search_grids(search_id, request))

            print(f"[{search_id}] Background grid processing started")

            # Return success immediately
            return {
                'success': True,
                'search_id': search_id,
                'status': 'pending',
                'message': 'Search accepted, grids are being generated in background'
            }

    except Exception as e:
        print(f"Error creating search: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/database-stats")
async def get_database_stats():
    """Get overall database statistics"""
    try:
        stats = await db.get_database_statistics()
        return {
            'success': True,
            'statistics': stats
        }
    except Exception as e:
        print(f"Error getting database stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/roads/nearby")
async def get_nearby_roads(lat: float, lon: float, radius: float = 0.25, limit: int = 100):
    """
    Get roads near a location from database (for iOS apps)

    Parameters:
    - lat: Latitude
    - lon: Longitude
    - radius: Search radius in miles (default 0.25)
    - limit: Max number of roads to return (default 100)

    Example: /api/roads/nearby?lat=27.8428&lon=-82.8106&radius=0.5
    """
    try:
        roads = await db.get_roads_in_area(lat, lon, radius, limit)
        return {
            'success': True,
            'location': {'lat': lat, 'lon': lon},
            'radius_miles': radius,
            'roads_found': len(roads),
            'roads': roads
        }
    except Exception as e:
        print(f"Error getting nearby roads: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/roads/{road_id}")
async def get_road_detail(road_id: str):
    """
    Get detailed information about a specific road including waypoints

    Example: /api/roads/abc123def456
    """
    try:
        road = await db.get_road_with_waypoints(road_id)
        if not road:
            raise HTTPException(status_code=404, detail="Road not found")

        return {
            'success': True,
            'road': road
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting road detail: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ========================================
# SEARCH TRACKING API ENDPOINTS
# ========================================

class GetGridsRequest(BaseModel):
    search_id: str

class AssignGridRequest(BaseModel):
    search_id: str
    pet_id: str
    searcher_id: str
    searcher_name: str
    timeframe_minutes: int  # 30, 60, 90, or 120

class UpdateProgressRequest(BaseModel):
    assignment_id: str
    search_id: str
    pet_id: str
    grid_id: int
    searcher_id: str
    lat: float
    lon: float
    accuracy_meters: Optional[float] = None
    roads_covered: Optional[List[Dict]] = None  # [{road_id, road_name}, ...]
    distance_miles: Optional[float] = None  # Cumulative distance traveled in miles
    elapsed_minutes: Optional[int] = None  # Cumulative time elapsed in minutes

class PetDetailsRequest(BaseModel):
    pet_id: str
    pet_name: str
    pet_photo_url: Optional[str] = None

class CompleteSearchRequest(BaseModel):
    search_id: str
    pet_id: str
    searcher_id: str
    total_distance_miles: float
    duration_minutes: int

class UpdateSearchCenterRequest(BaseModel):
    search_id: str
    latitude: float
    longitude: float

@app.post("/api/get-grids", dependencies=[Depends(verify_api_key)])
async def get_grids(request: GetGridsRequest):
    """
    Get grids for an existing search by search_id
    Returns all grid details with their bounds and roads
    """
    try:
        # This would need to retrieve stored grid data
        # For now, we return the grid status
        grid_status = await db.get_grid_status(request.search_id)

        return {
            'success': True,
            'search_id': request.search_id,
            'grids': grid_status
        }
    except Exception as e:
        print(f"Error getting grids: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/assign-grid", dependencies=[Depends(verify_api_key)])
async def assign_grid_endpoint(request: AssignGridRequest):
    """
    Assign grid(s) to a searcher based on their timeframe
    - 30 min = 1 grid
    - 60 min = 2 grids
    - 90 min = 3 grids
    - 120 min = 4 grids

    Grids are assigned closest to center first (grid1, grid2, grid3...)
    """
    try:
        # Validate timeframe
        if request.timeframe_minutes not in [30, 60, 90, 120]:
            raise HTTPException(status_code=400, detail="Timeframe must be 30, 60, 90, or 120 minutes")

        # Calculate number of grids to assign
        num_grids = request.timeframe_minutes // 30

        # Get search details to check status and total grids
        search_sql = "SELECT status, total_grids FROM pet_searches WHERE search_id = ?"
        search_result = await db.execute(search_sql, [request.search_id])

        if not search_result.get('results') or len(search_result['results']) == 0:
            raise HTTPException(status_code=404, detail="Search not found")

        # Check if grids are still being generated
        status = search_result['results'][0].get('status', 'pending')
        total_grids = search_result['results'][0].get('total_grids', 0)

        if status == 'pending':
            return {
                'success': False,
                'error': 'grids_not_ready',
                'message': 'Grid generation in progress, please retry in a few seconds',
                'retry_after_seconds': 5,
                'search_id': request.search_id,
                'status': 'pending'
            }

        # Get available grids
        available_grids = await db.get_available_grids(request.search_id, total_grids)

        # Handle race condition: status='active' but grids not saved yet
        if len(available_grids) == 0 and status == 'active':
            return {
                'success': False,
                'error': 'grids_not_ready',
                'message': 'Grid generation in progress, please retry in a few seconds',
                'retry_after_seconds': 5,
                'search_id': request.search_id,
                'status': 'pending'
            }

        if len(available_grids) < num_grids:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough available grids. Requested: {num_grids}, Available: {len(available_grids)}"
            )

        # Assign the requested number of grids
        assignments = []
        for i in range(num_grids):
            grid_id = available_grids[i]
            assignment = await db.assign_grid(
                request.search_id,
                request.pet_id,
                grid_id,
                request.searcher_id,
                request.searcher_name,
                request.timeframe_minutes
            )
            assignments.append(assignment)

        return {
            'success': True,
            'assignments': assignments,
            'total_assigned': len(assignments),
            'timeframe_minutes': request.timeframe_minutes
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error assigning grid: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/update-progress", dependencies=[Depends(verify_api_key)])
async def update_progress_endpoint(request: UpdateProgressRequest):
    """
    Update search progress with GPS tracking and roads covered
    Called every 30 seconds by iOS app
    """
    try:
        # Save GPS tracking point
        progress_id = await db.update_search_progress(
            request.assignment_id,
            request.search_id,
            request.pet_id,
            request.grid_id,
            request.searcher_id,
            request.lat,
            request.lon,
            request.accuracy_meters,
            request.distance_miles,
            request.elapsed_minutes
        )

        # Mark roads as searched if provided
        roads_marked = []
        if request.roads_covered:
            for road in request.roads_covered:
                road_search_id = await db.mark_road_searched(
                    request.assignment_id,
                    request.search_id,
                    request.pet_id,
                    request.grid_id,
                    request.searcher_id,
                    road['road_id'],
                    road['road_name']
                )
                roads_marked.append(road_search_id)

        # Calculate grid completion percentage
        # Note: We need total roads in grid - this should come from grid data
        # For now, we'll calculate it based on what's been searched
        # TODO: Store total_roads_in_grid when creating search

        return {
            'success': True,
            'progress_id': progress_id,
            'roads_marked': len(roads_marked),
            'message': 'Progress updated successfully'
        }

    except Exception as e:
        print(f"Error updating progress: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/grid-status", dependencies=[Depends(verify_api_key)])
async def get_grid_status_endpoint(search_id: str):
    """
    Get current status of all grids for a search
    Shows which grids are assigned, completed, or available
    """
    try:
        grid_status = await db.get_grid_status(search_id)

        return {
            'success': True,
            'search_id': search_id,
            'grids': grid_status,
            'total_grids': len(grid_status)
        }

    except Exception as e:
        print(f"Error getting grid status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search-history", dependencies=[Depends(verify_api_key)])
async def get_search_history_endpoint(searcher_id: str):
    """
    Get search history for a specific searcher
    Returns all completed search sessions with pet details and GPS routes

    Query parameter:
    - searcher_id: Unique identifier for the volunteer searcher

    Returns array of search sessions with:
    - search_id, pet_id, pet_name, pet_photo_url
    - searched_at (timestamp when started)
    - completed_at (timestamp when completed)
    - total_distance_miles (cumulative distance traveled)
    - duration_minutes (total time spent searching)
    - route (array of GPS points with lat/lon)
    """
    try:
        search_history = await db.get_search_history(searcher_id)

        return {
            'success': True,
            'searcher_id': searcher_id,
            'total_searches': len(search_history),
            'searches': search_history
        }

    except Exception as e:
        print(f"Error getting search history: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/save-pet-details", dependencies=[Depends(verify_api_key)])
async def save_pet_details_endpoint(request: PetDetailsRequest):
    """
    Save or update lost pet details (name and photo URL)

    Called by iOS app when user creates or updates a lost pet report

    Request body:
    - pet_id: Unique identifier for the pet
    - pet_name: Name of the lost pet
    - pet_photo_url: URL to pet's photo (optional)
    """
    try:
        pet_id = await db.save_lost_pet(
            request.pet_id,
            request.pet_name,
            request.pet_photo_url
        )

        return {
            'success': True,
            'pet_id': pet_id,
            'message': 'Pet details saved successfully'
        }

    except Exception as e:
        print(f"Error saving pet details: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/update-search-center", dependencies=[Depends(verify_api_key)])
async def update_search_center_endpoint(request: UpdateSearchCenterRequest):
    """
    Update the center point used for grid assignment prioritization.

    Called when a sighting is confirmed within the existing search radius.
    Does NOT regenerate grids - just changes which grids are assigned first.

    Request body:
    - search_id: The search to update
    - latitude: New center latitude (e.g., confirmed sighting location)
    - longitude: New center longitude
    """
    try:
        # Get current center for response
        sql = "SELECT center_lat, center_lon FROM pet_searches WHERE search_id = ?"
        result = await db.execute(sql, [request.search_id])

        if not result.get('results') or len(result['results']) == 0:
            raise HTTPException(status_code=404, detail=f"Search not found: {request.search_id}")

        previous = result['results'][0]
        previous_center = {
            "lat": previous['center_lat'],
            "lon": previous['center_lon']
        }

        # Update the center point
        update_sql = "UPDATE pet_searches SET center_lat = ?, center_lon = ? WHERE search_id = ?"
        await db.execute(update_sql, [request.latitude, request.longitude, request.search_id])

        print(f"[{request.search_id}] Updated search center from ({previous_center['lat']}, {previous_center['lon']}) to ({request.latitude}, {request.longitude})")

        return {
            'success': True,
            'search_id': request.search_id,
            'previous_center': previous_center,
            'new_center': {
                "lat": request.latitude,
                "lon": request.longitude
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating search center: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/complete-search", dependencies=[Depends(verify_api_key)])
async def complete_search_endpoint(request: CompleteSearchRequest):
    """
    Mark a search as completed with final statistics

    Called by iOS app when user finishes searching

    Request body:
    - search_id: Unique identifier for the search
    - pet_id: Unique identifier for the pet
    - searcher_id: Unique identifier for the searcher
    - total_distance_miles: Total distance traveled during search
    - duration_minutes: Total time spent searching in minutes

    This marks the search session as completed so it appears in /api/search-history
    """
    try:
        result = await db.complete_search(
            request.search_id,
            request.pet_id,
            request.searcher_id,
            request.total_distance_miles,
            request.duration_minutes
        )

        return {
            'success': True,
            'message': 'Search completed and added to history',
            'assignment_id': result['assignment_id'],
            'total_distance_miles': result['total_distance_miles'],
            'duration_minutes': result['duration_minutes']
        }

    except Exception as e:
        print(f"Error completing search: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/all-searches")
async def get_all_searches_endpoint():
    """
    Get all search locations ever created
    Returns search locations from both searches and pet_searches tables

    No API key required - read-only public endpoint for map display
    """
    try:
        # Get searches from both tables
        searches_old = await db.execute('''
            SELECT
                id as search_id,
                address,
                center_lat,
                center_lon,
                radius_miles,
                created_at
            FROM searches
            ORDER BY created_at DESC
        ''', [])

        searches_new = await db.execute('''
            SELECT
                search_id,
                address,
                center_lat,
                center_lon,
                radius_miles,
                created_at,
                status
            FROM pet_searches
            ORDER BY created_at DESC
        ''', [])

        all_searches = []

        # Combine results from both tables
        if searches_old.get('results'):
            all_searches.extend(searches_old['results'])

        if searches_new.get('results'):
            all_searches.extend(searches_new['results'])

        return {
            'success': True,
            'searches': all_searches,
            'total_searches': len(all_searches)
        }

    except Exception as e:
        print(f"Error getting all searches: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/active-searchers")
async def get_active_searchers_endpoint(search_id: str = 'all'):
    """
    Get all active searchers with their latest GPS positions
    Real-time tracking for coordinator view

    No API key required - read-only public endpoint for tracking display
    If search_id is 'all' or not provided, returns all active searchers across all searches
    """
    try:
        searchers = await db.get_active_searchers_with_positions(search_id)

        return {
            'success': True,
            'search_id': search_id,
            'active_searchers': searchers,
            'total_active': len(searchers)
        }

    except Exception as e:
        print(f"Error getting active searchers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def sanitize_float_values(obj):
    """
    Recursively sanitize NaN and Infinity float values to None for JSON serialization
    """
    import math

    if isinstance(obj, dict):
        return {k: sanitize_float_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_float_values(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    else:
        return obj

@app.get("/api/get-grid", dependencies=[Depends(verify_api_key)])
async def get_grid_endpoint(search_id: str, grid_id: int):
    """
    Get a single grid's full road details

    Used by iOS app to retrieve detailed road information for assigned grids

    Query Parameters:
    - search_id: The search identifier
    - grid_id: The grid number (1, 2, 3, etc.)

    Returns:
    Full grid tile object including all road details with waypoints
    """
    try:
        grid_data = await db.get_grid_data(search_id, grid_id)

        if not grid_data:
            raise HTTPException(
                status_code=404,
                detail=f"Grid {grid_id} not found for search {search_id}"
            )

        # FIX: Ensure 'id' field is a STRING (iOS expects string "1", not integer 1)
        # Original stored as string like "search-xxx-grid-1", iOS needs just the number as string
        grid_data['id'] = str(grid_id)

        # Sanitize NaN/Inf values for JSON serialization
        grid_data = sanitize_float_values(grid_data)

        return grid_data

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting grid data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search-by-pet", dependencies=[Depends(verify_api_key)])
async def get_search_by_pet_id(pet_id: str):
    """
    Look up search_id by pet_id for iOS app

    Query Parameters:
    - pet_id: The Airtable pet record ID (e.g., "84")

    Returns:
    {
        "success": true,
        "pet_id": "84",
        "search_id": "search-63ef94cd-ce99-479f-ac13-a26b8f52b419",
        "total_grids": 151,
        "created_at": 1761313770,
        "status": "active"
    }
    """
    try:
        sql = "SELECT search_id, total_grids, created_at, status FROM pet_searches WHERE pet_id = ? ORDER BY created_at DESC LIMIT 1"
        result = await db.execute(sql, [pet_id])

        if not result.get('results') or len(result['results']) == 0:
            raise HTTPException(status_code=404, detail=f"No search found for pet_id: {pet_id}")

        search = result['results'][0]
        return {
            "success": True,
            "pet_id": pet_id,
            "search_id": search['search_id'],
            "total_grids": search['total_grids'],
            "created_at": search['created_at'],
            "status": search['status']
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error looking up search by pet_id: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search-stats", dependencies=[Depends(verify_api_key)])
async def get_search_stats_endpoint(search_id: str):
    """
    Get aggregate statistics for all searchers who searched for this pet

    Query Parameters:
    - search_id: The search identifier

    Returns:
    {
        "search_id": "search-XXX",
        "total_searchers": 3,
        "total_distance_miles": 8.5,
        "total_hours": 2.5,
        "total_grids_assigned": 5,
        "total_grids_completed": 2
    }
    """
    try:
        stats = await db.get_search_stats(search_id)
        return stats
    except Exception as e:
        print(f"Error getting search stats: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/retry-search", dependencies=[Depends(verify_api_key)])
async def retry_failed_search(pet_id: Optional[str] = None, search_id: Optional[str] = None):
    """
    Retry a failed search by pet_id or search_id

    This endpoint allows retrying grid generation for searches that failed
    (e.g., due to Overpass API timeouts or other transient errors)

    Query Parameters (provide one):
    - pet_id: The pet ID (e.g., "119")
    - search_id: The search identifier (e.g., "search-XXX")

    Returns:
    {
        "success": true,
        "search_id": "search-XXX",
        "pet_id": "119",
        "status": "pending",
        "message": "Search retry initiated, grids are being regenerated"
    }
    """
    try:
        # Validate that at least one parameter is provided
        if not pet_id and not search_id:
            raise HTTPException(
                status_code=400,
                detail="Must provide either pet_id or search_id"
            )

        # Look up the search
        if pet_id:
            sql = "SELECT search_id, pet_id, center_lat, center_lon, radius_miles, grid_size_miles, address, status FROM pet_searches WHERE pet_id = ? ORDER BY created_at DESC LIMIT 1"
            result = await db.execute(sql, [pet_id])
        else:
            sql = "SELECT search_id, pet_id, center_lat, center_lon, radius_miles, grid_size_miles, address, status FROM pet_searches WHERE search_id = ?"
            result = await db.execute(sql, [search_id])

        if not result.get('results') or len(result['results']) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No search found for {'pet_id: ' + pet_id if pet_id else 'search_id: ' + search_id}"
            )

        search = result['results'][0]
        search_id = search['search_id']
        current_status = search['status']

        print(f"[{search_id}] Retry requested for pet_id={search['pet_id']}, current status={current_status}")

        # Allow retry for failed, pending, or completed searches
        # (pending might be stuck, completed might want regeneration)
        if current_status not in ['failed', 'pending', 'completed']:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot retry search with status '{current_status}'. Only 'failed', 'pending', or 'completed' searches can be retried."
            )

        # Update status to pending
        await db.execute(
            "UPDATE pet_searches SET status = ?, total_grids = ? WHERE search_id = ?",
            ['pending', 0, search_id]
        )

        print(f"[{search_id}] Status updated to 'pending', restarting grid generation")

        # Recreate the SearchRequest object from stored data
        request = SearchRequest(
            lat=search['center_lat'],
            lon=search['center_lon'],
            radius_miles=search['radius_miles'],
            grid_size_miles=search.get('grid_size_miles', 0.3),
            address=search.get('address', ''),
            pet_id=search['pet_id']
        )

        # Spawn background task to reprocess grids
        import asyncio
        asyncio.create_task(process_search_grids(search_id, request))

        print(f"[{search_id}] Background grid processing restarted")

        return {
            'success': True,
            'search_id': search_id,
            'pet_id': search['pet_id'],
            'status': 'pending',
            'message': 'Search retry initiated, grids are being regenerated in background'
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error retrying search: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/all-pets")
async def get_all_pets():
    """Get all lost pets with their most recent search information"""
    try:
        sql = """
            SELECT
                lp.pet_id,
                lp.pet_name,
                lp.pet_photo_url,
                lp.created_at,
                lp.status,
                ps.search_id,
                ps.total_grids,
                ps.created_at as search_created_at
            FROM lost_pets lp
            LEFT JOIN pet_searches ps ON lp.pet_id = ps.pet_id
            ORDER BY lp.created_at DESC
        """
        result = await db.execute(sql)

        pets = result.get('results', [])

        return {
            'success': True,
            'pets': pets
        }
    except Exception as e:
        print(f"Error getting pets: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search-grids/{search_id}")
async def get_search_grids(search_id: str):
    """Get all grid data for a specific search"""
    try:
        # Get basic search info
        sql = """
            SELECT search_id, total_grids, created_at, status, pet_id
            FROM pet_searches
            WHERE search_id = ?
        """
        search_result = await db.execute(sql, [search_id])

        if not search_result.get('results') or len(search_result['results']) == 0:
            raise HTTPException(status_code=404, detail="Search not found")

        search_info = search_result['results'][0]

        # Get all grid_ids that exist for this search
        grid_ids_sql = """
            SELECT grid_id FROM search_grids
            WHERE search_id = ?
            ORDER BY grid_id
        """
        grid_ids_result = await db.execute(grid_ids_sql, [search_id])
        grid_ids = [row['grid_id'] for row in grid_ids_result.get('results', [])]

        # Get all grid data
        grids = []
        for grid_id in grid_ids:
            grid_data = await db.get_grid_data(search_id, grid_id)
            if grid_data:
                grids.append({
                    'grid_id': grid_id,
                    'data': grid_data
                })

        return {
            'success': True,
            'search_id': search_id,
            'pet_id': search_info['pet_id'],
            'total_grids': search_info['total_grids'],
            'grids': grids
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting search grids: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pets")
async def view_pets_page():
    """Pet selector page - choose a pet to view their search grids"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>View Pet Search Grids</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body { margin: 0; padding: 20px; font-family: Arial; background: #f5f5f5; }
            .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; margin-bottom: 10px; }
            .header { border-bottom: 3px solid #4CAF50; padding-bottom: 20px; margin-bottom: 30px; }
            .controls {
                background: #E8F5E9;
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                border: 1px solid #4CAF50;
            }
            select {
                padding: 12px 20px;
                font-size: 16px;
                border: 2px solid #4CAF50;
                border-radius: 6px;
                width: 100%;
                max-width: 500px;
                cursor: pointer;
                background: white;
            }
            select:focus { outline: none; border-color: #45a049; }
            label {
                display: block;
                font-weight: bold;
                margin-bottom: 10px;
                color: #333;
                font-size: 16px;
            }
            #map {
                height: 600px;
                margin-top: 20px;
                border: 2px solid #333;
                border-radius: 8px;
                display: none;
            }
            .info-grid-container {
                display: flex;
                gap: 20px;
                margin: 20px 0;
            }
            .info-left {
                flex: 1;
            }
            .info-right {
                flex: 1;
            }
            .pet-info {
                background: #FFF3E0;
                padding: 15px;
                border-radius: 6px;
                border-left: 4px solid #FF9800;
                display: none;
            }
            .grid-stats {
                background: #E3F2FD;
                padding: 15px;
                border-radius: 6px;
                margin-top: 10px;
                border-left: 4px solid #2196F3;
            }
            .loading {
                text-align: center;
                padding: 20px;
                font-size: 18px;
                color: #666;
            }
            .error {
                background: #FFCDD2;
                padding: 15px;
                border-radius: 6px;
                margin: 10px 0;
                color: #c62828;
                border-left: 4px solid #c62828;
            }
            .grid-selector {
                display: none;
            }
            .grid-selector label {
                display: block;
                font-weight: bold;
                margin-bottom: 10px;
                color: #333;
                font-size: 14px;
            }
            .grid-selector select {
                padding: 10px 15px;
                font-size: 14px;
                border: 2px solid #2196F3;
                border-radius: 6px;
                width: 100%;
                cursor: pointer;
                background: white;
            }
            .grid-selector select:focus {
                outline: none;
                border-color: #1976D2;
            }
            .nav-links {
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
            }
            .nav-links a {
                color: #2196F3;
                text-decoration: none;
                margin-right: 20px;
                font-weight: 500;
            }
            .nav-links a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🐾 View Pet Search Grids</h1>
                <p style="color: #666; margin: 10px 0 0 0;">Select a pet to view their search grids on the map</p>
            </div>

            <div class="controls">
                <label for="petSelect">Select Pet:</label>
                <select id="petSelect" onchange="loadPetGrids()">
                    <option value="">-- Choose a pet --</option>
                </select>
            </div>

            <div id="loading" class="loading" style="display:none;">Loading grids...</div>
            <div id="error" class="error" style="display:none;"></div>

            <div class="info-grid-container">
                <div class="info-left">
                    <div id="petInfo" class="pet-info"></div>
                    <div id="gridStats" class="grid-stats" style="display:none;"></div>
                </div>
                <div class="info-right">
                    <div id="gridSelector" class="grid-selector">
                        <label for="gridSelect">Select Grid to Highlight:</label>
                        <select id="gridSelect" onchange="selectGridFromDropdown()">
                            <option value="">-- All grids --</option>
                        </select>
                    </div>
                </div>
            </div>

            <div id="map"></div>

            <div class="nav-links">
                <a href="/">← Back to Grid Generator</a>
                <a href="/track">View Active Searches</a>
            </div>
        </div>

        <script>
            let map = null;
            let gridRectangles = [];
            let roadPolylines = [];

            // Initialize map
            function initMap() {
                if (!map) {
                    map = L.map('map', {
                        scrollWheelZoom: true,
                        zoomControl: true,
                        doubleClickZoom: true,
                        touchZoom: true
                    }).setView([27.8428, -82.8106], 14);
                    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                        attribution: '© OpenStreetMap contributors',
                        maxZoom: 19,
                        minZoom: 10
                    }).addTo(map);
                }
            }

            // Load all pets into dropdown
            async function loadPets() {
                try {
                    const response = await fetch('/api/all-pets');
                    const data = await response.json();

                    if (data.success && data.pets) {
                        const select = document.getElementById('petSelect');
                        select.innerHTML = '<option value="">-- Choose a pet --</option>';

                        data.pets.forEach(pet => {
                            const option = document.createElement('option');
                            option.value = JSON.stringify({
                                pet_id: pet.pet_id,
                                search_id: pet.search_id,
                                pet_name: pet.pet_name
                            });
                            const searchInfo = pet.search_id ? ` (Search ID: ${pet.search_id.substring(7, 15)}...)` : ' (No search yet)';
                            option.textContent = `${pet.pet_name} - ID: ${pet.pet_id}${searchInfo}`;
                            select.appendChild(option);
                        });
                    }
                } catch (error) {
                    console.error('Error loading pets:', error);
                    showError('Failed to load pets: ' + error.message);
                }
            }

            // Load grids for selected pet
            async function loadPetGrids() {
                const select = document.getElementById('petSelect');
                const selectedValue = select.value;

                if (!selectedValue) {
                    // Hide everything if no pet selected
                    document.getElementById('map').style.display = 'none';
                    document.getElementById('petInfo').style.display = 'none';
                    document.getElementById('gridStats').style.display = 'none';
                    document.getElementById('gridSelector').style.display = 'none';
                    document.getElementById('error').style.display = 'none';
                    clearMap();
                    return;
                }

                const petData = JSON.parse(selectedValue);

                if (!petData.search_id) {
                    showError('This pet does not have an active search yet.');
                    return;
                }

                showLoading(true);
                hideError();
                clearMap();

                try {
                    const response = await fetch(`/api/search-grids/${petData.search_id}`);
                    const data = await response.json();

                    if (data.success) {
                        showPetInfo(petData, data);
                        // Show map container first
                        document.getElementById('map').style.display = 'block';
                        // Initialize map after container is visible
                        initMap();
                        // Force map to recalculate size
                        setTimeout(() => {
                            map.invalidateSize();
                            displayGrids(data.grids);
                        }, 100);
                    } else {
                        showError('Failed to load grids');
                    }
                } catch (error) {
                    console.error('Error loading grids:', error);
                    showError('Error loading grids: ' + error.message);
                } finally {
                    showLoading(false);
                }
            }

            function showPetInfo(petData, searchData) {
                const infoDiv = document.getElementById('petInfo');
                const statsDiv = document.getElementById('gridStats');

                infoDiv.innerHTML = `
                    <strong>Pet:</strong> ${petData.pet_name} (ID: ${petData.pet_id})<br>
                    <strong>Search ID:</strong> ${searchData.search_id}
                `;
                infoDiv.style.display = 'block';

                statsDiv.innerHTML = `
                    <strong>Total Grids:</strong> ${searchData.total_grids} grid squares created
                `;
                statsDiv.style.display = 'block';
            }

            function displayGrids(grids) {
                const gridSelect = document.getElementById('gridSelect');
                const gridSelector = document.getElementById('gridSelector');

                // Reset dropdown
                gridSelect.innerHTML = '<option value="">-- All grids --</option>';

                grids.forEach(grid => {
                    const gridData = grid.data;

                    // Draw grid rectangle
                    const bounds = [
                        [gridData.bounds.min_lat, gridData.bounds.min_lon],
                        [gridData.bounds.max_lat, gridData.bounds.max_lon]
                    ];

                    const rect = L.rectangle(bounds, {
                        color: '#4CAF50',
                        weight: 2,
                        fillOpacity: 0.1
                    }).addTo(map);

                    rect.gridId = grid.grid_id;
                    gridRectangles.push(rect);

                    // Draw roads in this grid
                    const roads = gridData.road_details || [];
                    if (roads && roads.length > 0) {
                        roads.forEach(road => {
                            if (road.waypoints && road.waypoints.length > 1) {
                                const coords = road.waypoints.map(wp => [wp.lat, wp.lon]);
                                const polyline = L.polyline(coords, {
                                    color: '#2196F3',
                                    weight: 3,
                                    opacity: 0.7
                                }).addTo(map);
                                polyline.gridId = grid.grid_id;
                                roadPolylines.push(polyline);
                            }
                        });
                    }

                    // Add to dropdown
                    const roadCount = gridData.road_count || roads.length;
                    const distance = gridData.total_distance_miles ? gridData.total_distance_miles.toFixed(2) : '0';
                    const option = document.createElement('option');
                    option.value = grid.grid_id;
                    option.textContent = `Grid ${grid.grid_id} - ${roadCount} roads (${distance} miles)`;
                    gridSelect.appendChild(option);
                });

                // Show dropdown
                gridSelector.style.display = 'block';

                // Fit map to show all grids with better zoom
                if (gridRectangles.length > 0) {
                    const group = new L.featureGroup(gridRectangles);
                    map.fitBounds(group.getBounds(), {
                        padding: [50, 50],
                        maxZoom: 15
                    });
                }
            }

            function selectGridFromDropdown() {
                const gridSelect = document.getElementById('gridSelect');
                const selectedGridId = gridSelect.value;

                if (selectedGridId === '') {
                    // Reset all to default
                    gridRectangles.forEach(rect => {
                        rect.setStyle({ color: '#4CAF50', weight: 2 });
                    });
                    roadPolylines.forEach(line => {
                        line.setStyle({ color: '#2196F3', weight: 3, opacity: 0.7 });
                    });
                } else {
                    highlightGrid(parseInt(selectedGridId));
                }
            }

            function highlightGrid(gridId) {
                // Reset all styles
                gridRectangles.forEach(rect => {
                    rect.setStyle({ color: '#4CAF50', weight: 2 });
                });
                roadPolylines.forEach(line => {
                    line.setStyle({ color: '#2196F3', weight: 3, opacity: 0.7 });
                });

                // Highlight selected grid
                gridRectangles.forEach(rect => {
                    if (rect.gridId === gridId) {
                        rect.setStyle({ color: '#FFD700', weight: 5 });
                    }
                });
                roadPolylines.forEach(line => {
                    if (line.gridId === gridId) {
                        line.setStyle({ color: '#FF6B6B', weight: 5, opacity: 1 });
                    }
                });
            }

            function clearMap() {
                if (map) {
                    gridRectangles.forEach(rect => map.removeLayer(rect));
                    roadPolylines.forEach(line => map.removeLayer(line));
                    gridRectangles = [];
                    roadPolylines = [];
                    // Reset map view
                    map.setView([27.8428, -82.8106], 13);
                }
            }

            function showLoading(show) {
                document.getElementById('loading').style.display = show ? 'block' : 'none';
            }

            function showError(message) {
                const errorDiv = document.getElementById('error');
                errorDiv.textContent = message;
                errorDiv.style.display = 'block';
            }

            function hideError() {
                document.getElementById('error').style.display = 'none';
            }

            // Load pets on page load
            loadPets();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/track")
async def track_searchers():
    """Tracking interface with two views: All Searches and Active Searchers"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pet Search Tracking</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body { margin: 0; padding: 0; font-family: Arial; }

            /* Tab Navigation */
            .tabs {
                display: flex;
                background: #333;
                padding: 0;
                margin: 0;
            }
            .tab {
                flex: 1;
                padding: 15px;
                text-align: center;
                color: #aaa;
                cursor: pointer;
                border: none;
                background: #333;
                font-size: 16px;
                font-weight: bold;
                transition: all 0.3s;
            }
            .tab:hover {
                background: #444;
                color: #fff;
            }
            .tab.active {
                background: #fff;
                color: #333;
            }

            /* Content Areas */
            .tab-content {
                display: none;
                padding: 20px;
            }
            .tab-content.active {
                display: block;
            }

            #map { height: 700px; margin-top: 20px; border: 2px solid #333; }
            .info { margin: 10px 0; padding: 10px; background: #E8F5E9; border-radius: 4px; }
            .stats {
                background: #FFF3E0; padding: 15px; margin: 10px 0;
                border-left: 4px solid #FF9800; border-radius: 4px;
            }
            .searcher-list {
                max-height: 200px; overflow-y: auto;
                background: white; border: 1px solid #ddd;
                margin-top: 10px; border-radius: 4px;
            }
            .searcher-item {
                padding: 10px; border-bottom: 1px solid #eee;
            }
            .online { color: #4CAF50; font-weight: bold; }
            .offline { color: #999; }
            button {
                padding: 10px 20px; font-size: 16px; cursor: pointer;
                background: #4CAF50; color: white; border: none; border-radius: 4px;
                margin: 5px;
            }
            button:hover { background: #45a049; }
        </style>
    </head>
    <body>
        <div class="tabs">
            <button class="tab active" onclick="switchTab('all-searches')">📍 All Searches</button>
            <button class="tab" onclick="switchTab('active-searchers')">🔴 Active Searchers</button>
        </div>

        <!-- Tab 1: All Searches -->
        <div id="all-searches" class="tab-content active">
            <h2>All Pet Searches Ever Created</h2>
            <div id="searches-info" class="info">Loading all search locations...</div>
            <div id="searches-stats" class="stats" style="display:none;"></div>
            <div id="map"></div>
        </div>

        <!-- Tab 2: Active Searchers -->
        <div id="active-searchers" class="tab-content">
            <h2>Real-Time Active Searchers</h2>
            <div style="margin-bottom: 20px;">
                <button onclick="toggleAutoRefresh()">
                    <span id="refreshStatus">Auto-Refresh: ON</span>
                </button>
                <button onclick="loadActiveSearchers()">Refresh Now</button>
            </div>
            <div id="searchers-info" class="info">Showing all active searchers across all searches</div>
            <div id="searchers-stats" class="stats" style="display:none;"></div>
            <div id="searchers-map" style="height: 700px; margin-top: 20px; border: 2px solid #333;"></div>
            <div id="searcherList" class="searcher-list" style="display:none;"></div>
        </div>

        <script>
            // Florida center: 27.6648° N, 81.5158° W
            const FLORIDA_CENTER = [27.6648, -81.5158];
            const FLORIDA_ZOOM = 7;

            let currentTab = 'all-searches';
            let allSearchesMap = null;
            let activeSearchersMap = null;
            let searchMarkers = {};
            let searcherMarkers = {};
            let autoRefresh = true;
            let refreshInterval = null;

            // Initialize maps
            function initMaps() {
                // Map 1: All Searches
                allSearchesMap = L.map('map').setView(FLORIDA_CENTER, FLORIDA_ZOOM);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(allSearchesMap);

                // Map 2: Active Searchers
                activeSearchersMap = L.map('searchers-map').setView(FLORIDA_CENTER, FLORIDA_ZOOM);
                L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(activeSearchersMap);
            }

            // Tab switching
            function switchTab(tabName) {
                // Update tab buttons
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelector(`button[onclick="switchTab('${tabName}')"]`).classList.add('active');

                // Update content
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                document.getElementById(tabName).classList.add('active');

                currentTab = tabName;

                // Invalidate map size after tab switch
                setTimeout(() => {
                    if (tabName === 'all-searches') {
                        allSearchesMap.invalidateSize();
                    } else {
                        activeSearchersMap.invalidateSize();
                    }
                }, 100);

                // Load data for the tab
                if (tabName === 'all-searches') {
                    loadAllSearches();
                    stopAutoRefresh();
                } else {
                    loadActiveSearchers();
                    if (autoRefresh) startAutoRefresh();
                }
            }

            // Load all search locations
            async function loadAllSearches() {
                try {
                    const response = await fetch('/api/all-searches');
                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.detail || 'Failed to fetch searches');
                    }

                    // Clear existing markers
                    Object.values(searchMarkers).forEach(m => allSearchesMap.removeLayer(m));
                    searchMarkers = {};

                    // Create search icon (red pin)
                    const searchIcon = L.divIcon({
                        html: '<div style="background: #E53935; color: white; border-radius: 50%; width: 24px; height: 24px; display: flex; align-items: center; justify-content: center; font-size: 14px; border: 3px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.3);">📍</div>',
                        iconSize: [24, 24],
                        className: 'search-marker'
                    });

                    // Add markers for each search
                    data.searches.forEach(search => {
                        const marker = L.marker([search.center_lat, search.center_lon], { icon: searchIcon })
                            .addTo(allSearchesMap);

                        const date = new Date(search.created_at * 1000).toLocaleDateString();
                        marker.bindPopup(`
                            <b>Search Location</b><br>
                            ${search.address}<br>
                            <small>Created: ${date}</small><br>
                            <small>Radius: ${search.radius_miles} miles</small><br>
                            <small>ID: ${search.search_id.substring(0, 20)}...</small>
                        `);

                        searchMarkers[search.search_id] = marker;
                    });

                    // Update info
                    document.getElementById('searches-info').innerHTML =
                        `✅ Showing ${data.total_searches} search location${data.total_searches !== 1 ? 's' : ''}`;

                    document.getElementById('searches-stats').style.display = 'block';
                    document.getElementById('searches-stats').innerHTML = `
                        <strong>Total Searches Created:</strong> ${data.total_searches}<br>
                        <strong>Locations:</strong> Across Florida
                    `;

                    // Fit bounds if there are markers
                    if (data.searches.length > 0) {
                        const group = L.featureGroup(Object.values(searchMarkers));
                        allSearchesMap.fitBounds(group.getBounds().pad(0.2));
                    }

                } catch (error) {
                    console.error('Error loading searches:', error);
                    document.getElementById('searches-info').style.background = '#FFCDD2';
                    document.getElementById('searches-info').innerHTML = '❌ ' + error.message;
                }
            }

            // Load active searchers
            async function loadActiveSearchers() {
                try {
                    const response = await fetch('/api/active-searchers?search_id=all');
                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.detail || 'Failed to fetch data');
                    }

                    // Clear existing markers
                    Object.values(searcherMarkers).forEach(m => activeSearchersMap.removeLayer(m));
                    searcherMarkers = {};

                    // Searcher icon (orange person)
                    const searcherIcon = L.divIcon({
                        html: '<div style="background: #FF5722; color: white; border-radius: 50%; width: 30px; height: 30px; display: flex; align-items: center; justify-content: center; font-size: 16px; border: 3px solid white; box-shadow: 0 2px 5px rgba(0,0,0,0.3);">👤</div>',
                        iconSize: [30, 30],
                        className: 'searcher-marker'
                    });

                    // Add searcher markers
                    const currentSearcherIds = new Set();
                    data.active_searchers.forEach(searcher => {
                        currentSearcherIds.add(searcher.searcher_id);

                        if (searcher.location && searcher.location.lat) {
                            const pos = [searcher.location.lat, searcher.location.lon];
                            const now = Math.floor(Date.now() / 1000);
                            const lastUpdate = now - (searcher.location.timestamp || 0);

                            const marker = L.marker(pos, { icon: searcherIcon }).addTo(activeSearchersMap);
                            searcherMarkers[searcher.searcher_id] = marker;

                            marker.bindPopup(`
                                <b>${searcher.searcher_name || 'Unknown'}</b><br>
                                Grid: grid${searcher.grid_id}<br>
                                Progress: ${searcher.completion_percentage.toFixed(1)}%<br>
                                Last update: ${lastUpdate}s ago<br>
                                Accuracy: ±${searcher.location.accuracy_meters || '?'}m
                            `);
                        }
                    });

                    // Update info
                    document.getElementById('searchers-info').style.background = '#E8F5E9';
                    document.getElementById('searchers-info').innerHTML =
                        `✅ Tracking ${data.total_active} active searcher${data.total_active !== 1 ? 's' : ''} across all searches`;

                    document.getElementById('searchers-stats').style.display = 'block';
                    document.getElementById('searchers-stats').innerHTML = `
                        <strong>Active Searchers:</strong> ${data.total_active}<br>
                        <strong>Last Update:</strong> ${new Date().toLocaleTimeString()}
                    `;

                    // Update searcher list
                    const searcherListHtml = data.active_searchers.map(s => {
                        const hasLocation = s.location && s.location.lat;
                        const lastUpdate = hasLocation ?
                            Math.floor(Date.now() / 1000) - s.location.timestamp : null;

                        return `
                            <div class="searcher-item">
                                <span class="${hasLocation && lastUpdate < 60 ? 'online' : 'offline'}">●</span>
                                <strong>${s.searcher_name || 'Unknown'}</strong> -
                                grid${s.grid_id} (${s.completion_percentage.toFixed(1)}%)
                                ${hasLocation ? ` - Updated ${lastUpdate}s ago` : ' - No location yet'}
                            </div>
                        `;
                    }).join('');

                    document.getElementById('searcherList').innerHTML =
                        '<div style="padding: 10px; background: #f5f5f5; font-weight: bold;">Active Searchers</div>' +
                        searcherListHtml;
                    document.getElementById('searcherList').style.display = 'block';

                    // Auto-zoom to show all searchers
                    if (Object.keys(searcherMarkers).length > 0) {
                        const group = L.featureGroup(Object.values(searcherMarkers));
                        activeSearchersMap.fitBounds(group.getBounds().pad(0.1));
                    } else {
                        activeSearchersMap.setView(FLORIDA_CENTER, FLORIDA_ZOOM);
                    }

                } catch (error) {
                    console.error('Error updating searchers:', error);
                    document.getElementById('searchers-info').style.background = '#FFCDD2';
                    document.getElementById('searchers-info').innerHTML = '❌ ' + error.message;
                }
            }

            // Auto-refresh controls
            function toggleAutoRefresh() {
                autoRefresh = !autoRefresh;
                document.getElementById('refreshStatus').textContent =
                    autoRefresh ? 'Auto-Refresh: ON' : 'Auto-Refresh: OFF';

                if (autoRefresh && currentTab === 'active-searchers') {
                    startAutoRefresh();
                } else {
                    stopAutoRefresh();
                }
            }

            function startAutoRefresh() {
                stopAutoRefresh();
                refreshInterval = setInterval(() => {
                    if (autoRefresh && currentTab === 'active-searchers') {
                        loadActiveSearchers();
                    }
                }, 10000); // Refresh every 10 seconds
            }

            function stopAutoRefresh() {
                if (refreshInterval) {
                    clearInterval(refreshInterval);
                    refreshInterval = null;
                }
            }

            // Initialize on page load
            window.onload = () => {
                initMaps();
                loadAllSearches();
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.get("/")
async def root():
    """Interactive map interface with geographic grids"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pet Search - Geographic Grids</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body { margin: 0; padding: 20px; font-family: Arial; }
            #map { height: 700px; margin-top: 20px; border: 2px solid #333; }
            .controls { margin-bottom: 20px; background: #f5f5f5; padding: 15px; border-radius: 5px; }
            button {
                padding: 10px 20px; font-size: 16px; cursor: pointer;
                background: #4CAF50; color: white; border: none; border-radius: 4px;
                margin-right: 10px;
            }
            button:hover { background: #45a049; }
            input[type="text"] { padding: 8px; width: 400px; margin-right: 10px; }
            select { padding: 8px; font-size: 14px; margin-right: 10px; }
            .info { margin: 10px 0; padding: 10px; background: #E8F5E9; border-radius: 4px; }
            .success { background: #C8E6C9; padding: 10px; margin: 10px 0; border-radius: 4px; }
            .error { background: #FFCDD2; padding: 10px; margin: 10px 0; border-radius: 4px; }
            .stats {
                background: #FFF3E0; padding: 15px; margin: 10px 0;
                border-left: 4px solid #FF9800; border-radius: 4px;
            }
            .grid-list {
                max-height: 300px; overflow-y: auto;
                background: white; border: 1px solid #ddd;
                margin-top: 20px; border-radius: 4px;
            }
            .grid-item {
                padding: 10px; border-bottom: 1px solid #eee;
                cursor: pointer; transition: background 0.2s;
            }
            .grid-item:hover { background: #f0f0f0; }
            .grid-item.selected { background: #FFD700; font-weight: bold; }
            .instructions {
                background: #E3F2FD; padding: 15px; margin: 10px 0;
                border-left: 4px solid #2196F3; border-radius: 4px;
            }
            @keyframes pulse {
                0%, 100% { stroke-width: 3; }
                50% { stroke-width: 6; }
            }
            .highlighted { stroke: #FFD700 !important; stroke-width: 5; animation: pulse 1.5s infinite; }
        </style>
    </head>
    <body>
        <h1>🗺️ Pet Search - Geographic Grid System</h1>

        <div class="instructions">
            <strong>📋 Instructions:</strong><br>
            • Select grid size (0.3-1.0 miles per square)<br>
            • Click <strong>Grid Rectangles</strong> on map to highlight all roads in that grid<br>
            • Click <strong>Grid Info Panels</strong> below map to highlight grids<br>
            • Click empty map area to deselect
        </div>

        <div class="controls">
            <input type="text" id="address" placeholder="Enter any address"
                   value="11388 86th Ave N, Seminole, FL 33772">
            <br><br>
            <label><strong>Search Radius:</strong></label>
            <input type="number" id="searchRadius" min="0.25" max="5" step="0.25" value="1.5" style="width: 80px;">
            <label>miles</label>

            <label style="margin-left: 20px;"><strong>Grid Size:</strong></label>
            <select id="gridSize">
                <option value="0.3">0.3 miles</option>
                <option value="0.5" selected>0.5 miles</option>
                <option value="0.7">0.7 miles</option>
                <option value="1.0">1.0 miles</option>
            </select>

            <button onclick="createSearch()">Generate Geographic Grids</button>
        </div>

        <div id="info" class="info">
            Creates geographic grid squares of specified size with volunteer-friendly search areas
        </div>
        <div id="stats" class="stats" style="display:none;"></div>
        <div id="map"></div>
        <div id="gridList" class="grid-list" style="display:none;"></div>

        <script>
            let map = L.map('map').setView([27.8428, -82.8106], 15);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

            let gridRectangles = [];
            let roadPolylines = [];
            let gridData = [];
            let selectedGridId = null;

            // Click on map to deselect
            map.on('click', function(e) {
                if (!e.originalEvent.gridClicked) {
                    deselectAll();
                }
            });

            function deselectAll() {
                selectedGridId = null;

                // Remove highlight from all grid rectangles
                gridRectangles.forEach(rect => {
                    rect.setStyle({
                        color: '#3388ff',
                        weight: 2,
                        fillOpacity: 0.1
                    });
                });

                // Reset all road colors
                roadPolylines.forEach(poly => {
                    poly.setStyle({
                        color: poly.options.originalColor,
                        weight: 3
                    });
                });

                // Remove selection from grid list items
                document.querySelectorAll('.grid-item').forEach(item => {
                    item.classList.remove('selected');
                });

                document.getElementById('info').innerHTML =
                    'Click on a grid rectangle or grid panel to highlight roads';
            }

            function highlightGrid(gridId) {
                deselectAll();
                selectedGridId = gridId;

                const grid = gridData.find(g => g.grid_id === gridId);
                if (!grid) return;

                // Highlight grid rectangle with gold pulsing border
                const rect = gridRectangles.find(r => r.gridId === gridId);
                if (rect) {
                    rect.setStyle({
                        color: '#FFD700',
                        weight: 5,
                        fillOpacity: 0.2,
                        fillColor: '#FFD700'
                    });
                    rect.bringToFront();
                }

                // Highlight all roads in this grid with bright red
                roadPolylines.forEach(poly => {
                    if (poly.gridId === gridId) {
                        poly.setStyle({
                            color: '#FF0000',
                            weight: 5
                        });
                        poly.bringToFront();
                    }
                });

                // Highlight grid item in list
                const gridItem = document.querySelector(`.grid-item[data-grid-id="${gridId}"]`);
                if (gridItem) {
                    gridItem.classList.add('selected');
                    gridItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }

                document.getElementById('info').innerHTML =
                    `🔥 <strong>HIGHLIGHTED:</strong> grid${gridId} with ${grid.road_count} roads (${grid.total_distance_miles} miles)`;
            }

            async function createSearch() {
                document.getElementById('info').innerHTML = '⏳ Creating geographic grids...';
                document.getElementById('stats').style.display = 'none';
                document.getElementById('gridList').style.display = 'none';

                const address = document.getElementById('address').value;
                const searchRadius = parseFloat(document.getElementById('searchRadius').value);
                const gridSize = parseFloat(document.getElementById('gridSize').value);

                try {
                    const response = await fetch('/api/create-search', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            lat: 0, lon: 0,
                            radius_miles: searchRadius,
                            grid_size_miles: gridSize,
                            address
                        })
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.detail || 'Server error');
                    }

                    // Clear old layers
                    gridRectangles.forEach(r => map.removeLayer(r));
                    roadPolylines.forEach(r => map.removeLayer(r));
                    gridRectangles = [];
                    roadPolylines = [];
                    gridData = data.tiles;
                    selectedGridId = null;

                    // Center map
                    map.setView([data.center.lat, data.center.lon], 15);

                    // Add center marker
                    L.marker([data.center.lat, data.center.lon])
                        .addTo(map)
                        .bindPopup('Search Center');

                    // Draw grid rectangles
                    data.tiles.forEach((tile, i) => {
                        const bounds = [
                            [tile.bounds.min_lat, tile.bounds.min_lon],
                            [tile.bounds.max_lat, tile.bounds.max_lon]
                        ];

                        const rect = L.rectangle(bounds, {
                            color: '#3388ff',
                            weight: 2,
                            fillOpacity: 0.1,
                            fillColor: '#3388ff'
                        }).addTo(map);

                        rect.gridId = tile.grid_id;

                        rect.on('click', function(e) {
                            e.originalEvent.gridClicked = true;
                            highlightGrid(tile.grid_id);
                        });

                        rect.bindPopup(`
                            <b>grid${tile.grid_id}</b><br>
                            Roads: ${tile.road_count}<br>
                            Total: ${tile.total_distance_miles} miles<br>
                            Est. time: ${tile.estimated_minutes} min<br>
                            Size: ${tile.grid_size_miles}x${tile.grid_size_miles} miles
                        `);

                        gridRectangles.push(rect);

                        // Draw each road in this grid separately
                        if (tile.road_details && tile.road_details.length > 0) {
                            const colors = [
                                '#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336',
                                '#00BCD4', '#8BC34A', '#E91E63', '#3F51B5', '#FFC107'
                            ];

                            tile.road_details.forEach((road, roadIdx) => {
                                const validPts = road.waypoints
                                    .filter(p => !isNaN(p.lat) && !isNaN(p.lon))
                                    .map(p => [p.lat, p.lon]);

                                if (validPts.length > 1) {
                                    // Color based on whether road has a name
                                    const color = road.has_name ?
                                        colors[roadIdx % colors.length] :
                                        '#808080';  // Gray for unnamed

                                    const poly = L.polyline(validPts, {
                                        color: color,
                                        weight: road.has_name ? 4 : 3,
                                        opacity: road.has_name ? 0.9 : 0.6,
                                        originalColor: color
                                    }).addTo(map);

                                    poly.gridId = tile.grid_id;
                                    poly.bindPopup(`
                                        <b>${road.name}</b><br>
                                        Type: ${road.highway_type}<br>
                                        Length: ${road.length_meters}m<br>
                                        grid${tile.grid_id}
                                    `);

                                    roadPolylines.push(poly);
                                }
                            });
                        }
                    });

                    // Update UI
                    document.getElementById('info').className = 'success';
                    document.getElementById('info').innerHTML =
                        `✅ Created ${data.total_tiles} grids (${gridSize}x${gridSize} miles each) with ${data.total_roads} roads`;

                    document.getElementById('stats').style.display = 'block';
                    document.getElementById('stats').innerHTML = `
                        <strong>Geographic Grid Statistics:</strong><br>
                        • Search radius: ${searchRadius} miles<br>
                        • Total grids: ${data.total_tiles} (non-empty)<br>
                        • Grid size: ${gridSize} x ${gridSize} miles<br>
                        • Total roads: ${data.total_roads}<br>
                        • Fake diagonals removed: ${data.filtered_count}<br>
                        • Overlap: 20% between adjacent grids
                    `;

                    // Create grid list
                    const gridListHtml = data.tiles.map(tile => `
                        <div class="grid-item" data-grid-id="${tile.grid_id}" onclick="highlightGrid(${tile.grid_id})">
                            <strong>grid${tile.grid_id}</strong> -
                            ${tile.road_count} roads,
                            ${tile.total_distance_miles} miles,
                            ~${tile.estimated_minutes} min
                        </div>
                    `).join('');

                    document.getElementById('gridList').innerHTML =
                        '<h3 style="padding: 10px; margin: 0; background: #f5f5f5;">Grid Details (click to highlight)</h3>' +
                        gridListHtml;
                    document.getElementById('gridList').style.display = 'block';

                } catch (error) {
                    document.getElementById('info').className = 'error';
                    document.getElementById('info').innerHTML = '❌ ' + error.message;
                    console.error(error);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8443,
        ssl_keyfile="/opt/petsearch/key.pem",
        ssl_certfile="/opt/petsearch/cert.pem"
    )
