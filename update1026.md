# Pet Search Grid System - Technical Update 10/26/2025

## Failures and Fixes

### 1. OSMnx Timeout/Hanging (Original System)
- **Problem**: OSMnx hanging indefinitely during `graph_from_point()` calls, causing server crashes
- **Root Cause**: OSMnx downloads entire road network from Overpass API synchronously, no timeout control
- **Impact**: Server crashed 6+ times, workers killed by systemd
- **Resolution**: Abandoned OSMnx completely

### 2. Pyrosm Path Error
- **Problem**: `FileNotFoundError: /root/osm_data/florida.osm.pbf`
- **Root Cause**: Hardcoded path in wrong location
- **Fix**: Changed `PBF_FILE_PATH` from `/root/osm_data/` to `/opt/petsearch/osm_data/florida.osm.pbf`

### 3. Pyrosm Memory Overflow (15GB RAM)
- **Problem**: Pyrosm loaded entire 601MB Florida PBF file into memory despite `bounding_box` parameter
- **Root Cause**: `bounding_box` parameter only filters AFTER loading entire file into memory
- **Impact**: Workers killed with SIGKILL, OOM errors
- **Symptoms**: 15GB memory usage per worker
- **Resolution**: Abandoned Pyrosm, switched to Overpass API direct queries

### 4. Missing geopy Module
- **Problem**: `ModuleNotFoundError: No module named 'geopy'`
- **Root Cause**: New dependency not installed for geodesic distance calculations
- **Fix**: `pip install geopy`

### 5. LineString Import Conflict
- **Problem**: `UnboundLocalError: cannot access local variable 'LineString'`
- **Root Cause**: Duplicate import inside function after already using LineString
- **Location**: Line 1214 in `download_osm_with_overpass()`
- **Fix**: Removed local import statement

### 6. Spatial Index Lookup Failure (2,762 Failed Lookups)
- **Problem**: All geometry lookups returned None from `geom_to_road` dictionary
- **Root Cause**: Python object identity - `STRtree.query()` returns new geometry instances, not original objects
- **Code Pattern**:
  ```python
  # BROKEN:
  geom_to_road = {road['geometry']: road for road in roads}
  indices = spatial_index.query(bbox)
  for geom in indices:
      road = geom_to_road.get(geom)  # Always None!
  ```
- **Fix**: Index-based lookup instead of object identity:
  ```python
  roads_with_geom = [road for road in roads if road.get('geometry')]
  road_geometries = [road['geometry'] for road in roads_with_geom]
  spatial_index = STRtree(road_geometries)
  candidate_indices = spatial_index.query(expanded_bbox, predicate='intersects')
  for idx in candidate_indices:
      road = roads_with_geom[idx]  # Direct array access
  ```
- **Location**: Lines 466-477 in `create_grids_from_roads()`

### 7. Grid Expansion Too Small (0 Candidates)
- **Problem**: Spatial queries found 0 candidate roads despite 2,762 total roads
- **Root Cause**: `expansion_factor=2.0` created bounding box too small (0.022 miles radius)
- **Fix**: Increased to `expansion_factor=10.0` (0.11 miles radius from grid center)
- **Location**: Line 512 in `create_grids_from_roads()`

### 8. Missing Roads (10.44 miles / 23% Unassigned)
- **Problem**: Only 33.76 miles in grids out of 44.20 total
- **Root Cause**: BFS algorithm stopped adding roads when approaching `max_miles`, creating many small disconnected clusters that were skipped by `min_miles=4` threshold
- **Original Logic**: Skip grids with <4 miles
- **Fix**: Removed minimum miles requirement - create grids for ALL road clusters regardless of size
- **Result**: 100% road coverage - all 44.20 miles now assigned to 62 grids

### 9. Race Condition in /api/assign-grid (HTTP 400 Error)
- **Problem**: iOS app getting HTTP 400 "Not enough available grids. Requested: 1, Available: 0" immediately after search creation
- **Root Cause**: Status updated to 'active' at line 1522, but grids saved to database at lines 1527-1528. iOS app hits `/api/assign-grid` between these operations.
- **Impact**: iOS app treats as fatal error instead of retrying
- **Fix**: Added check at lines 1790-1799 - if `status='active'` but `available_grids=0`, return HTTP 200 with same retry payload as `status='pending'`
- **Response Format**:
  ```json
  {
      "success": false,
      "error": "grids_not_ready",
      "message": "Grid generation in progress, please retry in a few seconds",
      "retry_after_seconds": 5,
      "search_id": "search-uuid",
      "status": "pending"
  }
  ```
- **iOS Requirement**: Must check `success` field and retry if `error: "grids_not_ready"`, not treat HTTP 200 as automatic success

## Current System Architecture

### Data Flow
1. Client sends POST to `/api/create-search` with lat/lon/radius
2. Server creates search record with status='pending'
3. Background task starts grid generation via `asyncio.create_task()`
4. Overpass API queried for road network within radius
5. Roads extracted from OSM graph, converted to road segments
6. Spatial index built for fast geometric queries
7. Grid generation using NW-to-SE sweep pattern
8. Grids saved to Cloudflare D1 database
9. Status updated to 'active', client polls for completion

### Overpass API Integration

**Query URL**: `https://overpass-api.de/api/interpreter`

**Query Template**:
```
[out:json][timeout:180];
(
  way["highway"](around:{radius_meters},{lat},{lon});
  >;
);
out body;
```

**Timeout**: 180 seconds
**Download Radius**: `radius_miles * 1.18` (adds 500ft completion buffer)
**Road Types**: All `highway` tagged ways (primary, secondary, tertiary, residential, service, etc.)

**Implementation**: Lines 1150-1230 in `server_geographic_grids.py`

### Road Extraction

**Process**:
1. Overpass returns nodes (lat/lon points) and ways (ordered node sequences)
2. Build NetworkX MultiDiGraph with nodes as vertices
3. For each way, create edges between consecutive nodes
4. Calculate edge length using `geopy.geodesic()` (accounts for Earth curvature)
5. Extract geometry as Shapely LineString
6. Filter out "fake diagonals" (edges >200m without geometry)

**Output**: List of road dictionaries:
```python
{
    'id': 'way123-node456-node789',
    'name': 'Main Street',
    'geometry': LineString([...]),
    'length_meters': 98.47,
    'waypoints': [
        {'lat': 27.8428, 'lon': -82.8106},
        {'lat': 27.8429, 'lon': -82.8107}
    ]
}
```

**Average Road Length**: 25.8 meters (85 feet) - short because Overpass splits at every intersection

### Grid Generation Algorithm

**Strategy**: Geographic sweep pattern (NOT radial from center)

**Steps**:
1. Find bounding box of all roads
2. Start at NW corner (max_lat, min_lon)
3. Create 0.5-mile grid cells in pattern: West→East, North→South
4. For each grid position:
   - Query spatial index for candidate roads within 10x expansion radius
   - Find seed road closest to grid center
   - BFS to build connected component up to `max_miles=8`
   - Stop at `target_miles=6` if possible
   - Assign ALL roads regardless of total mileage (no minimum)
5. Mark roads as assigned, continue to next grid position

**Parameters**:
- `grid_size_miles`: 0.5 (cell size for sweep pattern)
- `target_miles`: 6 (optimal for search teams)
- `max_miles`: 8 (hard stop)
- `expansion_factor`: 10.0 (multiplier for spatial queries)

**BFS Connectivity**:
```python
endpoint_to_roads = {}  # Map normalized endpoints to road indices
for road in candidates:
    start = normalize_point(waypoints[0])
    end = normalize_point(waypoints[-1])
    endpoint_to_roads[start].add(road_idx)
    endpoint_to_roads[end].add(road_idx)

# BFS traversal
queue = [seed_road_idx]
while queue and grid_miles < max_miles:
    road_idx = queue.pop(0)
    grid_roads.append(road)
    for endpoint in [start, end]:
        for connected_idx in endpoint_to_roads[endpoint]:
            queue.append(connected_idx)
```

**Point Normalization**: Round to 5 decimal places (±1.1m precision) to handle GPS noise

**Implementation**: Lines 418-747 in `server_geographic_grids.py`

### Cleanup Step (Currently Unused)

**Purpose**: Assign orphaned roads if main algorithm misses any
**Status**: All roads now assigned during main generation, cleanup doesn't execute
**Location**: Lines 633-746

**Logic** (if needed):
1. Find roads where `road['id'] not in assigned_road_ids`
2. Try to add to nearest grid within 50 feet
3. Group remaining orphans by proximity (0.05 miles)
4. Create bonus grid if cluster >1 mile
5. Otherwise add to nearest grid

## API Endpoints

**Base URL**: `https://api.psar.app`
**Auth**: Header `X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67`
**Protocol**: HTTPS (TLS cert from Let's Encrypt)

### POST /api/create-search
**Purpose**: Create new search and generate grids in background

**Request**:
```json
{
    "pet_id": "84",
    "lat": 27.8428,
    "lon": -82.8106,
    "radius_miles": 0.5,
    "grid_size_miles": 0.5
}
```

**Response** (immediate):
```json
{
    "success": true,
    "search_id": "search-uuid",
    "status": "pending",
    "message": "Search accepted, grids are being generated in background"
}
```

**Background Processing**:
- Queries Overpass API
- Generates grids
- Saves to D1 database
- Updates status to 'active'

**Timing**: 60-90 seconds for 0.5 mile radius

### POST /api/get-grids
**Purpose**: Retrieve grid tiles for existing search

**Request**:
```json
{
    "search_id": "search-uuid"
}
```

**Response**:
```json
{
    "success": true,
    "grids": [
        {
            "id": 1,
            "bounds": {
                "min_lat": 27.8350,
                "max_lat": 27.8655,
                "min_lon": -82.8282,
                "max_lon": -82.7954
            },
            "center": {"lat": 27.8503, "lon": -82.8118},
            "total_miles": 6.05,
            "roads_count": 198,
            "status": "available"
        }
    ],
    "total_grids": 62
}
```

### POST /api/assign-grid
**Purpose**: Assign grid(s) to volunteer searcher

**Request**:
```json
{
    "search_id": "search-uuid",
    "searcher_name": "John Doe",
    "searcher_phone": "+1234567890",
    "timeframe_minutes": 60
}
```

**Assignment Logic**:
- 30 min = 1 grid
- 60 min = 2 grids
- 90 min = 3 grids
- 120 min = 4 grids

**Response**:
```json
{
    "success": true,
    "assignment_id": "assign-uuid",
    "grids_assigned": [1, 2],
    "total_miles": 12.06
}
```

### POST /api/update-progress
**Purpose**: GPS tracking updates from iOS app (called every 30 seconds)

**Request**:
```json
{
    "assignment_id": "assign-uuid",
    "search_id": "search-uuid",
    "lat": 27.8428,
    "lon": -82.8106,
    "accuracy_meters": 5.0,
    "roads_covered": [
        {"road_id": "way123-node456-node789", "road_name": "Main St"}
    ],
    "distance_miles": 1.25,
    "elapsed_minutes": 15
}
```

**Response**:
```json
{
    "success": true,
    "progress_id": "progress-uuid"
}
```

### GET /api/get-grid?search_id={id}&grid_id={num}
**Purpose**: Get detailed road data for specific grid

**Response**:
```json
{
    "success": true,
    "grid": {
        "id": 1,
        "total_miles": 6.05,
        "roads_count": 198,
        "roads": [
            {
                "id": "way123-node456-node789",
                "name": "Main Street",
                "length_meters": 98.47,
                "waypoints": [
                    {"lat": 27.8428, "lon": -82.8106},
                    {"lat": 27.8429, "lon": -82.8107}
                ]
            }
        ]
    }
}
```

### GET /api/search-by-pet?pet_id={id}
**Purpose**: Lookup search_id from Airtable pet ID

**Response**:
```json
{
    "success": true,
    "search_id": "search-uuid",
    "status": "active",
    "total_grids": 62
}
```

### GET /api/grid-status?search_id={id}
**Purpose**: Get assignment status of all grids

**Response**:
```json
{
    "success": true,
    "grids": [
        {"grid_id": 1, "status": "assigned", "searcher": "John Doe"},
        {"grid_id": 2, "status": "completed"},
        {"grid_id": 3, "status": "available"}
    ]
}
```

### GET /api/search-stats?search_id={id}
**Purpose**: Aggregate statistics for search

**Response**:
```json
{
    "success": true,
    "search_id": "search-uuid",
    "total_searchers": 5,
    "total_distance_miles": 28.4,
    "total_time_minutes": 240,
    "grids_completed": 8,
    "grids_in_progress": 2,
    "grids_available": 52
}
```

## Database Schema (Cloudflare D1)

**Table**: `pet_searches`
```sql
CREATE TABLE pet_searches (
    search_id TEXT PRIMARY KEY,
    pet_id TEXT,
    center_lat REAL,
    center_lon REAL,
    radius_miles REAL,
    grid_size_miles REAL,
    total_grids INTEGER,
    status TEXT,  -- 'pending', 'active', 'completed', 'failed'
    created_at INTEGER
);
```

**Table**: `geographic_grids`
```sql
CREATE TABLE geographic_grids (
    search_id TEXT,
    grid_id INTEGER,
    bounds_json TEXT,  -- JSON blob with min/max lat/lon
    center_lat REAL,
    center_lon REAL,
    total_miles REAL,
    roads_count INTEGER,
    roads_json TEXT,  -- JSON array of road objects
    status TEXT,  -- 'available', 'assigned', 'completed'
    PRIMARY KEY (search_id, grid_id)
);
```

**Table**: `grid_assignments`
```sql
CREATE TABLE grid_assignments (
    assignment_id TEXT PRIMARY KEY,
    search_id TEXT,
    grid_id INTEGER,
    searcher_name TEXT,
    searcher_phone TEXT,
    timeframe_minutes INTEGER,
    assigned_at INTEGER,
    status TEXT  -- 'active', 'completed', 'abandoned'
);
```

**Table**: `search_progress`
```sql
CREATE TABLE search_progress (
    progress_id TEXT PRIMARY KEY,
    assignment_id TEXT,
    search_id TEXT,
    lat REAL,
    lon REAL,
    accuracy_meters REAL,
    roads_covered_json TEXT,
    distance_miles REAL,
    elapsed_minutes INTEGER,
    tracked_at INTEGER
);
```

## Server Configuration

**Service**: `/etc/systemd/system/petsearch.service`
**Command**: `gunicorn server_geographic_grids:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8443 --certfile=/etc/letsencrypt/live/api.psar.app/fullchain.pem --keyfile=/etc/letsencrypt/live/api.psar.app/privkey.pem --timeout 300`
**Workers**: 4 (one per CPU core)
**Timeout**: 300 seconds
**Port**: 8443 (HTTPS)
**Working Directory**: `/opt/petsearch`
**Python**: `/opt/petsearch/venv/bin/python3.11`

**Dependencies**:
```
fastapi
uvicorn
gunicorn
overpy
networkx
shapely
geopy
httpx
```

**Restart Service**: `sudo systemctl restart petsearch.service`

## Test Results (10/26/2025)

**Location**: 27.8428, -82.8106 (0.5 mile radius)
**Total Roads**: 2,762 segments
**Total Road Length**: 44.20 miles
**Average Road Length**: 25.8 meters (85 feet)

**Grid Results**:
- Total Grids: 62
- Roads Assigned: 2,762 (100%)
- Miles Assigned: 44.19 (99.98%)
- Unassigned Roads: 0

**Large Grids (≥3 miles)**:
- Grid 1: 198 roads, 6.05 miles
- Grid 6: 189 roads, 6.01 miles
- Grid 7: 361 roads, 6.01 miles
- Grid 9: 333 roads, 6.02 miles
- Grid 13: 384 roads, 6.09 miles
- Grid 34: 194 roads, 3.58 miles
- **Subtotal**: 6 grids, 33.76 miles

**Small Clusters (<3 miles)**:
- 56 grids containing disconnected road segments
- **Subtotal**: 10.43 miles

**Performance**:
- Overpass query: ~5 seconds
- Road extraction: ~2 seconds
- Grid generation: ~45 seconds
- Database save: ~10 seconds
- **Total**: ~60 seconds

## Debug Logging

**Location**: `/tmp/grid_gen_search-{search_id}.log`
**Format**: Timestamped progress messages

**Key Messages**:
```
[search-uuid] Starting grid generation...
Downloading OSM data for {lat}, {lon}
[OVERPASS] Got {n} ways and {m} nodes
[OVERPASS] Created NetworkX graph with {n} nodes, {m} edges
Extracted {n} edges from graph (filtered {m} fake diagonals)
[DEBUG] Total road length: {n}m ({n} miles), avg per road: {n}m
Building grids from {n} roads (target: 6 miles per grid)...
  Grid {n}: {m} roads, {x} miles (connected)
  Grid {n}: {m} roads, {x} miles (small isolated cluster)
Created {n} grids from roads
[DEBUG] Checking cleanup: {n} total roads, {m} marked as assigned, {k} unassigned
[search-uuid] Grid generation COMPLETE - {n} grids created, status=active
```

## Known Issues / Future Work

1. **Small Cluster Optimization**: 56 grids are <1 mile - could merge adjacent small clusters to reduce grid count
2. **Grid Assignment Strategy**: Currently assigns closest grids first - could optimize based on road connectivity
3. **Road Naming**: Some roads have no name (residential, service roads) - may confuse volunteers
4. **Overpass Rate Limiting**: Public API has rate limits - may need self-hosted Overpass instance for production
5. **Progress Tracking**: Currently stores GPS points but doesn't calculate actual road coverage percentage
6. **Duplicate Road Handling**: If volunteer backtracks, same road counted multiple times in progress

## Files Modified

- `/opt/petsearch/server_geographic_grids.py` - Main server (lines 11-13, 418-747, 1150-1230)
- `/opt/petsearch/osm_data/florida.osm.pbf` - Downloaded but unused (601MB)
- `/tmp/grid_gen_search-*.log` - Debug logs per search request

## Environment Variables

```bash
API_KEY=petsearch_2024_secure_key_f8d92a1b3c4e5f67
CLOUDFLARE_ACCOUNT_ID={account_id}
CLOUDFLARE_DATABASE_ID={database_id}
CLOUDFLARE_API_TOKEN={api_token}
```
