-- Pet Search Roads Database Schema
-- Tracks discovered roads with deduplication

-- Searches table: tracks each search operation
CREATE TABLE IF NOT EXISTS searches (
    id TEXT PRIMARY KEY,
    center_lat REAL NOT NULL,
    center_lon REAL NOT NULL,
    radius_miles REAL NOT NULL,
    address TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    total_roads INTEGER DEFAULT 0,
    filtered_roads INTEGER DEFAULT 0
);

-- Roads table: stores unique roads (deduplicated)
CREATE TABLE IF NOT EXISTS roads (
    id TEXT PRIMARY KEY,  -- hash of geometry + name for deduplication
    name TEXT NOT NULL,
    highway_type TEXT,
    length_meters REAL,
    first_seen_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_seen_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    times_seen INTEGER DEFAULT 1
);

-- Road waypoints: stores all coordinate points for each road
CREATE TABLE IF NOT EXISTS road_waypoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    road_id TEXT NOT NULL,
    sequence_order INTEGER NOT NULL,  -- order of waypoint in the road
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    FOREIGN KEY (road_id) REFERENCES roads(id) ON DELETE CASCADE
);

-- Search-Road junction: tracks which roads were found in which searches
CREATE TABLE IF NOT EXISTS search_roads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    search_id TEXT NOT NULL,
    road_id TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (search_id) REFERENCES searches(id) ON DELETE CASCADE,
    FOREIGN KEY (road_id) REFERENCES roads(id) ON DELETE CASCADE,
    UNIQUE(search_id, road_id)  -- prevent duplicates within same search
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_roads_name ON roads(name);
CREATE INDEX IF NOT EXISTS idx_roads_highway_type ON roads(highway_type);
CREATE INDEX IF NOT EXISTS idx_road_waypoints_road_id ON road_waypoints(road_id);
CREATE INDEX IF NOT EXISTS idx_search_roads_search_id ON search_roads(search_id);
CREATE INDEX IF NOT EXISTS idx_search_roads_road_id ON search_roads(road_id);
CREATE INDEX IF NOT EXISTS idx_searches_created_at ON searches(created_at);
