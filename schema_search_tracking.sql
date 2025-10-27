-- Cloudflare D1 Database Schema for Pet Search Grid Tracking System
-- This schema supports iOS app integration for coordinated volunteer search efforts

-- Table 1: searches
-- Stores overall information about each lost pet search
CREATE TABLE IF NOT EXISTS searches (
    search_id TEXT PRIMARY KEY,           -- Generated on EC2 when creating grids
    pet_id TEXT NOT NULL,                 -- From iOS: hash of Apple ID + date submitted
    address TEXT NOT NULL,                -- Where the pet was last seen
    center_lat REAL NOT NULL,             -- Geocoded latitude
    center_lon REAL NOT NULL,             -- Geocoded longitude
    radius_miles REAL NOT NULL,           -- Search radius
    grid_size_miles REAL DEFAULT 0.3,     -- Size of each grid (0.3 miles)
    total_grids INTEGER NOT NULL,         -- Total number of grids created
    created_at INTEGER NOT NULL,          -- Unix timestamp
    status TEXT DEFAULT 'active',         -- active, paused, completed, cancelled
    UNIQUE(pet_id)                        -- One search per pet_id
);

-- Table 2: grid_assignments
-- Tracks which volunteer is assigned to which grid
CREATE TABLE IF NOT EXISTS grid_assignments (
    assignment_id TEXT PRIMARY KEY,       -- Unique assignment ID
    search_id TEXT NOT NULL,              -- Links to searches table
    pet_id TEXT NOT NULL,                 -- For quick filtering
    grid_id INTEGER NOT NULL,             -- Grid number (1, 2, 3...)
    searcher_id TEXT NOT NULL,            -- From iOS: unique identifier for volunteer
    searcher_name TEXT,                   -- Optional: volunteer's name
    assigned_at INTEGER NOT NULL,         -- Unix timestamp when assigned
    timeframe_minutes INTEGER NOT NULL,   -- 30, 60, 90, or 120
    expires_at INTEGER NOT NULL,          -- assigned_at + timeframe_minutes
    grace_expires_at INTEGER NOT NULL,    -- expires_at + 10 minutes
    status TEXT DEFAULT 'active',         -- active, expired, completed, abandoned
    completion_percentage REAL DEFAULT 0, -- 0-100, completed at 85%
    completed_at INTEGER,                 -- When marked complete (85%+)
    FOREIGN KEY (search_id) REFERENCES searches(search_id)
);

-- Table 3: search_progress
-- Stores GPS tracking breadcrumbs from volunteers
CREATE TABLE IF NOT EXISTS search_progress (
    progress_id TEXT PRIMARY KEY,         -- Unique progress entry ID
    assignment_id TEXT NOT NULL,          -- Links to grid_assignments
    search_id TEXT NOT NULL,              -- For quick queries
    pet_id TEXT NOT NULL,                 -- For filtering by pet
    grid_id INTEGER NOT NULL,             -- Which grid they're searching
    searcher_id TEXT NOT NULL,            -- Who is searching
    lat REAL NOT NULL,                    -- GPS latitude
    lon REAL NOT NULL,                    -- GPS longitude
    timestamp INTEGER NOT NULL,           -- Unix timestamp
    accuracy_meters REAL,                 -- GPS accuracy if available
    distance_miles REAL,                  -- Distance traveled in miles (cumulative)
    elapsed_minutes INTEGER,              -- Time elapsed in minutes (cumulative)
    FOREIGN KEY (assignment_id) REFERENCES grid_assignments(assignment_id),
    FOREIGN KEY (search_id) REFERENCES searches(search_id)
);

-- Table 4: roads_searched
-- Tracks which roads have been searched and when
CREATE TABLE IF NOT EXISTS roads_searched (
    road_search_id TEXT PRIMARY KEY,      -- Unique ID
    assignment_id TEXT NOT NULL,          -- Which assignment searched this road
    search_id TEXT NOT NULL,              -- Which pet search
    pet_id TEXT NOT NULL,                 -- For filtering
    grid_id INTEGER NOT NULL,             -- Which grid contains this road
    searcher_id TEXT NOT NULL,            -- Who searched it
    road_id TEXT NOT NULL,                -- Road identifier from OSM data
    road_name TEXT NOT NULL,              -- Street/road name
    searched_at INTEGER NOT NULL,         -- When it was searched
    last_searched_at INTEGER,             -- Track re-searches (after 12 hours)
    search_count INTEGER DEFAULT 1,       -- How many times searched
    FOREIGN KEY (assignment_id) REFERENCES grid_assignments(assignment_id),
    FOREIGN KEY (search_id) REFERENCES searches(search_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_searches_pet_id ON searches(pet_id);
CREATE INDEX IF NOT EXISTS idx_searches_status ON searches(status);

CREATE INDEX IF NOT EXISTS idx_assignments_search_id ON grid_assignments(search_id);
CREATE INDEX IF NOT EXISTS idx_assignments_pet_id ON grid_assignments(pet_id);
CREATE INDEX IF NOT EXISTS idx_assignments_searcher_id ON grid_assignments(searcher_id);
CREATE INDEX IF NOT EXISTS idx_assignments_status ON grid_assignments(status);
CREATE INDEX IF NOT EXISTS idx_assignments_grid_id ON grid_assignments(search_id, grid_id);

CREATE INDEX IF NOT EXISTS idx_progress_assignment_id ON search_progress(assignment_id);
CREATE INDEX IF NOT EXISTS idx_progress_search_id ON search_progress(search_id);
CREATE INDEX IF NOT EXISTS idx_progress_timestamp ON search_progress(timestamp);

CREATE INDEX IF NOT EXISTS idx_roads_search_id ON roads_searched(search_id);
CREATE INDEX IF NOT EXISTS idx_roads_assignment_id ON roads_searched(assignment_id);
CREATE INDEX IF NOT EXISTS idx_roads_grid_id ON roads_searched(search_id, grid_id);
CREATE INDEX IF NOT EXISTS idx_roads_searched_at ON roads_searched(searched_at);
