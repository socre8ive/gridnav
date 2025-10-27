#!/usr/bin/env python3
"""
Setup script to create search tracking tables in Cloudflare D1
"""
import asyncio
from database import db

async def create_tables():
    """Create all search tracking tables"""

    print("Creating search tracking tables in Cloudflare D1...")

    # Table 1: searches
    print("\n1. Creating 'searches' table...")
    await db.execute("""
        CREATE TABLE IF NOT EXISTS searches (
            search_id TEXT PRIMARY KEY,
            pet_id TEXT NOT NULL,
            address TEXT NOT NULL,
            center_lat REAL NOT NULL,
            center_lon REAL NOT NULL,
            radius_miles REAL NOT NULL,
            grid_size_miles REAL DEFAULT 0.3,
            total_grids INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            status TEXT DEFAULT 'active',
            UNIQUE(pet_id)
        )
    """)
    print("✓ 'searches' table created")

    # Table 2: grid_assignments
    print("\n2. Creating 'grid_assignments' table...")
    await db.execute("""
        CREATE TABLE IF NOT EXISTS grid_assignments (
            assignment_id TEXT PRIMARY KEY,
            search_id TEXT NOT NULL,
            pet_id TEXT NOT NULL,
            grid_id INTEGER NOT NULL,
            searcher_id TEXT NOT NULL,
            searcher_name TEXT,
            assigned_at INTEGER NOT NULL,
            timeframe_minutes INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            grace_expires_at INTEGER NOT NULL,
            status TEXT DEFAULT 'active',
            completion_percentage REAL DEFAULT 0,
            completed_at INTEGER
        )
    """)
    print("✓ 'grid_assignments' table created")

    # Table 3: search_progress
    print("\n3. Creating 'search_progress' table...")
    await db.execute("""
        CREATE TABLE IF NOT EXISTS search_progress (
            progress_id TEXT PRIMARY KEY,
            assignment_id TEXT NOT NULL,
            search_id TEXT NOT NULL,
            pet_id TEXT NOT NULL,
            grid_id INTEGER NOT NULL,
            searcher_id TEXT NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            timestamp INTEGER NOT NULL,
            accuracy_meters REAL
        )
    """)
    print("✓ 'search_progress' table created")

    # Table 4: roads_searched
    print("\n4. Creating 'roads_searched' table...")
    await db.execute("""
        CREATE TABLE IF NOT EXISTS roads_searched (
            road_search_id TEXT PRIMARY KEY,
            assignment_id TEXT NOT NULL,
            search_id TEXT NOT NULL,
            pet_id TEXT NOT NULL,
            grid_id INTEGER NOT NULL,
            searcher_id TEXT NOT NULL,
            road_id TEXT NOT NULL,
            road_name TEXT NOT NULL,
            searched_at INTEGER NOT NULL,
            last_searched_at INTEGER,
            search_count INTEGER DEFAULT 1
        )
    """)
    print("✓ 'roads_searched' table created")

    # Create indexes
    print("\n5. Creating indexes...")

    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_searches_pet_id ON searches(pet_id)",
        "CREATE INDEX IF NOT EXISTS idx_searches_status ON searches(status)",
        "CREATE INDEX IF NOT EXISTS idx_assignments_search_id ON grid_assignments(search_id)",
        "CREATE INDEX IF NOT EXISTS idx_assignments_pet_id ON grid_assignments(pet_id)",
        "CREATE INDEX IF NOT EXISTS idx_assignments_searcher_id ON grid_assignments(searcher_id)",
        "CREATE INDEX IF NOT EXISTS idx_assignments_status ON grid_assignments(status)",
        "CREATE INDEX IF NOT EXISTS idx_progress_assignment_id ON search_progress(assignment_id)",
        "CREATE INDEX IF NOT EXISTS idx_progress_search_id ON search_progress(search_id)",
        "CREATE INDEX IF NOT EXISTS idx_progress_timestamp ON search_progress(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_roads_search_id ON roads_searched(search_id)",
        "CREATE INDEX IF NOT EXISTS idx_roads_assignment_id ON roads_searched(assignment_id)",
        "CREATE INDEX IF NOT EXISTS idx_roads_searched_at ON roads_searched(searched_at)",
    ]

    for idx_sql in indexes:
        await db.execute(idx_sql)

    print(f"✓ Created {len(indexes)} indexes")

    print("\n" + "="*50)
    print("✅ ALL TABLES CREATED SUCCESSFULLY!")
    print("="*50)

    # Verify tables exist
    print("\n6. Verifying tables...")
    result = await db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")

    if result.get('results'):
        print("\nTables in database:")
        for row in result['results']:
            print(f"  - {row['name']}")

    print("\n✅ Setup complete! The tracking system is ready to use.")

if __name__ == "__main__":
    asyncio.run(create_tables())
