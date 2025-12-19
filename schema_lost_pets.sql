-- Lost Pets Information Table
-- Stores details about lost pets for display in search history

CREATE TABLE IF NOT EXISTS lost_pets (
    pet_id TEXT PRIMARY KEY,              -- Unique pet identifier (from iOS)
    pet_name TEXT NOT NULL,               -- Name of the lost pet
    pet_photo_url TEXT,                   -- URL to pet's photo
    created_at INTEGER NOT NULL,          -- Unix timestamp when pet was reported
    updated_at INTEGER,                   -- Last update timestamp
    status TEXT DEFAULT 'lost'            -- lost, found, searching, etc.
);

-- Index for quick lookups
CREATE INDEX IF NOT EXISTS idx_lost_pets_status ON lost_pets(status);
CREATE INDEX IF NOT EXISTS idx_lost_pets_created_at ON lost_pets(created_at);
