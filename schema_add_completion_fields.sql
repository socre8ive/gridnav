-- Add completion tracking fields to grid_assignments table
-- These fields store the final distance and duration when a search is completed

-- Add total_distance_miles column if it doesn't exist
ALTER TABLE grid_assignments ADD COLUMN total_distance_miles REAL DEFAULT 0;

-- Add duration_minutes column if it doesn't exist
ALTER TABLE grid_assignments ADD COLUMN duration_minutes INTEGER DEFAULT 0;
