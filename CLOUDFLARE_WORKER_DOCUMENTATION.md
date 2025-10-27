# Pet Search Grid System - Cloudflare Worker & iOS Integration Guide

## System Overview

This system coordinates volunteer searches for lost pets using a geographic grid system. The EC2 API creates grids, assigns them to searchers, and tracks progress in real-time.

**Architecture:**
- **iOS App** → **Cloudflare Workers** → **EC2 API (http://54.163.97.184:8109)** → **Cloudflare D1 Database**

---

## Database Setup

### Step 1: Create D1 Database Tables

Run the SQL schema on your Cloudflare D1 database. The schema is in `/opt/petsearch/schema_search_tracking.sql`.

**Tables Created:**
- `searches` - Overall search info per lost pet
- `grid_assignments` - Who's searching which grid
- `search_progress` - GPS tracking breadcrumbs
- `roads_searched` - Roads covered with timestamps

---

## EC2 API Endpoints

Base URL: `http://54.163.97.184:8109`

### 1. POST /api/create-search

Creates geographic grids for a search area and optionally saves to tracking database.

**Request:**
```json
{
  "lat": 27.8428,
  "lon": -82.8106,
  "radius_miles": 1.5,
  "grid_size_miles": 0.3,
  "address": "11388 86th Ave N, Seminole, FL 33772",
  "pet_id": "apple_user_123_20251021"
}
```

**Response:**
```json
{
  "search_id": "search-abc123...",
  "center": {"lat": 27.8428, "lon": -82.8106},
  "tiles": [
    {
      "id": "search-abc123-grid-1",
      "grid_id": 1,
      "road_count": 12,
      "total_distance_miles": 0.45,
      "estimated_minutes": 15,
      "bounds": {
        "min_lat": 27.84,
        "max_lat": 27.85,
        "min_lon": -82.82,
        "max_lon": -82.81
      },
      "center": {"lat": 27.845, "lon": -82.815},
      "road_details": [
        {
          "id": "search-abc123-road1",
          "name": "86th Ave N",
          "waypoints": [{"lat": 27.84, "lon": -82.81}, ...],
          "highway_type": "residential",
          "length_meters": 120.5,
          "has_name": true
        }
      ],
      "grid_size_miles": 0.3
    }
  ],
  "total_tiles": 45,
  "total_roads": 523,
  "filtered_count": 0,
  "grid_size_miles": 0.3,
  "message": "Created 45 geographic grids with 523 real roads"
}
```

**Notes:**
- Grids are numbered 1-N starting from center (closest to last sighting)
- Grid size should be 0.3 miles for iOS app
- If `pet_id` is provided, search is saved to tracking database

---

### 2. POST /api/assign-grid

Assigns grid(s) to a volunteer based on their available time.

**Request:**
```json
{
  "search_id": "search-abc123...",
  "pet_id": "apple_user_123_20251021",
  "searcher_id": "volunteer_xyz_789",
  "searcher_name": "John Smith",
  "timeframe_minutes": 60
}
```

**Timeframe Options:**
- `30` = 1 grid
- `60` = 2 grids
- `90` = 3 grids
- `120` = 4 grids

**Response:**
```json
{
  "success": true,
  "assignments": [
    {
      "assignment_id": "assign-def456...",
      "grid_id": 1,
      "expires_at": 1729521600,
      "grace_expires_at": 1729522200
    },
    {
      "assignment_id": "assign-ghi789...",
      "grid_id": 2,
      "expires_at": 1729521600,
      "grace_expires_at": 1729522200
    }
  ],
  "total_assigned": 2,
  "timeframe_minutes": 60
}
```

**Assignment Logic:**
- Assigns only **unsearched grids** (closest to center first)
- Grids stay assigned until `grace_expires_at` (timeframe + 10 minutes)
- Completed grids (85%+) become available again after 12 hours
- Returns error if not enough grids available

---

### 3. POST /api/update-progress

Updates searcher's GPS location and roads covered. Called every 30 seconds by iOS app.

**Request:**
```json
{
  "assignment_id": "assign-def456...",
  "search_id": "search-abc123...",
  "pet_id": "apple_user_123_20251021",
  "grid_id": 1,
  "searcher_id": "volunteer_xyz_789",
  "lat": 27.8432,
  "lon": -82.8098,
  "accuracy_meters": 10.5,
  "roads_covered": [
    {
      "road_id": "search-abc123-road1",
      "road_name": "86th Ave N"
    },
    {
      "road_id": "search-abc123-road2",
      "road_name": "Main St"
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "progress_id": "progress-jkl012...",
  "roads_marked": 2,
  "message": "Progress updated successfully"
}
```

**Notes:**
- `roads_covered` can be empty or contain newly completed roads
- System automatically calculates grid completion percentage
- Grid marked "completed" at 85% coverage

---

### 4. GET /api/grid-status

Returns current status of all grids for a search.

**Request:**
```
GET /api/grid-status?search_id=search-abc123...
```

**Response:**
```json
{
  "success": true,
  "search_id": "search-abc123...",
  "grids": [
    {
      "grid_id": 1,
      "searcher_id": "volunteer_xyz_789",
      "searcher_name": "John Smith",
      "assigned_at": 1729518000,
      "expires_at": 1729521600,
      "grace_expires_at": 1729522200,
      "status": "active",
      "completion_percentage": 45.5,
      "completed_at": null,
      "timeframe_minutes": 60
    },
    {
      "grid_id": 3,
      "searcher_id": "volunteer_abc_456",
      "searcher_name": "Jane Doe",
      "assigned_at": 1729518000,
      "expires_at": 1729519800,
      "grace_expires_at": 1729520400,
      "status": "completed",
      "completion_percentage": 100,
      "completed_at": 1729519500,
      "timeframe_minutes": 30
    }
  ],
  "total_grids": 2
}
```

**Status Values:**
- `active` - Currently being searched
- `completed` - 85%+ coverage achieved
- `expired` - Grace period passed without completion

---

## Cloudflare Worker Templates

### Worker 1: Grid Request Worker

Handles grid creation and retrieval for iOS app.

```javascript
export default {
  async fetch(request, env) {
    const EC2_API_BASE = 'http://54.163.97.184:8109';

    // CORS headers
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    try {
      const body = await request.json();

      // Forward request to EC2 API
      const response = await fetch(`${EC2_API_BASE}/api/create-search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          lat: body.lat || 0,
          lon: body.lon || 0,
          radius_miles: body.radius_miles || 1.5,
          grid_size_miles: 0.3,  // Fixed for iOS app
          address: body.address,
          pet_id: body.pet_id
        })
      });

      const data = await response.json();

      return new Response(JSON.stringify(data), {
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
        },
      });

    } catch (error) {
      return new Response(JSON.stringify({
        success: false,
        error: error.message
      }), {
        status: 500,
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
        },
      });
    }
  },
};
```

### Worker 2: Progress Update Worker

Handles real-time GPS tracking and progress updates.

```javascript
export default {
  async fetch(request, env) {
    const EC2_API_BASE = 'http://54.163.97.184:8109';

    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    try {
      const body = await request.json();
      const action = body.action; // 'assign' or 'update_progress'

      let endpoint;
      if (action === 'assign') {
        endpoint = '/api/assign-grid';
      } else if (action === 'update_progress') {
        endpoint = '/api/update-progress';
      } else {
        throw new Error('Invalid action. Must be "assign" or "update_progress"');
      }

      // Forward to EC2 API
      const response = await fetch(`${EC2_API_BASE}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body.data)
      });

      const data = await response.json();

      return new Response(JSON.stringify(data), {
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
        },
      });

    } catch (error) {
      return new Response(JSON.stringify({
        success: false,
        error: error.message
      }), {
        status: 500,
        headers: {
          ...corsHeaders,
          'Content-Type': 'application/json',
        },
      });
    }
  },
};
```

---

## Instructions for Claude (Cursor) - iOS App Development

### Part 1: Create Lost Pet Report Screen

"Create an iOS Swift app screen for reporting a lost pet. The screen should:

1. Collect the following information:
   - Pet name and description
   - Last seen address (use text field with geocoding)
   - Date/time last seen
   - Owner's contact info

2. Generate a unique `pet_id` using:
   ```swift
   let petId = "\\(AppleIDHash)_\\(currentDate)"
   ```

3. When user submits, make a POST request to Cloudflare Worker #1:
   ```swift
   struct CreateSearchRequest: Codable {
       let lat: Double
       let lon: Double
       let radius_miles: Double = 1.5
       let address: String
       let pet_id: String
   }
   ```

4. Store the returned `search_id` and `tiles` data locally for the search coordinator screen.

5. Use standard iOS geocoding to convert address to lat/lon before sending to worker."

---

### Part 2: Create Search Coordinator Screen

"Create an iOS Swift screen for coordinating volunteer searches:

1. Display a map showing:
   - Center point (last seen location)
   - All grids as rectangles color-coded by status:
     - Gray = unsearched
     - Yellow = in progress
     - Green = completed

2. Add a volunteer sign-up form:
   - Name field
   - Time available picker (30, 60, 90, or 120 minutes)
   - Submit button to request grid assignment

3. When volunteer signs up, generate a unique `searcher_id` and POST to Cloudflare Worker #2:
   ```swift
   struct AssignGridRequest: Codable {
       let action: String = \"assign\"
       let data: AssignData
   }

   struct AssignData: Codable {
       let search_id: String
       let pet_id: String
       let searcher_id: String
       let searcher_name: String
       let timeframe_minutes: Int
   }
   ```

4. Display assigned grids to the volunteer with:
   - Grid number(s)
   - Time remaining
   - Start search button

5. Poll `/api/grid-status?search_id=XXX` every 30 seconds to update map colors."

---

### Part 3: Create Active Search Screen

"Create an iOS Swift screen for volunteers actively searching:

1. Display:
   - Map showing assigned grid(s) with road overlays
   - Current GPS location (blue dot)
   - List of roads in grid with checkboxes
   - Progress bar showing completion percentage
   - Timer showing time remaining

2. Request location permissions and track GPS every 30 seconds:
   ```swift
   struct UpdateProgressRequest: Codable {
       let action: String = \"update_progress\"
       let data: ProgressData
   }

   struct ProgressData: Codable {
       let assignment_id: String
       let search_id: String
       let pet_id: String
       let grid_id: Int
       let searcher_id: String
       let lat: Double
       let lon: Double
       let accuracy_meters: Double?
       let roads_covered: [RoadCovered]?
   }

   struct RoadCovered: Codable {
       let road_id: String
       let road_name: String
   }
   ```

3. Use CoreLocation to detect when user is on/near a road and automatically check it off.

4. When user manually checks off a road, include it in next progress update.

5. Show notification when grid reaches 85% completion.

6. Include 'Finish Early' button that saves partial progress and returns to coordinator screen."

---

## Testing the System

### 1. Test Grid Creation

```bash
curl -X POST http://54.163.97.184:8109/api/create-search \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 27.8428,
    "lon": -82.8106,
    "radius_miles": 1.5,
    "grid_size_miles": 0.3,
    "address": "11388 86th Ave N, Seminole, FL 33772",
    "pet_id": "test_user_123_20251021"
  }'
```

### 2. Test Grid Assignment

```bash
curl -X POST http://54.163.97.184:8109/api/assign-grid \
  -H "Content-Type: application/json" \
  -d '{
    "search_id": "search-XXXXX",
    "pet_id": "test_user_123_20251021",
    "searcher_id": "volunteer_001",
    "searcher_name": "Test Volunteer",
    "timeframe_minutes": 60
  }'
```

### 3. Test Progress Update

```bash
curl -X POST http://54.163.97.184:8109/api/update-progress \
  -H "Content-Type: application/json" \
  -d '{
    "assignment_id": "assign-XXXXX",
    "search_id": "search-XXXXX",
    "pet_id": "test_user_123_20251021",
    "grid_id": 1,
    "searcher_id": "volunteer_001",
    "lat": 27.8430,
    "lon": -82.8100,
    "accuracy_meters": 10.0,
    "roads_covered": []
  }'
```

### 4. Test Grid Status

```bash
curl "http://54.163.97.184:8109/api/grid-status?search_id=search-XXXXX"
```

---

## System Flow Summary

1. **User reports lost pet** → iOS app sends to Worker #1 → Creates grids on EC2 → Returns grid data
2. **Volunteer signs up** → iOS app sends to Worker #2 → Assigns grids → Returns assignment IDs
3. **Volunteer searches** → iOS app sends GPS every 30s to Worker #2 → Updates progress → Tracks roads
4. **Grid completion** → When 85% reached → Marked complete → Available again after 12 hours
5. **Real-time updates** → iOS app polls grid-status → Updates map colors → Shows progress

---

## Key Features

✅ Circular grid numbering (1, 2, 3... from center outward)
✅ Fixed 0.3 mile grids
✅ Automatic grid assignment (closest first)
✅ 10 minute grace period on expiration
✅ 85% completion threshold
✅ 12 hour re-search availability
✅ Real-time GPS tracking every 30 seconds
✅ Road-level progress tracking
✅ Multi-volunteer coordination
✅ Cloudflare D1 database storage

---

## Contact & Support

- EC2 Server: http://54.163.97.184
- API Port: 8109
- Database: Cloudflare D1 (Account ID: f11101e2d3a37c630bcf73d6f8792bcb)
