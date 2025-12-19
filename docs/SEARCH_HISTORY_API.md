# Search History API Documentation

## Overview

The search history feature allows users to view their past search sessions, including GPS routes, distance traveled, and duration for each completed search.

## Backend Implementation ✅

All backend components are now **LIVE** and ready to use:

### 1. Database Schema
- ✅ `lost_pets` table created (stores pet name and photo URL)
- ✅ `search_progress` table stores GPS breadcrumbs (already existed)
- ✅ `grid_assignments` table tracks search sessions (already existed)

### 2. Endpoints

#### GET /api/search-history
**Status:** ✅ LIVE and tested

**Endpoint:** `https://api.psar.app/api/search-history?searcher_id={userId}`

**Headers Required:**
```
X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67
```

**Query Parameters:**
- `searcher_id` (required): Unique identifier for the volunteer searcher

**Response Format:**
```json
{
  "success": true,
  "searcher_id": "user-abc-123",
  "total_searches": 2,
  "searches": [
    {
      "search_id": "search-uuid-123",
      "pet_id": "pet-456",
      "pet_name": "Buddy",
      "pet_photo_url": "https://example.com/buddy.jpg",
      "searched_at": 1699024200,
      "completed_at": 1699026900,
      "total_distance_miles": 2.5,
      "duration_minutes": 45,
      "route": [
        {"lat": 37.7749, "lon": -122.4194},
        {"lat": 37.7750, "lon": -122.4195},
        {"lat": 37.7751, "lon": -122.4196}
      ]
    }
  ]
}
```

**Response Fields:**
- `search_id`: Unique identifier for the search
- `pet_id`: Unique identifier for the pet
- `pet_name`: Name of the lost pet (or "Unknown Pet" if not set)
- `pet_photo_url`: URL to pet's photo (nullable)
- `searched_at`: Unix timestamp when search started
- `completed_at`: Unix timestamp when search completed
- `total_distance_miles`: Total distance traveled during search
- `duration_minutes`: Total time spent searching (in minutes)
- `route`: Array of GPS coordinates in chronological order

**Notes:**
- Only returns **completed** searches (where status='completed' and completed_at is not null)
- Results are sorted by completion date (most recent first)
- Empty array if no completed searches found

---

## iOS App Integration Required

### 1. Pet Details Storage

**When creating a search** (calling `/api/create-search`), you also need to store pet details:

**New Endpoint to Call:**
```
POST https://api.psar.app/api/save-pet-details
```

**Request Body:**
```json
{
  "pet_id": "hash-of-apple-id-plus-date",
  "pet_name": "Buddy",
  "pet_photo_url": "https://your-storage/photo.jpg"
}
```

**Implementation Note:**
Currently, the iOS app needs to call a new endpoint to save pet details. Add this method to `database.py`:

```python
@app.post("/api/save-pet-details", dependencies=[Depends(verify_api_key)])
async def save_pet_details_endpoint(request: PetDetailsRequest):
    """Save pet name and photo URL"""
    try:
        pet_id = await db.save_lost_pet(
            request.pet_id,
            request.pet_name,
            request.pet_photo_url
        )
        return {
            'success': True,
            'pet_id': pet_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**When to call:**
- When user creates a lost pet report
- When user updates pet information

### 2. GPS Tracking (Already Working) ✅

The `/api/update-progress` endpoint **already stores GPS breadcrumbs** automatically.

**Current Implementation:**
- App calls `/api/update-progress` every 30 seconds
- Each call stores: lat, lon, timestamp, accuracy_meters, distance_miles, elapsed_minutes
- All GPS points are linked to the assignment_id

**No changes needed** - GPS tracking is working as expected!

### 3. Search Session Completion

**Ensure you mark searches as complete** when the user finishes:

The `grid_assignments` table tracks this automatically when completion_percentage >= 85%.

**No changes needed** - assignment completion is handled by the backend!

---

## Example iOS Code

### Fetching Search History

```swift
func fetchSearchHistory(for searcherID: String) async throws -> [SearchSession] {
    let url = URL(string: "https://api.psar.app/api/search-history?searcher_id=\(searcherID)")!
    var request = URLRequest(url: url)
    request.setValue("petsearch_2024_secure_key_f8d92a1b3c4e5f67", forHTTPHeaderField: "X-API-Key")

    let (data, response) = try await URLSession.shared.data(for: request)

    guard let httpResponse = response as? HTTPURLResponse,
          httpResponse.statusCode == 200 else {
        throw NetworkError.invalidResponse
    }

    let decoder = JSONDecoder()
    let searchHistory = try decoder.decode(SearchHistoryResponse.self, from: data)

    return searchHistory.searches
}
```

### Data Models

```swift
struct SearchHistoryResponse: Codable {
    let success: Bool
    let searcherID: String
    let totalSearches: Int
    let searches: [SearchSession]

    enum CodingKeys: String, CodingKey {
        case success
        case searcherID = "searcher_id"
        case totalSearches = "total_searches"
        case searches
    }
}

struct SearchSession: Codable {
    let searchID: String
    let petID: String
    let petName: String
    let petPhotoURL: String?
    let searchedAt: Int
    let completedAt: Int?
    let totalDistanceMiles: Double
    let durationMinutes: Int
    let route: [GPSPoint]

    enum CodingKeys: String, CodingKey {
        case searchID = "search_id"
        case petID = "pet_id"
        case petName = "pet_name"
        case petPhotoURL = "pet_photo_url"
        case searchedAt = "searched_at"
        case completedAt = "completed_at"
        case totalDistanceMiles = "total_distance_miles"
        case durationMinutes = "duration_minutes"
        case route
    }
}

struct GPSPoint: Codable {
    let lat: Double
    let lon: Double
}
```

---

## Summary for iOS Team

### ✅ Backend Ready
1. Database tables created
2. GET `/api/search-history` endpoint live
3. GPS tracking already working via `/api/update-progress`

### 📱 iOS Tasks Remaining

1. **Add Pet Details Submission** (NEW)
   - When user creates a pet report, call new endpoint to save pet_name and pet_photo_url
   - Backend method exists (`db.save_lost_pet`), just needs endpoint wrapper
   - See "Pet Details Storage" section above for implementation

2. **Integrate Search History UI** (NEW)
   - Call GET `/api/search-history?searcher_id={userId}`
   - Display list of completed searches with pet details
   - Show route on map using the GPS coordinates array
   - Display distance and duration stats

3. **Test End-to-End** (REQUIRED)
   - Complete a search session
   - Verify it appears in search history
   - Verify pet name and photo display correctly
   - Verify GPS route displays correctly

---

## Testing

### Manual Testing

1. **Create a test pet with details:**
```bash
curl -k -X POST https://localhost:8443/api/save-pet-details \
  -H "X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67" \
  -H "Content-Type: application/json" \
  -d '{
    "pet_id": "test-pet-123",
    "pet_name": "Test Dog",
    "pet_photo_url": "https://example.com/dog.jpg"
  }'
```

2. **Create a search and complete an assignment**
   (Use existing endpoints)

3. **Fetch search history:**
```bash
curl -k -H "X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67" \
  "https://localhost:8443/api/search-history?searcher_id=test-user-123"
```

---

## Database Schema Reference

### lost_pets table
```sql
CREATE TABLE lost_pets (
    pet_id TEXT PRIMARY KEY,
    pet_name TEXT NOT NULL,
    pet_photo_url TEXT,
    created_at INTEGER NOT NULL,
    updated_at INTEGER,
    status TEXT DEFAULT 'lost'
);
```

### Existing Tables (Already in Use)

**search_progress** - GPS breadcrumbs
- Stores every GPS point sent via `/api/update-progress`
- Linked to assignment_id

**grid_assignments** - Search sessions
- Tracks when searcher started/completed
- Stores completion status and percentage
- Linked to searcher_id, search_id, pet_id

---

## Support

If you encounter issues:
1. Check API key is included in headers
2. Verify searcher_id format matches what's stored in grid_assignments
3. Ensure searches are marked as completed (status='completed')
4. Check backend logs: `sudo journalctl -u petsearch.service -f`

Backend is live at: `https://api.psar.app`
