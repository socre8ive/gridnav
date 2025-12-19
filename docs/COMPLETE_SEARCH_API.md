# Complete Search API Documentation

## Overview

The `/api/complete-search` endpoint allows iOS app to mark a search session as completed with final statistics (distance and duration). This ensures the search appears in the user's search history with accurate metrics.

## Endpoint

### POST /api/complete-search

**Status:** ✅ LIVE and tested

**URL:** `https://api.psar.app/api/complete-search`

**Method:** POST

**Authentication:** Required - X-API-Key header

**Headers:**
```
X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67
Content-Type: application/json
```

## Request Body

```json
{
  "search_id": "abc123",
  "pet_id": "456",
  "searcher_id": "user789",
  "total_distance_miles": 2.5,
  "duration_minutes": 45
}
```

### Parameters

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `search_id` | string | Yes | Unique identifier for the search |
| `pet_id` | string | Yes | Unique identifier for the lost pet |
| `searcher_id` | string | Yes | Unique identifier for the volunteer searcher |
| `total_distance_miles` | float | Yes | Total distance traveled during search (in miles) |
| `duration_minutes` | int | Yes | Total time spent searching (in minutes) |

## Response

### Success Response

**Status Code:** 200 OK

```json
{
  "success": true,
  "message": "Search completed and added to history",
  "assignment_id": "assign-d3086ef0-f0a5-4fea-97f5-334762958d96",
  "total_distance_miles": 2.5,
  "duration_minutes": 45
}
```

### Error Response

**Status Code:** 500 Internal Server Error

```json
{
  "detail": "No assignment found for searcher user789 on search abc123"
}
```

**Common Errors:**
- No assignment found: The searcher was never assigned to this search
- Invalid search_id: Search doesn't exist
- Missing required fields: One or more required parameters not provided

## What It Does

When this endpoint is called, it:

1. **Finds the most recent assignment** for the given searcher/search combination
2. **Updates the assignment** in `grid_assignments` table:
   - Sets `status = 'completed'`
   - Sets `completed_at = NOW()`
   - Sets `total_distance_miles = {value}`
   - Sets `duration_minutes = {value}`
   - Sets `completion_percentage = 100`
3. **Returns confirmation** with the stored values

This marks the search as completed, making it appear in `/api/search-history` results.

## Database Changes

### Schema Updates

Added two new columns to `grid_assignments` table:

```sql
ALTER TABLE grid_assignments ADD COLUMN total_distance_miles REAL DEFAULT 0;
ALTER TABLE grid_assignments ADD COLUMN duration_minutes INTEGER DEFAULT 0;
```

### Updated Table Structure

```sql
grid_assignments:
  - assignment_id (PK)
  - search_id
  - pet_id
  - searcher_id
  - assigned_at
  - completed_at
  - status ('active', 'completed', 'expired')
  - total_distance_miles  -- NEW
  - duration_minutes      -- NEW
  - completion_percentage
  ... other fields
```

## iOS Integration

### When to Call

Call this endpoint when:
- User manually ends their search session
- Search timer expires
- User completes their assigned grid(s)
- App moves to background and search is ending

### Example Swift Code

```swift
struct CompleteSearchRequest: Codable {
    let searchId: String
    let petId: String
    let searcherId: String
    let totalDistanceMiles: Double
    let durationMinutes: Int

    enum CodingKeys: String, CodingKey {
        case searchId = "search_id"
        case petId = "pet_id"
        case searcherId = "searcher_id"
        case totalDistanceMiles = "total_distance_miles"
        case durationMinutes = "duration_minutes"
    }
}

struct CompleteSearchResponse: Codable {
    let success: Bool
    let message: String
    let assignmentId: String
    let totalDistanceMiles: Double
    let durationMinutes: Int

    enum CodingKeys: String, CodingKey {
        case success, message
        case assignmentId = "assignment_id"
        case totalDistanceMiles = "total_distance_miles"
        case durationMinutes = "duration_minutes"
    }
}

func completeSearch(searchId: String, petId: String, searcherId: String,
                   distance: Double, duration: Int) async throws {
    let url = URL(string: "https://api.psar.app/api/complete-search")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("petsearch_2024_secure_key_f8d92a1b3c4e5f67",
                     forHTTPHeaderField: "X-API-Key")
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")

    let body = CompleteSearchRequest(
        searchId: searchId,
        petId: petId,
        searcherId: searcherId,
        totalDistanceMiles: distance,
        durationMinutes: duration
    )

    request.httpBody = try JSONEncoder().encode(body)

    let (data, response) = try await URLSession.shared.data(for: request)

    guard let httpResponse = response as? HTTPURLResponse,
          httpResponse.statusCode == 200 else {
        throw NetworkError.invalidResponse
    }

    let result = try JSONDecoder().decode(CompleteSearchResponse.self, from: data)
    print("Search completed: \(result.message)")
}
```

### Usage Example

```swift
// When user ends their search
Task {
    do {
        try await completeSearch(
            searchId: currentSearch.id,
            petId: currentSearch.petId,
            searcherId: UserDefaults.userId,
            distance: locationManager.totalDistance,
            duration: Int(searchTimer.elapsedMinutes)
        )

        // Now show success message or navigate to history
        showSearchHistory()

    } catch {
        print("Failed to complete search: \(error)")
        // Handle error appropriately
    }
}
```

## Workflow

### Complete Search Flow

```
iOS App                     Backend API                 Database
   |                            |                           |
   |  POST /api/complete-search |                           |
   |--------------------------->|                           |
   |                            |  Find assignment          |
   |                            |-------------------------->|
   |                            |  UPDATE grid_assignments  |
   |                            |  SET status='completed'   |
   |                            |  SET total_distance=2.5   |
   |                            |  SET duration=45          |
   |                            |  SET completed_at=NOW()   |
   |                            |-------------------------->|
   |                            |<--------------------------|
   |  Success response          |                           |
   |<---------------------------|                           |
   |                            |                           |
   |  GET /api/search-history   |                           |
   |--------------------------->|                           |
   |                            |  SELECT from assignments  |
   |                            |  WHERE status='completed' |
   |                            |-------------------------->|
   |                            |<--------------------------|
   |  [Completed search data]   |                           |
   |<---------------------------|                           |
```

## Integration with Search History

Once `/api/complete-search` is called:

1. The search becomes visible in `/api/search-history`
2. The stored `total_distance_miles` and `duration_minutes` are used in history
3. GPS route is still available from `search_progress` table

**Example:**

```bash
# Complete the search
curl -X POST https://api.psar.app/api/complete-search \
  -H "X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67" \
  -H "Content-Type: application/json" \
  -d '{
    "search_id": "search-123",
    "pet_id": "pet-456",
    "searcher_id": "user-789",
    "total_distance_miles": 2.5,
    "duration_minutes": 45
  }'

# Immediately fetch history
curl -H "X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67" \
  "https://api.psar.app/api/search-history?searcher_id=user-789"

# Response includes the completed search
{
  "searches": [
    {
      "search_id": "search-123",
      "total_distance_miles": 2.5,
      "duration_minutes": 45,
      "route": [...GPS points...]
    }
  ]
}
```

## Testing

### Manual Testing

See `/opt/petsearch/test_http_complete_search.sh` for a complete test script.

**Quick Test:**

```bash
# Run the comprehensive test
cd /opt/petsearch
./test_http_complete_search.sh
```

**Expected Output:**
```
1. Creating pet details... ✓
2. Creating search and assignment... ✓
3. Completing search via HTTP endpoint... ✓
   {
     "success": true,
     "message": "Search completed and added to history"
   }
4. Fetching search history... ✓
   {
     "total_searches": 1,
     "searches": [...]
   }
```

## Error Handling

### No Assignment Found

**Cause:** The searcher was never assigned to this search, or the search doesn't exist.

**Solution:**
- Ensure `/api/assign-grid` was called before completing
- Verify search_id and searcher_id are correct
- Check that the assignment hasn't already been completed

### Database Error

**Cause:** Connection issue or invalid data

**Solution:**
- Check backend logs: `sudo journalctl -u petsearch.service -f`
- Verify all required fields are provided
- Ensure values are within valid ranges (distance >= 0, duration >= 0)

## Backend Files Modified

1. **database.py** - Added `complete_search()` method (lines 886-935)
2. **server_geographic_grids.py** - Added:
   - `CompleteSearchRequest` model (lines 1944-1949)
   - POST `/api/complete-search` endpoint (lines 2198-2235)
3. **Schema migration** - `schema_add_completion_fields.sql`

## Summary

✅ **Endpoint:** POST `/api/complete-search`
✅ **Status:** Live and tested
✅ **Purpose:** Mark search as completed with final statistics
✅ **Result:** Search appears in history with accurate distance/duration

**Next Step:** iOS team should call this endpoint when user completes a search session, then fetch updated history.
