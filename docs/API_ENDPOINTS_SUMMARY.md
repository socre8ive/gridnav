# Search History & Completion API - Complete Summary

## 🎯 Overview

Three new endpoints have been implemented to support search history tracking in the iOS app:

1. **POST /api/save-pet-details** - Save lost pet information
2. **POST /api/complete-search** - Mark search as completed (REQUIRED)
3. **GET /api/search-history** - Retrieve completed searches

**Status:** ✅ All endpoints are LIVE and tested on https://api.psar.app

---

## 📋 API Endpoints

### 1. POST /api/save-pet-details

**Purpose:** Save pet name and photo URL when user creates a lost pet report

**URL:** `https://api.psar.app/api/save-pet-details`

**Request:**
```json
{
  "pet_id": "unique-pet-id",
  "pet_name": "Buddy",
  "pet_photo_url": "https://storage.example.com/photo.jpg"
}
```

**Response:**
```json
{
  "success": true,
  "pet_id": "unique-pet-id",
  "message": "Pet details saved successfully"
}
```

**When to call:** When user creates or updates a lost pet report

---

### 2. POST /api/complete-search ⚠️ REQUIRED

**Purpose:** Mark search session as completed with final statistics

**URL:** `https://api.psar.app/api/complete-search`

**Request:**
```json
{
  "search_id": "abc123",
  "pet_id": "456",
  "searcher_id": "user789",
  "total_distance_miles": 2.5,
  "duration_minutes": 45
}
```

**Response:**
```json
{
  "success": true,
  "message": "Search completed and added to history",
  "assignment_id": "assign-xyz",
  "total_distance_miles": 2.5,
  "duration_minutes": 45
}
```

**⚠️ CRITICAL:** This endpoint MUST be called when user finishes searching. Without calling this endpoint, the search will NOT appear in history!

**When to call:**
- User manually ends search session
- Search timer expires
- User completes assigned grid(s)
- App moves to background and search is ending

---

### 3. GET /api/search-history

**Purpose:** Retrieve all completed searches for a user

**URL:** `https://api.psar.app/api/search-history?searcher_id={userId}`

**Response:**
```json
{
  "success": true,
  "searcher_id": "user789",
  "total_searches": 2,
  "searches": [
    {
      "search_id": "search-123",
      "pet_id": "pet-456",
      "pet_name": "Buddy",
      "pet_photo_url": "https://example.com/buddy.jpg",
      "searched_at": 1699024200,
      "completed_at": 1699026900,
      "total_distance_miles": 2.5,
      "duration_minutes": 45,
      "route": [
        {"lat": 37.7749, "lon": -122.4194},
        {"lat": 37.7750, "lon": -122.4195}
      ]
    }
  ]
}
```

**When to call:** When user wants to view their search history

---

## 🔄 Complete Workflow

### iOS App Flow

```
1. User creates lost pet report
   ↓
   POST /api/save-pet-details
   {pet_id, pet_name, pet_photo_url}
   ✓ Pet details stored

2. User starts searching
   ↓
   POST /api/assign-grid (existing endpoint)
   ✓ Search session begins

3. GPS tracking (every 30 seconds)
   ↓
   POST /api/update-progress (existing endpoint)
   ✓ GPS breadcrumbs saved

4. User finishes searching
   ↓
   POST /api/complete-search ⚠️ REQUIRED
   {search_id, pet_id, searcher_id, distance, duration}
   ✓ Search marked as completed

5. User views search history
   ↓
   GET /api/search-history?searcher_id={userId}
   ✓ Returns list of completed searches
```

---

## 🗄️ Database Changes

### New Table: lost_pets

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

### Updated Table: grid_assignments

Added two new columns:

```sql
ALTER TABLE grid_assignments
  ADD COLUMN total_distance_miles REAL DEFAULT 0;

ALTER TABLE grid_assignments
  ADD COLUMN duration_minutes INTEGER DEFAULT 0;
```

These store the final statistics when `/api/complete-search` is called.

---

## 📱 iOS Implementation Guide

### Step 1: Save Pet Details

```swift
func savePetDetails(petId: String, name: String, photoUrl: String?) async throws {
    let url = URL(string: "https://api.psar.app/api/save-pet-details")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("petsearch_2024_secure_key_f8d92a1b3c4e5f67",
                     forHTTPHeaderField: "X-API-Key")
    request.setValue("application/json",
                     forHTTPHeaderField: "Content-Type")

    let body: [String: Any] = [
        "pet_id": petId,
        "pet_name": name,
        "pet_photo_url": photoUrl ?? ""
    ]
    request.httpBody = try JSONSerialization.data(withJSONObject: body)

    let (_, response) = try await URLSession.shared.data(for: request)
    guard let httpResponse = response as? HTTPURLResponse,
          httpResponse.statusCode == 200 else {
        throw NetworkError.invalidResponse
    }
}
```

### Step 2: Complete Search (REQUIRED)

```swift
func completeSearch(searchId: String, petId: String, searcherId: String,
                   distance: Double, duration: Int) async throws {
    let url = URL(string: "https://api.psar.app/api/complete-search")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("petsearch_2024_secure_key_f8d92a1b3c4e5f67",
                     forHTTPHeaderField: "X-API-Key")
    request.setValue("application/json",
                     forHTTPHeaderField: "Content-Type")

    let body: [String: Any] = [
        "search_id": searchId,
        "pet_id": petId,
        "searcher_id": searcherId,
        "total_distance_miles": distance,
        "duration_minutes": duration
    ]
    request.httpBody = try JSONSerialization.data(withJSONObject: body)

    let (_, response) = try await URLSession.shared.data(for: request)
    guard let httpResponse = response as? HTTPURLResponse,
          httpResponse.statusCode == 200 else {
        throw NetworkError.invalidResponse
    }
}
```

### Step 3: Fetch Search History

```swift
struct SearchHistoryResponse: Codable {
    let success: Bool
    let searcherId: String
    let totalSearches: Int
    let searches: [SearchSession]

    enum CodingKeys: String, CodingKey {
        case success
        case searcherId = "searcher_id"
        case totalSearches = "total_searches"
        case searches
    }
}

func fetchSearchHistory(searcherId: String) async throws -> [SearchSession] {
    let urlString = "https://api.psar.app/api/search-history?searcher_id=\(searcherId)"
    let url = URL(string: urlString)!

    var request = URLRequest(url: url)
    request.setValue("petsearch_2024_secure_key_f8d92a1b3c4e5f67",
                     forHTTPHeaderField: "X-API-Key")

    let (data, _) = try await URLSession.shared.data(for: request)
    let response = try JSONDecoder().decode(SearchHistoryResponse.self, from: data)

    return response.searches
}
```

---

## ✅ Testing

### Test Script

Complete end-to-end test is available:

```bash
cd /opt/petsearch
./test_http_complete_search.sh
```

This test:
1. Creates pet details ✓
2. Creates search and assignment ✓
3. Adds GPS tracking points ✓
4. Calls `/api/complete-search` ✓
5. Fetches search history ✓
6. Verifies data matches ✓

### Manual Testing with curl

```bash
# 1. Save pet details
curl -k -X POST https://api.psar.app/api/save-pet-details \
  -H "X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67" \
  -H "Content-Type: application/json" \
  -d '{"pet_id":"test-123","pet_name":"Max","pet_photo_url":"https://example.com/max.jpg"}'

# 2. Complete a search (requires existing assignment)
curl -k -X POST https://api.psar.app/api/complete-search \
  -H "X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67" \
  -H "Content-Type: application/json" \
  -d '{"search_id":"search-123","pet_id":"test-123","searcher_id":"user-456","total_distance_miles":2.5,"duration_minutes":45}'

# 3. Get search history
curl -k "https://api.psar.app/api/search-history?searcher_id=user-456" \
  -H "X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67"
```

---

## 📚 Documentation

Detailed documentation available:

1. **SEARCH_HISTORY_API.md** - Complete search history API docs
2. **COMPLETE_SEARCH_API.md** - Complete search endpoint details
3. **IMPLEMENTATION_SUMMARY.md** - Full technical implementation details

Location: `/opt/petsearch/docs/`

---

## ⚠️ Important Notes

### MUST Call complete-search

**The iOS app MUST call `/api/complete-search` when the user finishes searching!**

Without this call:
- ❌ Search will NOT appear in history
- ❌ Distance and duration will not be stored
- ❌ User won't see their completed searches

### GPS Tracking Still Works

The existing `/api/update-progress` endpoint continues to work:
- ✅ GPS breadcrumbs are stored automatically
- ✅ No changes needed to GPS tracking code
- ✅ Routes are available in search history

### Data Flow

```
Pet Details → lost_pets table
    ↓
Search Session → grid_assignments table
    ↓
GPS Tracking → search_progress table (every 30 seconds)
    ↓
Complete Search → Updates grid_assignments
    ↓
Search History → Joins all three tables
```

---

## 🔧 Backend Files Modified

1. **database.py**
   - Added `save_lost_pet()` method
   - Added `complete_search()` method
   - Updated `get_search_history()` method

2. **server_geographic_grids.py**
   - Added `PetDetailsRequest` model
   - Added `CompleteSearchRequest` model
   - Added POST `/api/save-pet-details` endpoint
   - Added POST `/api/complete-search` endpoint
   - Added GET `/api/search-history` endpoint

3. **Schema Files**
   - `schema_lost_pets.sql` - New table for pet details
   - `schema_add_completion_fields.sql` - Added columns to grid_assignments

---

## 🎯 Summary

✅ **Backend:** 100% complete and tested
✅ **Endpoints:** All three live on production
✅ **Database:** Schema updated and verified
✅ **Testing:** End-to-end tests passing

**iOS Team Action Items:**

1. Integrate POST `/api/save-pet-details` when creating pet reports
2. **CRITICAL:** Integrate POST `/api/complete-search` when ending searches
3. Integrate GET `/api/search-history` for displaying user's history
4. Test end-to-end workflow
5. Display search history UI with routes on map

**Support:**
- Backend logs: `sudo journalctl -u petsearch.service -f`
- Test script: `/opt/petsearch/test_http_complete_search.sh`
- API base URL: `https://api.psar.app`
