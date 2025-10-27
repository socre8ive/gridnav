# iOS App Integration Notes - Grid System Changes

## Critical Changes

### 1. Grid Generation is Now Asynchronous (BREAKING CHANGE)

**OLD**: `/api/create-search` returned grids immediately (or timed out)
**NEW**: Returns immediately with `status: "pending"`, grids generated in background

**iOS App Must**:
- Call `/api/create-search` and receive `search_id`
- Poll `/api/search-by-pet` or `/api/grid-status` every 5 seconds
- Wait for `status` to change from `"pending"` to `"active"`
- Expect 60-90 seconds for generation to complete
- Handle timeout if >120 seconds (show error, retry option)

**Example Flow**:
```swift
// 1. Create search
POST /api/create-search
Response: {"success": true, "search_id": "search-abc", "status": "pending"}

// 2. Poll for completion (every 5 seconds)
GET /api/search-by-pet?pet_id=84
Response: {"success": true, "search_id": "search-abc", "status": "pending"}
... wait 5 seconds ...
Response: {"success": true, "search_id": "search-abc", "status": "active", "total_grids": 62}

// 3. Now fetch grids
POST /api/get-grids {"search_id": "search-abc"}
```

### 2. Many More Grids (UI Impact)

**OLD**: 6-10 large grids (~6 miles each)
**NEW**: 62 grids total
- 6 large grids (3-6 miles) - optimal for volunteers
- 56 small clusters (0.01-2 miles) - disconnected road segments

**iOS App Should**:
- Display grid count as "6 main grids + 56 small areas" or similar
- Prioritize assigning large grids first (filter `total_miles >= 3.0`)
- Show small grids as optional "bonus coverage"
- Consider combining small adjacent clusters in UI

**Grid Size Distribution**:
```
≥6 miles: 5 grids (Grid 1, 6, 7, 9, 13)
3-6 miles: 1 grid (Grid 34)
1-3 miles: 5 grids
<1 mile: 51 grids
```

### 3. All Roads Now Captured (100% Coverage)

**OLD**: ~23% of roads were skipped (disconnected segments)
**NEW**: 100% of roads assigned to grids

**Impact**: No more missing coverage, but creates many tiny grids for isolated streets

## API Endpoints (No URL Changes)

All endpoints same, but behavior changes:

### POST /api/create-search
**NEW Response**:
```json
{
    "success": true,
    "search_id": "search-uuid",
    "status": "pending",  // ← NEW: Always "pending" now
    "message": "Search accepted, grids are being generated in background"
}
```

**OLD Response** (no longer returned):
```json
{
    "success": true,
    "search_id": "search-uuid",
    "grids": [...]  // ← Removed, fetch separately after status="active"
}
```

### GET /api/search-by-pet?pet_id={id}
**NEW Response Includes Status**:
```json
{
    "success": true,
    "search_id": "search-uuid",
    "status": "pending",  // ← Check this: "pending" | "active" | "failed"
    "total_grids": 62     // ← Only present when status="active"
}
```

### POST /api/get-grids
**NEW: More Grids Returned**:
```json
{
    "success": true,
    "grids": [
        {
            "id": 1,
            "total_miles": 6.05,
            "roads_count": 198,
            "bounds": {...},
            "center": {...},
            "status": "available"
        },
        // ... 61 more grids
    ],
    "total_grids": 62  // ← Was 6-10, now typically 50-70
}
```

### POST /api/assign-grid
**CRITICAL - Grids Not Ready Response**:

When grids are still being generated (or race condition during save), returns **HTTP 200** (NOT 400):
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

**iOS Must**:
- Always check `success` field in response (even if HTTP 200)
- If `success: false` and `error: "grids_not_ready"`, wait 5 seconds and retry
- Do NOT treat as failure - this is normal during generation
- Show "Preparing grids..." message to user
- Max 20 retries (100 seconds total) before showing error

**Successful Assignment Response**:
```json
{
    "success": true,
    "assignments": [
        {
            "assignment_id": "assign-uuid",
            "grid_id": 1,
            "total_miles": 6.05
        }
    ],
    "total_assigned": 1,
    "timeframe_minutes": 30
}
```

**Error Response** (legitimate failure - not enough grids):
```json
HTTP 400
{
    "detail": "Not enough available grids. Requested: 4, Available: 2"
}
```

## Recommended iOS Changes

### 1. Add Loading State After Search Creation
```swift
// Show spinner/progress view for 60-90 seconds
// Poll every 5 seconds
// Show "Generating search grids..." message
// Cancel button to abort
```

### 2. Filter Grids by Size in Assignment UI
```swift
// Only show grids with ≥3 miles for primary assignment
let primaryGrids = grids.filter { $0.total_miles >= 3.0 }

// Show small grids as separate section
let bonusGrids = grids.filter { $0.total_miles < 3.0 }
```

### 3. Handle Grid Count in UI
```swift
// Don't overwhelm user with 62 grids
// Group by size category
// Show map pins for large grids, cluster small ones
```

### 4. Update Assignment Logic
```swift
// OLD: Assign next available grid
// NEW: Prioritize large grids, skip tiny ones
func assignGrids(timeframe: Int) -> [Grid] {
    let primaryGrids = availableGrids.filter { $0.total_miles >= 3.0 }
        .sorted { $0.id < $1.id }  // Closest to center first

    let gridCount = timeframe / 30  // 1 grid per 30 min
    return Array(primaryGrids.prefix(gridCount))
}
```

### 5. Progress Tracking Still Works
No changes to `/api/update-progress` - continue sending GPS updates every 30 seconds

## Request Parameters (Unchanged)

```json
POST /api/create-search
{
    "pet_id": "84",
    "lat": 27.8428,
    "lon": -82.8106,
    "radius_miles": 0.5,      // Unchanged
    "grid_size_miles": 0.5    // Unchanged
}
```

## Error Handling

### New Status Values
- `"pending"` - Grids being generated, poll for updates
- `"active"` - Grids ready, can fetch and assign
- `"failed"` - Generation failed (Overpass timeout, network error)

### Timeout Handling
```swift
// If status="pending" for >120 seconds, show error
if elapsedTime > 120 && status == "pending" {
    showError("Grid generation timed out. Please try again.")
    // Offer retry button
}
```

### Failed Status
```swift
// If status="failed"
if status == "failed" {
    showError("Unable to generate grids for this location.")
    // Offer retry or contact support
}
```

## Authentication (Unchanged)

```
X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67
```

## Performance Expectations

| Metric | Old System | New System |
|--------|------------|------------|
| Response time | 120-300s (or timeout) | <1s (returns immediately) |
| Generation time | N/A | 60-90s (background) |
| Grid count | 6-10 | 50-70 |
| Road coverage | ~77% | 100% |
| Memory usage | 15GB (crashed) | 400MB (stable) |
| Success rate | ~40% (crashes) | ~98% |

## Testing Recommendations

### Test Case 1: Normal Flow
1. Create search for pet_id=84, lat=27.8428, lon=-82.8106, radius=0.5
2. Verify immediate response with status="pending"
3. Poll `/api/search-by-pet?pet_id=84` every 5 seconds
4. Verify status changes to "active" within 90 seconds
5. Fetch grids and verify ~62 grids returned
6. Verify 5-6 grids with ≥6 miles

### Test Case 2: Filter Large Grids
1. Get grids from test case 1
2. Filter `total_miles >= 3.0`
3. Verify 6 grids returned
4. Assign these to volunteer with 3-hour timeframe

### Test Case 3: Timeout Handling
1. Create search in remote area (test with bad lat/lon)
2. If status="pending" for >120s, show timeout error
3. Allow retry

### Test Case 4: Failed Generation
1. Create search with invalid coordinates
2. Verify status="failed"
3. Show appropriate error message

## Migration Checklist for iOS Developer

- [ ] Add polling logic after `/api/create-search` (every 5 seconds)
- [ ] Add loading/progress UI for 60-90 second wait
- [ ] Handle `status: "pending"` | `"active"` | `"failed"`
- [ ] Update grid count expectations (62 instead of 10)
- [ ] Filter grids by `total_miles >= 3.0` for primary assignment
- [ ] Update UI to handle 50-70 grids (don't show all as list)
- [ ] Add timeout handling (>120 seconds = error)
- [ ] Test with pet_id=84, lat=27.8428, lon=-82.8106, radius=0.5
- [ ] Consider map clustering for small grids
- [ ] Update "grids available" messaging (e.g., "6 main coverage areas + 56 small streets")

## Questions for iOS Developer?

If you hit issues or need clarification:
1. **Polling**: Do you want WebSocket instead of polling?
2. **Grid filtering**: Should API pre-filter small grids, or handle in app?
3. **Progress indicator**: Do you need estimated time remaining from API?
4. **Push notifications**: Should server notify when status="active"?

## No Changes Needed For

- ✅ Authentication (X-API-Key header)
- ✅ Progress tracking (POST /api/update-progress)
- ✅ Grid assignment (POST /api/assign-grid)
- ✅ Road data structure (waypoints array unchanged)
- ✅ GPS accuracy requirements
- ✅ Update frequency (still 30 seconds)
