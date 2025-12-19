# Diagnosing Grid Generation Issues

## iOS App Grid Flow (Reference)

Understanding how iOS retrieves grids is critical for diagnosis:

```
1. /api/create-search → returns search_id, starts background grid generation
2. /api/assign-grid → returns assignments array with grid IDs (this is the grid count source)
3. /api/get-grid (singular) → fetches each grid's detailed data by grid_id
```

**iOS does NOT use:**
- `/api/get-grids` (plural) - incomplete/unused endpoint
- `grids_created` field - informational logging only

## Quick Diagnosis Steps

When a search_id is created but grids aren't available:

### 1. Check if the log file exists

```bash
ls -la /tmp/grid_gen_search-*.log
```

Log files are created at `/tmp/grid_gen_{search_id}.log` during background grid processing.

### 2. Read the log file

```bash
cat /tmp/grid_gen_search-{search_id}.log
```

**Look for:**
- `[OVERPASS] SUCCESS!` - confirms OSM data was downloaded
- `Grid generation COMPLETE - X grids created` - confirms tiles were built
- Any `httpx.ReadTimeout` or `Traceback` errors

### 3. Test if grids were actually saved

```bash
# Test retrieving a specific grid (this is what iOS uses)
curl -sk "https://localhost:8443/api/get-grid?search_id=search-xxx&grid_id=1" \
  -H "X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67"
```

If this returns `"Grid 1 not found"` but the log shows grids were "saved", the database writes failed silently due to timeout.

### 4. Check search status

```bash
curl -sk "https://localhost:8443/api/search-stats?search_id=search-xxx" \
  -H "X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67"
```

## Common Issue: Database Timeout During Grid Saves

### Symptom
- Log shows `Grid generation COMPLETE - 55 grids created, status=active`
- Log shows `Saved 55 grid tiles for retrieval`
- But `/api/get-grid?grid_id=1` returns "Grid 1 not found"
- `httpx.ReadTimeout` errors appear in the log

### Root Cause
**Transient Turso database connectivity timeout.** The grid generation completed successfully, but some or all database writes to `search_grids` table failed due to network/timeout issues with the remote Turso database.

The log may show success messages because:
1. The save loop completed without throwing an exception to the caller
2. But individual INSERT operations timed out silently
3. Or a concurrent operation (like `save_pet_details`) caused connection pool exhaustion

### This is NOT a code bug
The `/api/get-grids` (plural) endpoint returning empty is unrelated - iOS doesn't use that endpoint. The issue is transient database connectivity.

### Resolution
Retry the search:
```bash
curl -sk -X POST "https://localhost:8443/api/retry-search" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67" \
  -d '{"search_id": "search-xxx"}'
```

Or create a new search from the iOS app.

## Log File Analysis

### Success Pattern
```
[search-xxx] Starting grid generation...
[OVERPASS] SUCCESS! Got X ways and Y nodes
[OVERPASS] Created NetworkX graph with N nodes, M edges
Building grids from X roads...
[search-xxx] Updated search tracking: 55 grids, status=active
[search-xxx] Saved 55 grid tiles for retrieval
[search-xxx] Grid generation COMPLETE - 55 grids created, status=active
```

**Verify with:** `/api/get-grid?search_id=xxx&grid_id=1` should return grid data.

### Failure Pattern - OSM Download Failed
```
[OVERPASS] Trying server: https://overpass-api.de/api/interpreter
[OVERPASS] Server https://overpass-api.de/api/interpreter failed: ...
[FALLBACK] All Overpass servers failed, falling back to OSMnx...
[OSMnx] FAILED: ...
```

### Failure Pattern - Database Timeout (Grids Lost)
```
httpx.ReadTimeout
...
[search-xxx] Saved 55 grid tiles for retrieval  ← This may be a lie!
```

The "Saved" message prints after the loop, but individual saves may have failed. Always verify with `/api/get-grid`.

## Service Management

```bash
# Check service status
sudo systemctl status petsearch.service

# View recent logs
sudo journalctl -u petsearch.service --since "10 minutes ago" --no-pager

# Restart service (if needed)
sudo systemctl restart petsearch.service
```

## API Endpoint Reference

| Endpoint | Method | Used By | Purpose |
|----------|--------|---------|---------|
| `/api/create-search` | POST | iOS | Create search, start grid generation |
| `/api/assign-grid` | POST | iOS | Assign grids to searcher, returns grid IDs |
| `/api/get-grid` | GET | iOS | Fetch single grid's road data |
| `/api/get-grids` | POST | **Unused** | Incomplete, do not rely on |
| `/api/retry-search` | POST | Admin | Retry failed grid generation |
