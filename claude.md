# Claude Code Session Notes

## Grid Verification Worker (2025-12-19)

### Problem
Pet searches were being created but grid generation would sometimes fail silently, leaving searches in broken states:
- `status='active'` but `total_grids=0` (no grids generated)
- `status='processing'` stuck indefinitely
- Grid count mismatches where `total_grids` didn't match actual grids in database (e.g., reported 72, actual 66)

Example: Pet 144 had `status='active'` with `total_grids=0` - appeared healthy but had no grids.

### Solution
Created a background worker that runs every 30 minutes to verify and fix grid issues.

### Files Created/Modified

#### `/opt/petsearch/grid_verification_worker.py`
Background worker script that:
1. **Detects missing grids** - Finds searches with `total_grids=0`
2. **Detects bad status** - Finds `failed`, `pending`, or stuck `processing` (>30 min old)
3. **Detects grid count mismatches** - Compares `total_grids` against actual count in `search_grids` table
4. **Fixes issues automatically**:
   - Updates grid counts to match reality (handles consolidation where 50 grids become 30 larger ones)
   - Triggers regeneration via `/api/retry-search` for missing grids
   - Sets broken `active`/`processing` status to `failed` before retry

#### `/etc/systemd/system/grid-verification.service`
Systemd service configuration:
```ini
[Unit]
Description=Pet Search Grid Verification Worker
After=network.target petsearch.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/petsearch
Environment="PATH=/opt/petsearch/venv/bin"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/opt/petsearch/venv/bin/python3 -u /opt/petsearch/grid_verification_worker.py
Restart=always
RestartSec=60

[Install]
WantedBy=multi-user.target
```

### Service Management

```bash
# Check status
sudo systemctl status grid-verification.service

# View logs
sudo journalctl -u grid-verification.service -f

# Restart
sudo systemctl restart grid-verification.service

# Stop
sudo systemctl stop grid-verification.service
```

### Log Output Example

```
[WORKER] === Starting verification run at 2025-12-19 23:47:02 ===
[WORKER] Found 3 searches needing attention
[WORKER] Checking pet_id=test-opt-001: status=failed, total_grids=0
[WORKER] Triggered regeneration for pet_id=test-opt-001
[WORKER] Checking for grid count mismatches...
[WORKER] Mismatch: pet_id=146, reported=72, actual=66
[WORKER] Fixing grid count mismatch for pet_id=146: 72 -> 66
[WORKER] === Verification complete ===
[WORKER] Checked: 3, Fixed: 2, Mismatches fixed: 10, Regenerating: 1, Failed: 0
[WORKER] Next check in 30 minutes...
```

### Configuration

- **Check interval**: 30 minutes (configurable via `CHECK_INTERVAL_SECONDS`)
- **Stuck processing threshold**: 30 minutes
- **API endpoint**: Uses `/api/retry-search` to trigger regeneration

### Why Grid Count Mismatches Happen

During grid generation, the system:
1. Creates initial grids (e.g., 224 small grids)
2. Consolidates tiny grids into larger ones (e.g., reduces to 36 grids)
3. Updates `total_grids` with final count

If the process crashes or times out between steps, the count can be wrong. The worker detects and corrects these mismatches.
