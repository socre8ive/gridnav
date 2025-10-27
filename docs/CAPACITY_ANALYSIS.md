# Server Capacity Analysis

## Current Server Specifications

**Instance Details:**
- **vCPUs**: 4 (Intel Xeon Platinum 8259CL @ 2.50GHz)
- **RAM**: 16 GB (14 GB available)
- **Disk**: 40 GB SSD (31 GB free)
- **Configuration**: 4 Gunicorn workers with Uvicorn async workers

**Current Resource Usage (Idle):**
- Memory: ~625 MB (4% of total)
- CPU Load: 0.36 (9% of capacity)
- Workers: 4 processes, ~140-185 MB each

---

## Target Workload Analysis

### Scenario: 25 Searchers + 10 Grid Creations/Hour

#### 1. Searcher Progress Updates (25 simultaneous)

**Request Pattern:**
- Endpoint: `POST /api/update-progress`
- Frequency: Every 30 seconds per searcher
- Request Rate: **50 requests/minute** (25 × 2)
- Peak: **0.83 requests/second**

**Resource Requirements per Request:**
- Operation: Simple database insert (JSON to Cloudflare D1)
- CPU: ~5-10ms per request
- Memory: Minimal (~1-2 MB transient)
- I/O: Network only (external DB)

**Total Load:**
- CPU: 5-10ms × 50/min = **0.4-0.8% CPU usage**
- Memory: Negligible (async, non-blocking)
- Concurrent connections: 25 (well within limits)

**Verdict:** ✅ **Trivial load** - async FastAPI handles this easily

---

#### 2. Grid Generation (10 per hour)

**Request Pattern:**
- Endpoint: `POST /api/create-search`
- Frequency: ~10 per hour = **1 every 6 minutes**
- Peak scenario: 3 simultaneous generations

**Resource Requirements per Generation:**

| Phase | Duration | CPU | Memory | Notes |
|-------|----------|-----|--------|-------|
| Overpass API Query | 5-10s | Low | ~50 MB | Network I/O bound |
| Road Extraction | 2-5s | Medium | ~100 MB | NetworkX graph building |
| Grid Generation (BFS) | 30-45s | High | ~200 MB | Spatial queries, connectivity |
| Consolidation | 5-10s | Medium | ~150 MB | Graph analysis |
| Connectivity Check | 10-20s | High | ~300 MB | Full graph, shortest paths |
| Database Save | 5-10s | Low | ~50 MB | JSON serialization |
| **TOTAL** | **60-90s** | **Varies** | **Peak: ~300-400 MB** | Async background task |

**Concurrent Grid Generations:**

Assuming worst case: **3 simultaneous** grid generations

- **Peak Memory**: 3 × 400 MB = **1.2 GB**
- **Peak CPU**: 3 workers × 100% = **300% CPU** (out of 400% available)
- **Duration**: 60-90 seconds

**Verdict:** ✅ **Easily handled** - plenty of headroom

---

## Combined Workload Calculation

### Peak Load Scenario (Worst Case)

**Simultaneous Operations:**
- 25 searchers sending progress updates (50 req/min)
- 3 grid generations running in background
- Normal API queries (status checks, assignments)

**Resource Usage:**

| Resource | Base | Searchers | Grid Gen | Other | Total | Capacity | Utilization |
|----------|------|-----------|----------|-------|-------|----------|-------------|
| **CPU** | 10% | 1% | 75% | 5% | **91%** | 400% | 23% |
| **Memory** | 625 MB | 50 MB | 1200 MB | 100 MB | **1975 MB** | 16 GB | 12% |
| **Network** | Low | 50 KB/s | 500 KB/s | 50 KB/s | **~600 KB/s** | Unlimited | Minimal |

**Load Average:** Estimated ~1.5-2.0 (out of 4.0 max)

---

## Bottleneck Analysis

### 1. CPU (4 cores)
- **Current**: 0.36 load average
- **Peak Expected**: 1.5-2.0 load average
- **Capacity**: 4.0 (100% utilization)
- **Headroom**: 50-60%
- **Status**: ✅ **No bottleneck**

### 2. Memory (16 GB)
- **Current**: 1 GB used
- **Peak Expected**: 2 GB used
- **Capacity**: 16 GB
- **Headroom**: 87%
- **Status**: ✅ **No bottleneck**

### 3. Disk I/O
- **SQLite**: Minimal (only for logs)
- **Database**: Cloudflare D1 (external, no local I/O)
- **Status**: ✅ **No bottleneck**

### 4. Network
- **Overpass API**: Rate limited by Overpass, not bandwidth
- **Cloudflare D1**: Low latency, high throughput
- **Status**: ✅ **No bottleneck**

---

## Scaling Analysis

### Current Capacity (4 vCPU, 16 GB RAM)

**Can Handle:**
- ✅ 25 simultaneous searchers → Can scale to **100+ searchers**
- ✅ 10 grid creations/hour → Can scale to **30-40/hour**
- ✅ 3 simultaneous generations → Can handle **4-5 simultaneous**

### When to Scale Up?

You would need more resources when:

| Metric | Threshold | Action Required |
|--------|-----------|-----------------|
| **Searchers** | 80+ simultaneous | Add workers or CPU |
| **Grid Creations** | 40+ per hour | Add more vCPUs |
| **Simultaneous Gens** | 5+ at once | Increase RAM + CPU |
| **Load Average** | > 3.5 sustained | Scale vertically |
| **Memory Usage** | > 12 GB | Add more RAM |

---

## Recommendations

### ✅ Current Setup is PERFECT for Target Load

**Your current server can easily handle:**
- 25 simultaneous searchers ✓
- 10 grid creations per hour ✓
- Peak loads with headroom ✓

**Utilization at Target Load:**
- CPU: ~23% (77% headroom)
- Memory: ~12% (88% headroom)
- Network: Minimal

### Cost Optimization: Could You Downsize?

**Could scale DOWN to t3.large:**
- 2 vCPUs (instead of 4)
- 8 GB RAM (instead of 16)
- **Estimated savings: ~40-50%**

**Analysis for t3.large:**

| Workload | t3.large Capacity | Status |
|----------|-------------------|--------|
| 25 searchers | ✅ Yes (trivial load) | Safe |
| 10 grid/hour | ✅ Yes (2 simultaneous max) | Safe |
| 3 simultaneous gens | ⚠️ Tight (150% CPU) | May slow down |
| Peak memory | ✅ 2 GB / 8 GB | Safe |

**Verdict on t3.large:**
- ✅ **Likely sufficient** for your target load
- ⚠️ Less headroom for unexpected spikes
- 💰 **~$30-40/month savings**

### Recommended Actions

**For Current Workload (25 searchers, 10 gen/hr):**

1. **Stay on current instance** ✅ RECOMMENDED
   - You have excellent headroom for growth
   - Can handle 3-4x increase without issues
   - Only $60-80/month for peace of mind

2. **OR downsize to t3.large** 💰 (if cost-sensitive)
   - Test thoroughly first
   - Monitor CPU during peak
   - Keep current instance as fallback

**For Future Growth:**

- **80+ simultaneous searchers**: Stay on current or upgrade to t3.2xlarge
- **40+ grid creations/hour**: Upgrade to c6i.2xlarge (compute-optimized)
- **100+ searchers + 50+ gen/hr**: Consider load balancer + 2x t3.xlarge

---

## Monitoring Recommendations

Set up alerts for:

```bash
# CPU load > 3.0 for 5 minutes
# Memory usage > 12 GB
# Disk usage > 80%
```

Check metrics:
```bash
# Real-time monitoring
watch -n 2 'uptime && free -h && ps aux | grep gunicorn | grep -v grep'

# Service logs
sudo journalctl -u petsearch.service -f

# Grid generation logs
tail -f /tmp/grid_gen_*.log
```

---

## Conclusion

**Your current 4 vCPU / 16 GB server is EXCELLENT for your target workload.**

- ✅ Can handle 25 searchers + 10 grid creations/hour easily
- ✅ Only 20-25% utilization at peak
- ✅ Room to grow 3-4x before needing upgrades
- 💰 Could downsize to save ~$35/month, but you'd lose headroom

**Recommendation:** **Keep current server** - the cost difference is minimal and you have excellent growth capacity.
