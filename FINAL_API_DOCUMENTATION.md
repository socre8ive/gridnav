# Pet Search API - Production Setup Complete

## 🌐 Production Domain
**API Base URL:** `https://api.psar.app`

**SSL Certificate:** Let's Encrypt (Auto-renews)
**Valid Until:** January 19, 2026
**API Key:** `petsearch_2024_secure_key_f8d92a1b3c4e5f67`

---

## Cloudflare Workers - Final Production Version

### Worker #1: Grid Requests

```javascript
export default {
  async fetch(request, env) {
    const EC2_API_BASE = 'https://api.psar.app';
    const API_KEY = 'petsearch_2024_secure_key_f8d92a1b3c4e5f67';

    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, X-API-Key',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    try {
      const body = await request.json();

      const response = await fetch(`${EC2_API_BASE}/api/create-search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY,
        },
        body: JSON.stringify({
          lat: body.lat || 0,
          lon: body.lon || 0,
          radius_miles: body.radius_miles || 1.5,
          grid_size_miles: 0.3,
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

---

### Worker #2: Progress & Assignment

```javascript
export default {
  async fetch(request, env) {
    const EC2_API_BASE = 'https://api.psar.app';
    const API_KEY = 'petsearch_2024_secure_key_f8d92a1b3c4e5f67';

    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, X-API-Key',
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    try {
      const url = new URL(request.url);
      const action = url.searchParams.get('action');

      // Handle GET requests (grid-status)
      if (request.method === 'GET') {
        const searchId = url.searchParams.get('search_id');
        if (!searchId) {
          throw new Error('search_id parameter required');
        }

        const response = await fetch(
          `${EC2_API_BASE}/api/grid-status?search_id=${searchId}`,
          {
            method: 'GET',
            headers: {
              'X-API-Key': API_KEY,
            },
          }
        );

        const data = await response.json();
        return new Response(JSON.stringify(data), {
          headers: {
            ...corsHeaders,
            'Content-Type': 'application/json',
          },
        });
      }

      // Handle POST requests
      const body = await request.json();

      let endpoint;
      if (action === 'assign') {
        endpoint = '/api/assign-grid';
      } else if (action === 'update_progress') {
        endpoint = '/api/update-progress';
      } else {
        throw new Error('Invalid action');
      }

      const response = await fetch(`${EC2_API_BASE}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-API-Key': API_KEY,
        },
        body: JSON.stringify(body)
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

## API Endpoints

All endpoints now use `https://api.psar.app`:

### Create Search
```
POST https://api.psar.app/api/create-search
Header: X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67
Body: {
  "lat": 27.8428,
  "lon": -82.8106,
  "radius_miles": 1.5,
  "grid_size_miles": 0.3,
  "address": "Seminole, FL",
  "pet_id": "apple_user_123_20251021"
}
```

### Assign Grid
```
POST https://api.psar.app/api/assign-grid
Header: X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67
Body: {
  "search_id": "search-abc123",
  "pet_id": "apple_user_123_20251021",
  "searcher_id": "volunteer_001",
  "searcher_name": "John Doe",
  "timeframe_minutes": 60
}
```

### Update Progress
```
POST https://api.psar.app/api/update-progress
Header: X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67
Body: {
  "assignment_id": "assign-def456",
  "search_id": "search-abc123",
  "pet_id": "apple_user_123_20251021",
  "grid_id": 1,
  "searcher_id": "volunteer_001",
  "lat": 27.8430,
  "lon": -82.8100,
  "accuracy_meters": 10.0,
  "roads_covered": []
}
```

### Get Grid Status
```
GET https://api.psar.app/api/grid-status?search_id=search-abc123
Header: X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67
```

---

## iOS App Update (Tell Cursor)

"Update the API configuration to use the production domain:

In `Constants.swift`, change:
```swift
struct APIConfig {
    // OLD: static let apiBaseURL = "https://54.163.97.184:8443"
    static let apiBaseURL = "https://api.psar.app"
}
```

Remove the NSAppTransportSecurity exception from Info.plist - it's no longer needed since we have a valid Let's Encrypt certificate.

The app will now use the production domain with proper SSL."

---

## Security Features

✅ **Valid SSL Certificate** - Let's Encrypt (trusted by all devices)
✅ **Auto-renewal** - Certificate renews automatically before expiration
✅ **API Key Authentication** - All endpoints protected
✅ **HTTPS Redirect** - HTTP automatically redirects to HTTPS
✅ **Professional Domain** - api.psar.app instead of IP address

---

## Architecture

```
iOS App
  ↓
Cloudflare Worker #1 (Grid Requests)
  ↓
https://api.psar.app → Nginx (Port 443) → FastAPI (Port 8443)
  ↓
Cloudflare D1 Database

iOS App
  ↓
Cloudflare Worker #2 (Progress Updates)
  ↓
https://api.psar.app → Nginx (Port 443) → FastAPI (Port 8443)
  ↓
Cloudflare D1 Database
```

---

## Test Commands

```bash
# Test with curl (no -k flag needed - real certificate!)
curl https://api.psar.app/api/database-stats

# Test create-search
curl -X POST https://api.psar.app/api/create-search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67" \
  -d '{"lat": 27.8428, "lon": -82.8106, "radius_miles": 0.5, "grid_size_miles": 0.3, "pet_id": "test123"}'
```

---

## Certificate Details

- **Domain:** api.psar.app
- **Issuer:** Let's Encrypt
- **Valid Until:** January 19, 2026
- **Auto-Renewal:** Enabled (checks twice daily)
- **Certificate Path:** `/etc/letsencrypt/live/api.psar.app/`

---

## What's Changed from IP-based Setup

| Old | New |
|-----|-----|
| `http://54.163.97.184:8109` | `https://api.psar.app` |
| Self-signed certificate | Let's Encrypt certificate |
| Port 8443 direct | Port 443 via Nginx |
| IP address | Professional domain |
| Manual renewal | Auto-renewal |

---

🎉 **Production Ready!**

Your Pet Search API is now running on a professional domain with enterprise-grade SSL!
