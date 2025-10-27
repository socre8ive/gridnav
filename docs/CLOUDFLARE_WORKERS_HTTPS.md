# Cloudflare Workers - HTTPS Version

**API Base URL:** `https://54.163.97.184:8443`
**API Key:** `petsearch_2024_secure_key_f8d92a1b3c4e5f67`

---

## Worker #1: Grid Requests

```javascript
export default {
  async fetch(request, env) {
    const EC2_API_BASE = 'https://54.163.97.184:8443';
    const API_KEY = 'petsearch_2024_secure_key_f8d92a1b3c4e5f67';

    // CORS headers
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

      // Forward request to EC2 API with API key
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

## Worker #2: Progress & Assignment

```javascript
export default {
  async fetch(request, env) {
    const EC2_API_BASE = 'https://54.163.97.184:8443';
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

      // Handle POST requests (assign or update_progress)
      const body = await request.json();

      let endpoint;
      if (action === 'assign') {
        endpoint = '/api/assign-grid';
      } else if (action === 'update_progress') {
        endpoint = '/api/update-progress';
      } else {
        throw new Error('Invalid action. Must be "assign" or "update_progress"');
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

## Quick Test Commands

Test directly from your machine (after opening port 8443):

```bash
# Test create-search (will fail without valid data, but proves HTTPS works)
curl -k -X POST https://54.163.97.184:8443/api/create-search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: petsearch_2024_secure_key_f8d92a1b3c4e5f67" \
  -d '{"lat": 27.8428, "lon": -82.8106, "radius_miles": 0.5, "grid_size_miles": 0.3, "pet_id": "test123"}'

# Test database stats (should work)
curl -k https://54.163.97.184:8443/api/database-stats
```

**Note:** Use `-k` flag with curl to accept self-signed certificate.

---

## What Changed

✅ **HTTP → HTTPS** (port 8109 → 8443)
✅ **Self-signed SSL certificate** installed
✅ **API key authentication** still required
✅ **All endpoints** now encrypted

Your data is now encrypted in transit! 🔒
