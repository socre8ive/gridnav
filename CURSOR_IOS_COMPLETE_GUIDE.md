# Complete iOS App Development Guide for Cursor/Claude

## 🔑 IMPORTANT: API Key

**Your API Key:** `petsearch_2024_secure_key_f8d92a1b3c4e5f67`

All requests to the EC2 server MUST include this in the `X-API-Key` header.

---

## AWS Security Group Setup

Since the API is now protected by API key, open port 8109 to all traffic:

1. Go to AWS Console → EC2 → Security Groups
2. Find your instance's security group
3. Add Inbound Rule:
   - **Type:** Custom TCP
   - **Port:** 8109
   - **Source:** 0.0.0.0/0 (Anywhere)
   - **Description:** Pet Search API (protected by API key)

---

## Cloudflare Workers Setup

### Worker #1: Grid Requests (`grid-requests` worker)

Deploy this to Cloudflare Workers:

```javascript
export default {
  async fetch(request, env) {
    const EC2_API_BASE = 'http://54.163.97.184:8109';
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

**Deployment:**
1. Go to Cloudflare Dashboard → Workers & Pages
2. Create Worker named `pet-search-grid-requests`
3. Paste code above
4. Deploy
5. Note the URL (e.g., `https://pet-search-grid-requests.YOUR-ACCOUNT.workers.dev`)

---

### Worker #2: Progress & Assignment (`search-progress` worker)

```javascript
export default {
  async fetch(request, env) {
    const EC2_API_BASE = 'http://54.163.97.184:8109';
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

      // Forward to EC2 API with API key
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

**Deployment:**
1. Create Worker named `pet-search-progress`
2. Paste code above
3. Deploy
4. Note the URL (e.g., `https://pet-search-progress.YOUR-ACCOUNT.workers.dev`)

---

## Complete iOS App Prompts for Cursor

Copy these to Cursor/Claude **in order**. Replace `YOUR-WORKER-URLs` with actual URLs from Cloudflare.

---

### **Prompt 1: Project Setup**

```
Create a new iOS Swift app called "PetSearchVolunteer" with:

1. Minimum iOS 16.0
2. SwiftUI interface
3. Request location permissions in Info.plist:
   - NSLocationWhenInUseUsageDescription: "We need your location to track search progress"
   - NSLocationAlwaysAndWhenInUseUsageDescription: "Background tracking helps record searched roads"

4. Create Constants.swift file with:
   struct APIConfig {
       static let gridRequestURL = "https://YOUR-GRID-WORKER.workers.dev"
       static let progressURL = "https://YOUR-PROGRESS-WORKER.workers.dev"
   }

5. Add MapKit and CoreLocation frameworks (built-in)
```

---

### **Prompt 2: Data Models**

```
Create Swift Codable models in Models/ folder:

1. File: CreateSearchRequest.swift
struct CreateSearchRequest: Codable {
    let lat: Double
    let lon: Double
    let radius_miles: Double
    let grid_size_miles: Double = 0.3
    let address: String
    let pet_id: String
}

2. File: CreateSearchResponse.swift
struct CreateSearchResponse: Codable {
    let search_id: String
    let center: Coordinate
    let tiles: [GridTile]
    let total_tiles: Int
    let total_roads: Int
    let grid_size_miles: Double
    let message: String
}

struct GridTile: Codable, Identifiable {
    let id: String
    let grid_id: Int
    let road_count: Int
    let total_distance_miles: Double
    let estimated_minutes: Int
    let bounds: GridBounds
    let center: Coordinate
    let road_details: [RoadDetail]
    let grid_size_miles: Double
}

struct GridBounds: Codable {
    let min_lat: Double
    let max_lat: Double
    let min_lon: Double
    let max_lon: Double
}

struct Coordinate: Codable {
    let lat: Double
    let lon: Double
}

struct RoadDetail: Codable, Identifiable {
    let id: String
    let name: String
    let waypoints: [Coordinate]
    let highway_type: String
    let length_meters: Double
    let has_name: Bool
}

3. File: AssignGridRequest.swift
struct AssignGridRequest: Codable {
    let search_id: String
    let pet_id: String
    let searcher_id: String
    let searcher_name: String
    let timeframe_minutes: Int
}

4. File: AssignGridResponse.swift
struct AssignGridResponse: Codable {
    let success: Bool
    let assignments: [GridAssignment]
    let total_assigned: Int
    let timeframe_minutes: Int
}

struct GridAssignment: Codable, Identifiable {
    let assignment_id: String
    let grid_id: Int
    let expires_at: Int
    let grace_expires_at: Int

    var id: String { assignment_id }
}

5. File: UpdateProgressRequest.swift
struct UpdateProgressRequest: Codable {
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

6. File: GridStatusResponse.swift
struct GridStatusResponse: Codable {
    let success: Bool
    let search_id: String
    let grids: [GridStatus]
    let total_grids: Int
}

struct GridStatus: Codable, Identifiable {
    let grid_id: Int
    let searcher_id: String?
    let searcher_name: String?
    let assigned_at: Int?
    let expires_at: Int?
    let grace_expires_at: Int?
    let status: String
    let completion_percentage: Double
    let completed_at: Int?
    let timeframe_minutes: Int?

    var id: Int { grid_id }
}
```

---

### **Prompt 3: Network Service**

```
Create NetworkService.swift that handles all API calls:

import Foundation

enum NetworkError: Error {
    case invalidURL
    case invalidResponse
    case serverError(String)
    case decodingError
}

class NetworkService {
    static let shared = NetworkService()

    private init() {}

    // MARK: - Create Search
    func createSearch(request: CreateSearchRequest) async throws -> CreateSearchResponse {
        let url = URL(string: "\(APIConfig.gridRequestURL)")!
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(request)
        urlRequest.timeoutInterval = 60

        let (data, response) = try await URLSession.shared.data(for: urlRequest)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw NetworkError.serverError("Failed to create search")
        }

        return try JSONDecoder().decode(CreateSearchResponse.self, from: data)
    }

    // MARK: - Assign Grid
    func assignGrid(request: AssignGridRequest) async throws -> AssignGridResponse {
        let url = URL(string: "\(APIConfig.progressURL)?action=assign")!
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(request)
        urlRequest.timeoutInterval = 30

        let (data, response) = try await URLSession.shared.data(for: urlRequest)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw NetworkError.serverError("Failed to assign grid")
        }

        return try JSONDecoder().decode(AssignGridResponse.self, from: data)
    }

    // MARK: - Update Progress
    func updateProgress(request: UpdateProgressRequest) async throws {
        let url = URL(string: "\(APIConfig.progressURL)?action=update_progress")!
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "POST"
        urlRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        urlRequest.httpBody = try JSONEncoder().encode(request)
        urlRequest.timeoutInterval = 30

        let (_, response) = try await URLSession.shared.data(for: urlRequest)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw NetworkError.serverError("Failed to update progress")
        }
    }

    // MARK: - Get Grid Status
    func getGridStatus(searchId: String) async throws -> GridStatusResponse {
        let url = URL(string: "\(APIConfig.progressURL)?search_id=\(searchId)")!
        var urlRequest = URLRequest(url: url)
        urlRequest.httpMethod = "GET"
        urlRequest.timeoutInterval = 30

        let (data, response) = try await URLSession.shared.data(for: urlRequest)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            throw NetworkError.serverError("Failed to get grid status")
        }

        return try JSONDecoder().decode(GridStatusResponse.self, from: data)
    }
}
```

---

### **Prompt 4: Report Lost Pet Screen**

```
Create ReportLostPetView.swift:

A SwiftUI form for reporting a lost pet with:
1. Text fields: pet name, description, last seen address, owner name, owner phone
2. DatePicker for last seen date/time
3. Use MKLocalSearch to geocode address to lat/lon when user types
4. Generate petId from: "\(UIDevice.current.identifierForVendor?.uuidString ?? "unknown")_\(Int(Date().timeIntervalSince1970))"
5. Submit button that calls NetworkService.shared.createSearch()
6. Show loading spinner during API call
7. On success, navigate to SearchCoordinatorView passing search data
8. On error, show alert with error message
9. Store search_id and pet_id in @AppStorage for persistence
```

---

### **Prompt 5: Search Coordinator Screen**

```
Create SearchCoordinatorView.swift:

Display a map interface for coordinating volunteers:

1. MapKit map showing:
   - Center marker at last seen location
   - Grid polygons color-coded:
     * Gray (0.3 alpha) = unsearched
     * Yellow (0.5 alpha) = in progress
     * Green (0.5 alpha) = completed
   - Grid labels (grid1, grid2, etc.)

2. Floating "Volunteer" button that presents sheet with:
   - TextField for name
   - Picker for timeframe (30, 60, 90, 120 minutes)
   - "Get Assignment" button

3. When volunteer submits:
   - Generate searcherId: UUID().uuidString
   - Call NetworkService.shared.assignGrid()
   - Navigate to ActiveSearchView with assignments

4. Poll grid status every 30 seconds:
   - Call NetworkService.shared.getGridStatus()
   - Update map overlay colors

5. Display list of active volunteers with:
   - Name, grid numbers, time remaining, completion %

6. Use @StateObject SearchViewModel for state management
```

---

### **Prompt 6: Active Search Screen**

```
Create ActiveSearchView.swift:

Volunteer's active search interface:

1. Request location permissions with CLLocationManager
2. Track GPS location every 30 seconds
3. Display MapKit map with:
   - Assigned grid polygon(s) highlighted blue
   - Road polylines (gray = not searched, green = completed)
   - Blue dot for current location

4. Show:
   - Countdown timer (time remaining)
   - Progress bar (completion %)
   - Scrollable road list with checkboxes

5. Every 30 seconds:
   - Call NetworkService.shared.updateProgress() with:
     * Current GPS location
     * List of completed roads

6. Auto-detect nearby roads:
   - Calculate distance from GPS to each road polyline
   - If within 20 meters, auto-check the road
   - Change polyline color to green

7. Manual road completion:
   - Tap road in list to check it off
   - Include in next progress update

8. When completion reaches 85%:
   - Show "Grid Complete!" notification
   - Return to SearchCoordinatorView

9. "Finish Early" button:
   - Saves current progress
   - Returns to coordinator

10. Use @StateObject for location tracking and progress state
```

---

### **Prompt 7: ViewModels**

```
Create SearchViewModel.swift and VolunteerViewModel.swift:

SearchViewModel (ObservableObject):
- @Published var searchId: String?
- @Published var petId: String?
- @Published var grids: [GridTile] = []
- @Published var gridStatuses: [GridStatus] = []
- @Published var isLoading = false
- @Published var errorMessage: String?

Methods:
- func createSearch(lat:lon:address:petId:) async
- func pollGridStatus() async // Every 30s
- func stopPolling()

VolunteerViewModel (ObservableObject):
- @Published var assignments: [GridAssignment] = []
- @Published var searcherId = UUID().uuidString
- @Published var searcherName = ""
- @Published var timeframe = 60

Methods:
- func requestAssignment(searchId:petId:) async
- func getAssignedGrids(from:[GridTile]) -> [GridTile]

Use @MainActor to ensure UI updates on main thread.
```

---

### **Prompt 8: Location Tracking**

```
Create LocationManager.swift:

ObservableObject that wraps CLLocationManager:

- Request when-in-use authorization
- Update location every 30 seconds (not continuous)
- @Published var location: CLLocation?
- @Published var accuracy: Double?
- Method: calculateDistanceToRoad(waypoints:[Coordinate]) -> Double
- Method: isNearRoad(road:RoadDetail) -> Bool (within 20m)
- Implement CLLocationManagerDelegate

Include geometry helper to calculate distance from point to line segment.
```

---

### **Prompt 9: Main App Structure**

```
Update App file to:

1. Create @StateObject var searchViewModel = SearchViewModel()
2. NavigationStack with ContentView as root
3. Pass searchViewModel as .environmentObject() to all views
4. Start with ReportLostPetView
5. Handle navigation to SearchCoordinatorView → ActiveSearchView

Use programmatic navigation with NavigationPath for proper state management.
```

---

## Testing Your Setup

### Test Worker #1 (Grid Requests):
```bash
curl -X POST https://YOUR-WORKER-1.workers.dev \
  -H "Content-Type: application/json" \
  -d '{
    "lat": 27.8428,
    "lon": -82.8106,
    "radius_miles": 1.5,
    "address": "Seminole, FL",
    "pet_id": "test_123_20251021"
  }'
```

### Test Worker #2 (Grid Status):
```bash
curl "https://YOUR-WORKER-2.workers.dev?search_id=SEARCH-ID-HERE"
```

---

## Summary

✅ EC2 server protected with API key
✅ Cloudflare Workers forward requests with API key
✅ iOS app only talks to Cloudflare Workers (never directly to EC2)
✅ No IP whitelisting needed - API key provides security
✅ Open port 8109 to all traffic in AWS (safe because of API key)

**API Key:** `petsearch_2024_secure_key_f8d92a1b3c4e5f67`
**EC2 Server:** http://54.163.97.184:8109
**Protection:** All API endpoints require X-API-Key header

