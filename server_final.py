#!/usr/bin/env python3
"""
FINAL SOLUTION - Uses OSMNX but aggressively filters fake connections
Identifies and removes only the artificial diagonal edges
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import osmnx as ox
import json
from typing import List, Dict, Optional
from pydantic import BaseModel
import uuid
import math
from database import db

# Configure OSMNX
ox.settings.use_cache = True
ox.settings.log_console = True

app = FastAPI(title="Pet Search Grid API - Final Solution")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    lat: float
    lon: float
    radius_miles: float = 0.25
    address: Optional[str] = None

def is_definitely_fake(edge, coords, edge_data):
    """
    Identify fake diagonal connections with high accuracy
    These are edges added by OSMNX for graph connectivity
    """
    # Get properties
    street_name = edge_data.get('name', None)
    if isinstance(street_name, list):
        street_name = street_name[0] if street_name else None

    highway_type = edge_data.get('highway', 'unknown')
    if isinstance(highway_type, list):
        highway_type = highway_type[0] if highway_type else 'unknown'

    length = edge_data.get('length', 0)

    # Real roads have names OR are short service roads
    if street_name:
        return False  # Named roads are always real

    # Check if it's a straight line (only 2 points)
    if len(coords) != 2:
        return False  # Multi-point segments are usually real

    # Calculate angle
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    angle_mod = abs(angle % 90)

    # Check if diagonal AND unnamed AND long
    if angle_mod > 30 and angle_mod < 60:  # Diagonal range
        if length > 50:  # Longer than 50 meters
            # This is very likely a fake connection
            return True

    # Also filter very long unnamed segments
    if not street_name and length > 150:
        return True

    return False

@app.post("/api/create-search")
async def create_search(request: SearchRequest):
    """Create search grid with aggressive filtering of fake connections"""
    try:
        search_id = f"search-{uuid.uuid4()}"

        # If address provided, geocode server-side
        if request.address and request.lat == 0 and request.lon == 0:
            print(f"Geocoding address: {request.address}")
            import requests
            geo_url = f"https://nominatim.openstreetmap.org/search?format=json&q={request.address}&limit=1"
            geo_resp = requests.get(geo_url, headers={'User-Agent': 'PetSearchGrid/1.0'})
            geo_data = geo_resp.json()

            if geo_data and len(geo_data) > 0:
                request.lat = float(geo_data[0]['lat'])
                request.lon = float(geo_data[0]['lon'])
                print(f"Geocoded to: {request.lat}, {request.lon}")
            else:
                raise HTTPException(status_code=400, detail=f"Could not geocode address: {request.address}")

        print(f"Downloading OSM data for {request.lat}, {request.lon}")

        # Use OSMNX to get comprehensive road network
        G = ox.graph_from_point(
            (request.lat, request.lon),
            dist=request.radius_miles * 1609.34,
            network_type='drive_service',  # Include service roads
            simplify=False  # Don't simplify
        )

        # Get edges
        edges = ox.graph_to_gdfs(G, nodes=False, edges=True)

        print(f"Downloaded {len(edges)} edges from OSMNX")

        tiles = []
        filtered_count = 0
        kept_count = 0

        # Process each edge
        for idx, edge in edges.iterrows():
            # Extract coordinates
            coords = list(edge.geometry.coords)

            # Check if this is a fake diagonal
            if is_definitely_fake(edge, coords, edge):
                filtered_count += 1
                continue

            kept_count += 1

            # Get street name
            street_name = edge.get('name', None)
            if isinstance(street_name, list):
                street_name = street_name[0] if street_name else None

            if not street_name:
                street_name = f"Road_{kept_count}"

            # Create waypoints (filter out NaN values)
            waypoints = []
            for lon, lat in coords:
                if not (math.isnan(lat) or math.isnan(lon) or math.isinf(lat) or math.isinf(lon)):
                    waypoints.append({'lat': lat, 'lon': lon})

            # Skip edges with no waypoints
            if len(waypoints) < 2:
                continue

            # Get edge properties
            highway_type = edge.get('highway', 'unknown')
            if isinstance(highway_type, list):
                highway_type = highway_type[0] if highway_type else 'unknown'

            length = edge.get('length', 0)
            # Handle NaN or infinity in length
            if not isinstance(length, (int, float)) or math.isnan(length) or math.isinf(length):
                length = 0

            tiles.append({
                'id': f"{search_id}-{idx}",
                'roads': [street_name],
                'waypoints': waypoints,
                'highway_type': highway_type,
                'length_meters': round(length, 1),
                'point_count': len(waypoints),
                'has_name': bool(edge.get('name')),
                'total_distance_miles': round(length / 1609.34, 3),
                'estimated_minutes': max(1, int(length / 50)) if length > 0 else 1
            })

        print(f"Kept {kept_count} real roads, filtered {filtered_count} fake diagonals")

        # Sort by whether they have names (named roads first)
        tiles.sort(key=lambda x: (not x['has_name'], x['length_meters']))

        # Clean any remaining NaN/Infinity values
        def clean_nan(obj):
            if isinstance(obj, dict):
                return {k: clean_nan(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan(item) for item in obj]
            elif isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            return obj

        result = {
            'search_id': search_id,
            'center': {'lat': request.lat, 'lon': request.lon},
            'tiles': tiles,
            'total_tiles': len(tiles),
            'filtered_count': filtered_count,
            'message': f'Showing {len(tiles)} real roads (filtered {filtered_count} fake connections)'
        }

        # Clean the result to remove any NaN/Infinity values
        cleaned_result = clean_nan(result)

        # Save to database in background (non-blocking)
        import asyncio

        async def save_to_database_background():
            """Save to database without blocking the response"""
            try:
                search_data = {
                    'search_id': search_id,
                    'center': {'lat': request.lat, 'lon': request.lon},
                    'radius_miles': request.radius_miles,
                    'address': request.address or '',
                    'total_tiles': len(tiles),
                    'filtered_count': filtered_count
                }
                db_stats = await db.save_search_results(search_data, tiles)
                print(f"Database: Saved {db_stats['new_roads']} new roads, {db_stats['existing_roads']} existing roads")
            except Exception as db_error:
                print(f"Database error: {str(db_error)}")

        # Start background task
        asyncio.create_task(save_to_database_background())

        # Return immediately without waiting for database
        cleaned_result['database'] = {
            'status': 'saving',
            'message': f'Saving {len(tiles)} roads to database in background'
        }

        return cleaned_result

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/database-stats")
async def get_database_stats():
    """Get overall database statistics"""
    try:
        stats = await db.get_database_statistics()
        return {
            'success': True,
            'statistics': stats
        }
    except Exception as e:
        print(f"Error getting database stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/roads/nearby")
async def get_nearby_roads(lat: float, lon: float, radius: float = 0.25, limit: int = 100):
    """
    Get roads near a location from database (for iOS apps)

    Parameters:
    - lat: Latitude
    - lon: Longitude
    - radius: Search radius in miles (default 0.25)
    - limit: Max number of roads to return (default 100)

    Example: /api/roads/nearby?lat=27.8428&lon=-82.8106&radius=0.5
    """
    try:
        roads = await db.get_roads_in_area(lat, lon, radius, limit)
        return {
            'success': True,
            'location': {'lat': lat, 'lon': lon},
            'radius_miles': radius,
            'roads_found': len(roads),
            'roads': roads
        }
    except Exception as e:
        print(f"Error getting nearby roads: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/roads/{road_id}")
async def get_road_detail(road_id: str):
    """
    Get detailed information about a specific road including waypoints

    Example: /api/roads/abc123def456
    """
    try:
        road = await db.get_road_with_waypoints(road_id)
        if not road:
            raise HTTPException(status_code=404, detail="Road not found")

        return {
            'success': True,
            'road': road
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting road detail: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Map interface showing filtered real roads"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Pet Search Grid - Final Solution</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <style>
            body { margin: 0; padding: 20px; font-family: Arial; }
            #map { height: 600px; margin-top: 20px; border: 2px solid #333; }
            .controls { margin-bottom: 20px; }
            button {
                padding: 10px 20px; font-size: 16px; cursor: pointer;
                background: #4CAF50; color: white; border: none;
            }
            button:hover { background: #45a049; }
            input { padding: 8px; width: 500px; }
            .info { margin: 10px 0; padding: 10px; background: #E8F5E9; }
            .success { background: #C8E6C9; padding: 10px; margin: 10px 0; }
            .error { background: #FFCDD2; padding: 10px; margin: 10px 0; }
            .stats {
                background: #FFF3E0; padding: 10px; margin: 10px 0;
                border-left: 4px solid #FF9800;
            }
        </style>
    </head>
    <body>
        <h1>🔍 Pet Search Grid - Real Roads Only</h1>
        <div class="controls">
            <input type="text" id="address" placeholder="Enter any address"
                   value="12196 86th avenue north, Seminole, FL 33772">
            <button onclick="createSearch()">Generate Search Grid</button>
        </div>
        <div id="info" class="info">
            Uses OSMNX for comprehensive coverage but filters out fake diagonal connections
        </div>
        <div id="stats" class="stats" style="display:none;"></div>
        <div id="map"></div>

        <script>
            let map = L.map('map').setView([27.8428, -82.8106], 16);
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
            let currentLayers = [];

            async function createSearch() {
                document.getElementById('info').innerHTML = '⏳ Loading roads and filtering fake connections...';
                document.getElementById('stats').style.display = 'none';

                const address = document.getElementById('address').value;

                try {
                    // Send address to server for geocoding
                    console.log('Sending address to server:', address);

                    // Get search grid (server will geocode if needed)
                    const response = await fetch('/api/create-search', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({ lat: 0, lon: 0, radius_miles: 0.25, address })
                    });

                    const data = await response.json();

                    if (!response.ok) {
                        throw new Error(data.detail || 'Server error');
                    }

                    // Clear old layers
                    currentLayers.forEach(l => map.removeLayer(l));
                    currentLayers = [];

                    // Center map
                    map.setView([data.center.lat, data.center.lon], 16);

                    // Add center marker
                    const marker = L.marker([data.center.lat, data.center.lon])
                        .addTo(map).bindPopup('Search Center');
                    currentLayers.push(marker);

                    // Colors for roads
                    const colors = [
                        '#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336',
                        '#00BCD4', '#8BC34A', '#E91E63', '#3F51B5', '#FFC107'
                    ];

                    let namedRoads = 0;
                    let unnamedRoads = 0;

                    // Draw roads
                    data.tiles.forEach((tile, i) => {
                        const validPts = tile.waypoints
                            .filter(p => !isNaN(p.lat) && !isNaN(p.lon))
                            .map(p => [p.lat, p.lon]);

                        if (validPts.length > 1) {
                            // Use different colors for named vs unnamed
                            const color = tile.has_name ?
                                colors[i % colors.length] :
                                '#808080';  // Gray for unnamed

                            const line = L.polyline(validPts, {
                                color: color,
                                weight: tile.has_name ? 4 : 3,
                                opacity: tile.has_name ? 0.9 : 0.6
                            }).addTo(map);

                            line.bindPopup(`
                                <b>${tile.roads[0]}</b><br>
                                Type: ${tile.highway_type}<br>
                                Length: ${tile.length_meters}m<br>
                                Points: ${tile.point_count}<br>
                                Named: ${tile.has_name ? 'Yes' : 'No'}
                            `);

                            currentLayers.push(line);

                            if (tile.has_name) namedRoads++;
                            else unnamedRoads++;
                        }
                    });

                    // Update UI
                    document.getElementById('info').className = 'success';
                    document.getElementById('info').innerHTML =
                        `✅ SUCCESS! Showing ${data.total_tiles} real roads`;

                    document.getElementById('stats').style.display = 'block';
                    document.getElementById('stats').innerHTML = `
                        <strong>Search Grid Statistics:</strong><br>
                        • Total roads: ${data.total_tiles}<br>
                        • Named roads: ${namedRoads} (colored)<br>
                        • Service roads: ${unnamedRoads} (gray)<br>
                        • Fake diagonals removed: ${data.filtered_count}<br>
                        • Coverage: 0.25 mile radius
                    `;

                } catch (error) {
                    document.getElementById('info').className = 'error';
                    document.getElementById('info').innerHTML = '❌ ' + error.message;
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8106)
