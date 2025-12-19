#!/bin/bash

API_KEY="petsearch_2024_secure_key_f8d92a1b3c4e5f67"
BASE_URL="https://localhost:8443"

echo "=========================================="
echo "Testing Complete Search HTTP Endpoint"
echo "=========================================="

# Generate unique IDs
PET_ID="test-pet-http-$(date +%s)"
SEARCH_ID="test-search-http-$(date +%s)"
SEARCHER_ID="test-searcher-http-$(date +%s)"

echo ""
echo "Test IDs:"
echo "  Pet ID: $PET_ID"
echo "  Search ID: $SEARCH_ID"
echo "  Searcher ID: $SEARCHER_ID"

# Step 1: Create pet details
echo ""
echo "1. Creating pet details..."
curl -k -X POST "$BASE_URL/api/save-pet-details" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"pet_id\": \"$PET_ID\",
    \"pet_name\": \"HTTP Test Dog\",
    \"pet_photo_url\": \"https://example.com/dog.jpg\"
  }" 2>/dev/null | python3 -m json.tool

# Step 2: Create search (using Python script)
echo ""
echo "2. Creating search and assignment (via database)..."
source venv/bin/activate
python3 << EOF
import asyncio
from database import db

async def setup():
    # Create search
    await db.create_search_tracking(
        "$SEARCH_ID",
        "$PET_ID",
        "456 Test Avenue",
        37.7749,
        -122.4194,
        0.5,
        4
    )

    # Create assignment
    assignment = await db.assign_grid(
        "$SEARCH_ID",
        "$PET_ID",
        1,
        "$SEARCHER_ID",
        "HTTP Test User",
        30
    )

    print(f"✓ Created assignment: {assignment['assignment_id']}")

    # Add GPS points
    for i, (lat, lon) in enumerate([
        (37.7749, -122.4194),
        (37.7750, -122.4195),
        (37.7751, -122.4196)
    ]):
        await db.update_search_progress(
            assignment['assignment_id'],
            "$SEARCH_ID",
            "$PET_ID",
            1,
            "$SEARCHER_ID",
            lat, lon, 10.0,
            0.5 + (i * 0.3),
            5 + (i * 5)
        )
    print(f"✓ Added 3 GPS points")

asyncio.run(setup())
EOF

# Step 3: Complete the search via HTTP endpoint
echo ""
echo "3. Completing search via HTTP endpoint..."
COMPLETE_RESPONSE=$(curl -k -X POST "$BASE_URL/api/complete-search" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"search_id\": \"$SEARCH_ID\",
    \"pet_id\": \"$PET_ID\",
    \"searcher_id\": \"$SEARCHER_ID\",
    \"total_distance_miles\": 4.7,
    \"duration_minutes\": 62
  }" 2>/dev/null)

echo "$COMPLETE_RESPONSE" | python3 -m json.tool

# Step 4: Fetch search history via HTTP endpoint
echo ""
echo "4. Fetching search history via HTTP endpoint..."
curl -k "$BASE_URL/api/search-history?searcher_id=$SEARCHER_ID" \
  -H "X-API-Key: $API_KEY" 2>/dev/null | python3 -m json.tool

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
