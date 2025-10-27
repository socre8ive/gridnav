# Pet Search API

A geographic grid-based API system for coordinating lost pet search efforts. The API divides search areas into manageable grids, assigns them to volunteers, and tracks search progress in real-time.

## Overview

Pet Search API enables efficient coordination of search efforts for lost pets by:
- Creating geographic search grids based on location and radius
- Assigning grid sections to volunteer searchers
- Tracking real-time progress and coverage
- Managing search assignments with time-based expiration
- Providing detailed status updates on search efforts

## Architecture

```
iOS App / Client
    ↓
Cloudflare Workers (CORS proxy)
    ↓
api.psar.app → Nginx (Port 443) → FastAPI (Port 8443)
    ↓
SQLite Database
```

## Features

- **Geographic Grid Generation**: Automatically creates search grids based on location and radius
- **Smart Assignment**: Assigns grid sections to volunteers with configurable timeframes
- **Progress Tracking**: Real-time updates on searcher locations and covered areas
- **Road Coverage**: Tracks which roads have been searched within each grid
- **API Key Authentication**: Secure access to all endpoints
- **SSL/TLS**: Production-ready with Let's Encrypt certificates

## API Endpoints

### Create Search
Creates a new search area with geographic grids.

```bash
POST /api/create-search
Headers: X-API-Key: <your-api-key>
Body: {
  "lat": 27.8428,
  "lon": -82.8106,
  "radius_miles": 1.5,
  "grid_size_miles": 0.3,
  "address": "Seminole, FL",
  "pet_id": "unique_pet_identifier"
}
```

### Assign Grid
Assigns a grid section to a searcher.

```bash
POST /api/assign-grid
Headers: X-API-Key: <your-api-key>
Body: {
  "search_id": "search-abc123",
  "pet_id": "unique_pet_identifier",
  "searcher_id": "volunteer_001",
  "searcher_name": "John Doe",
  "timeframe_minutes": 60
}
```

### Update Progress
Updates searcher location and progress.

```bash
POST /api/update-progress
Headers: X-API-Key: <your-api-key>
Body: {
  "assignment_id": "assign-def456",
  "search_id": "search-abc123",
  "pet_id": "unique_pet_identifier",
  "grid_id": 1,
  "searcher_id": "volunteer_001",
  "lat": 27.8430,
  "lon": -82.8100,
  "accuracy_meters": 10.0,
  "roads_covered": []
}
```

### Get Grid Status
Retrieves status of all grids in a search.

```bash
GET /api/grid-status?search_id=search-abc123
Headers: X-API-Key: <your-api-key>
```

## Installation

### Requirements
- Python 3.8+
- pip
- nginx (for production)

### Setup

1. Clone the repository
```bash
git clone <repository-url>
cd petsearch
```

2. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize the database
```bash
python3 setup_tracking_tables.py
```

6. Run the server
```bash
# Development
python3 server_geographic_grids.py

# Production (using systemd)
sudo systemctl start petsearch
```

## Configuration

The API requires the following environment variables in `.env`:

```
API_KEY=your_secure_api_key_here
DATABASE_PATH=/path/to/pets.db
PORT=8443
```

## Database Schema

The system uses SQLite with the following main tables:
- `searches`: Search configurations and metadata
- `grids`: Individual grid sections within searches
- `assignments`: Grid assignments to searchers
- `progress_updates`: Real-time location and progress tracking
- `roads`: Road coverage tracking

## Production Deployment

The production system uses:
- **Domain**: api.psar.app
- **SSL**: Let's Encrypt with auto-renewal
- **Web Server**: Nginx (reverse proxy)
- **App Server**: Gunicorn with FastAPI
- **Process Manager**: systemd

### Systemd Service

```bash
sudo systemctl enable petsearch
sudo systemctl start petsearch
sudo systemctl status petsearch
```

## Development

### Running Tests
```bash
# Check database schema
python3 check_db_tables.py

# Verify active searches
python3 check_active_searchers.py

# Check recent progress
python3 check_recent_progress.py
```

## Cloudflare Workers

The API is designed to work with Cloudflare Workers for CORS handling and global edge distribution. See `CLOUDFLARE_WORKER_DOCUMENTATION.md` for setup instructions.

## Security

- API key authentication on all endpoints
- HTTPS/TLS encryption
- CORS configuration for web clients
- Input validation and sanitization

## License

[Your License Here]

## Support

For issues and questions, please open an issue on GitHub.
