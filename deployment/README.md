# Deployment Configuration

This directory contains the configuration files needed to deploy the Pet Search API in production.

## Files

### petsearch.service
Systemd service file for running the API server.

**Installation:**
```bash
sudo cp petsearch.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable petsearch
sudo systemctl start petsearch
```

### nginx-api-psar.conf
Nginx reverse proxy configuration for the API.

**Installation:**
```bash
sudo cp nginx-api-psar.conf /etc/nginx/conf.d/api-psar.conf
sudo nginx -t
sudo systemctl reload nginx
```

## SSL Certificate Setup

The configuration uses Let's Encrypt certificates. To obtain them:

```bash
sudo certbot certonly --nginx -d api.psar.app
```

Certificates will be installed at:
- `/etc/letsencrypt/live/api.psar.app/fullchain.pem`
- `/etc/letsencrypt/live/api.psar.app/privkey.pem`

## Environment Variables

Copy `.env.example` to `.env` and fill in your actual values:

```bash
cp .env.example .env
# Edit .env with your configuration
```

## Quick Deployment Guide

1. Install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

3. Set up nginx:
   ```bash
   sudo cp deployment/nginx-api-psar.conf /etc/nginx/conf.d/api-psar.conf
   sudo systemctl reload nginx
   ```

4. Obtain SSL certificate:
   ```bash
   sudo certbot certonly --nginx -d api.psar.app
   ```

5. Install and start the service:
   ```bash
   sudo cp deployment/petsearch.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable petsearch
   sudo systemctl start petsearch
   ```

6. Check status:
   ```bash
   sudo systemctl status petsearch
   ```
