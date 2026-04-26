# Deployment Guide

## Local Deployment

See [README.md](../README.md) for quick start instructions.

## Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t drawing-error-correction .

# Run with environment variables
docker run -d \
  -p 5000:5000 \
  -e MULTIMODAL_API_URL='https://your-api-endpoint.com' \
  -e MULTIMODAL_API_KEY='your-api-key' \
  --name drawing-app \
  drawing-error-correction
```

### Docker Compose

```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Custom Configuration

Mount a custom `config.py` file:

```bash
docker run -d \
  -p 5000:5000 \
  -v /path/to/your/config.py:/app/config.py \
  -v /path/to/your/data:/app/data \
  drawing-error-correction
```

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 20M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    location /uploads/ {
        alias /path/to/drawing-error-correction/uploads/;
        expires 1h;
    }
}
```

### Systemd Service

```ini
[Unit]
Description=Drawing Error Correction Service
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/drawing-error-correction
Environment="MULTIMODAL_API_URL=https://your-api-endpoint.com"
Environment="MULTIMODAL_API_KEY=your-api-key"
ExecStart=/opt/drawing-error-correction/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

## Security Checklist

- [ ] `config.py` is not in version control
- [ ] API keys are set via environment variables or `config.py`
- [ ] Upload directory is not accessible directly
- [ ] File upload size limits are configured
- [ ] HTTPS is enabled in production
- [ ] Regular security updates for dependencies
