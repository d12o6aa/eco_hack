# Agri-Mind Dashboard - Deployment Guide

## 🚀 Deployment Options

### Option 1: Streamlit Community Cloud (Recommended for Demo)

**Pros:**
- Free hosting
- Automatic SSL/HTTPS
- Easy deployment from GitHub
- Built-in authentication

**Steps:**

1. **Prepare GitHub Repository**
```bash
git init
git add .
git commit -m "Initial commit - Agri-Mind Dashboard"
git remote add origin https://github.com/YOUR_USERNAME/agri-mind.git
git push -u origin main
```

2. **Deploy to Streamlit Cloud**
- Go to https://share.streamlit.io
- Sign in with GitHub
- Click "New app"
- Select your repository: `YOUR_USERNAME/agri-mind`
- Set main file: `app.py`
- Click "Deploy!"

3. **Configuration**
- App URL will be: `https://YOUR_USERNAME-agri-mind.streamlit.app`
- Set environment variables in Streamlit Cloud dashboard if needed

**Estimated deployment time:** 5 minutes

---

### Option 2: Heroku

**Pros:**
- Free tier available
- Easy scaling
- Custom domain support
- Add-ons ecosystem

**Setup Files Required:**

Create `Procfile`:
```
web: sh setup.sh && streamlit run app.py
```

Create `setup.sh`:
```bash
mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
```

Create `runtime.txt`:
```
python-3.11.0
```

**Deploy:**
```bash
heroku login
heroku create agri-mind-dashboard
git push heroku main
heroku open
```

**Estimated deployment time:** 10 minutes

---

### Option 3: AWS EC2

**Pros:**
- Full control
- Scalable
- Enterprise-ready
- Integration with other AWS services

**Steps:**

1. **Launch EC2 Instance**
- AMI: Ubuntu 22.04 LTS
- Instance type: t2.micro (free tier) or t2.small
- Security group: Allow ports 22 (SSH) and 8501 (Streamlit)

2. **Connect and Setup**
```bash
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python
sudo apt install python3-pip python3-venv -y

# Clone repository
git clone https://github.com/YOUR_USERNAME/agri-mind.git
cd agri-mind

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. **Run with systemd (for auto-restart)**

Create `/etc/systemd/system/agri-mind.service`:
```ini
[Unit]
Description=Agri-Mind Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/agri-mind
Environment="PATH=/home/ubuntu/agri-mind/venv/bin"
ExecStart=/home/ubuntu/agri-mind/venv/bin/streamlit run app.py --server.port 8501 --server.address 0.0.0.0

[Install]
WantedBy=multi-user.target
```

Start service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable agri-mind
sudo systemctl start agri-mind
```

4. **Setup Nginx Reverse Proxy (Optional)**
```bash
sudo apt install nginx -y
```

Create `/etc/nginx/sites-available/agri-mind`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/agri-mind /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

**Estimated deployment time:** 30 minutes

---

### Option 4: Docker Container

**Pros:**
- Consistent environment
- Easy to scale
- Deploy anywhere
- CI/CD friendly

**Create Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Copy application
COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Create docker-compose.yml:**
```yaml
version: '3.8'

services:
  agri-mind:
    build: .
    ports:
      - "8501:8501"
    environment:
      - DEMO_MODE=true
    restart: unless-stopped
    volumes:
      - ./data:/app/data  # Optional: for data persistence
```

**Deploy:**
```bash
# Build
docker-compose build

# Run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

**Estimated deployment time:** 15 minutes

---

## 🔧 Environment Variables

Set these environment variables for production:

```bash
# Optional: API Keys for future satellite integration
PLANETARY_COMPUTER_API_KEY=your_key_here
SENTINEL_HUB_CLIENT_ID=your_client_id
SENTINEL_HUB_CLIENT_SECRET=your_client_secret

# Optional: Analytics
GOOGLE_ANALYTICS_ID=UA-XXXXXXXXX-X

# Optional: Database (for future features)
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

---

## 🔒 Security Best Practices

1. **HTTPS Only**
   - Use SSL certificates (Let's Encrypt is free)
   - Force HTTPS redirects

2. **Authentication** (for production)
   - Implement Streamlit-authenticator
   - Or use OAuth with Google/Microsoft

3. **API Keys**
   - Never commit API keys to Git
   - Use environment variables
   - Rotate keys regularly

4. **Rate Limiting**
   - Implement request throttling
   - Use Cloudflare for DDoS protection

5. **Data Privacy**
   - Encrypt farm location data
   - GDPR compliance if serving EU users
   - Regular security audits

---

## 📊 Monitoring & Analytics

### Application Monitoring

**1. Streamlit Cloud (Built-in)**
- Auto-included with Streamlit Cloud deployment
- View metrics at app dashboard

**2. Custom Logging**

Add to `app.py`:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agri_mind.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Use throughout app
logger.info("User analyzed farm")
logger.error("Satellite data fetch failed")
```

**3. Google Analytics**

Add to custom CSS section in `app.py`:
```html
<!-- Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=UA-XXXXXXXXX-X"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'UA-XXXXXXXXX-X');
</script>
```

---

## 🧪 Testing Before Deployment

### 1. Local Testing
```bash
streamlit run app.py
```
Test all features:
- [ ] Map loads correctly
- [ ] Farm selection works
- [ ] Analysis runs without errors
- [ ] Arabic text displays properly
- [ ] All tabs function
- [ ] Sustainability calculations are accurate

### 2. Load Testing

Install locust:
```bash
pip install locust
```

Create `locustfile.py`:
```python
from locust import HttpUser, task, between

class AagriMindUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def view_dashboard(self):
        self.client.get("/")
    
    @task(3)
    def analyze_farm(self):
        self.client.post("/_stcore/upload", {
            "crop_type": "wheat",
            "farm_area": 10
        })
```

Run test:
```bash
locust -f locustfile.py
```

---

## 📱 Mobile Optimization

While the dashboard works on mobile, for optimal experience:

1. **Responsive Design** - Already implemented
2. **Touch Gestures** - Map supports touch
3. **Mobile App** (Future):
   - Convert to React Native using Streamlit API
   - Or create PWA (Progressive Web App)

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflow

Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Streamlit Cloud

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python -m pytest tests/
    
    - name: Lint code
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

---

## 🎯 Performance Optimization

1. **Caching**
```python
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_satellite_data(coords):
    # Expensive operation
    return data
```

2. **Lazy Loading**
- Load map tiles on demand
- Defer non-critical components

3. **Image Optimization**
- Compress satellite imagery
- Use progressive loading

4. **CDN**
- Host static assets on CDN
- Use Cloudflare for global distribution

---

## 💰 Cost Estimation

### Free Tier (Recommended for Pitch)
- **Streamlit Cloud**: Free (with limits)
- **Cost**: $0/month
- **Suitable for**: Demo, MVP, <500 users/day

### Startup Tier
- **Heroku**: $7/month
- **Cost**: ~$10/month total
- **Suitable for**: 1,000-5,000 users/day

### Growth Tier
- **AWS EC2 t2.medium**: $35/month
- **RDS Database**: $25/month
- **CloudFront CDN**: $10/month
- **Cost**: ~$70/month total
- **Suitable for**: 10,000+ users/day

### Enterprise Tier
- **AWS EC2 cluster**: $200+/month
- **Load Balancer**: $20/month
- **RDS Multi-AZ**: $100/month
- **S3 + CloudFront**: $50/month
- **Cost**: $370+/month
- **Suitable for**: 100,000+ users/day

---

## 🆘 Troubleshooting Deployment

**Issue**: Port already in use
```bash
# Kill process on port 8501
sudo lsof -ti:8501 | xargs kill -9
```

**Issue**: Permission denied
```bash
# Fix permissions
chmod +x run.sh
chmod 755 app.py
```

**Issue**: Out of memory
```bash
# Increase swap (Linux)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

---

## ✅ Pre-Launch Checklist

- [ ] All features tested locally
- [ ] Arabic text displays correctly
- [ ] Demo mode works flawlessly
- [ ] Sustainability calculations verified
- [ ] Mobile responsiveness checked
- [ ] Load testing completed
- [ ] Security audit passed
- [ ] Analytics configured
- [ ] Backup strategy in place
- [ ] Documentation updated
- [ ] Team trained on system
- [ ] Support email/system ready

---

## 🎬 Launch Day Checklist

- [ ] Final deployment test
- [ ] Monitor server resources
- [ ] Watch error logs in real-time
- [ ] Have rollback plan ready
- [ ] Support team on standby
- [ ] Social media posts scheduled
- [ ] Press kit prepared
- [ ] Demo video ready

---

**Good luck with your launch! 🚀🌾**
