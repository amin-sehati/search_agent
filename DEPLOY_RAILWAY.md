# Railway Deployment Guide

## Quick Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/)

## Manual Deployment

### Step 1: Prepare Your Repository

1. Fork this repository to your GitHub account
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/ai-research-assistant.git
   cd ai-research-assistant
   ```

### Step 2: Set Up Railway

1. Go to [railway.app](https://railway.app)
2. Sign up/Sign in with your GitHub account
3. Click **"Deploy from GitHub repo"**
4. Select your forked repository

### Step 3: Configure Environment Variables

In your Railway project dashboard, go to **Variables** tab and add:

**Required API Keys:**
```
CEREBRAS_API_KEY=your_cerebras_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
FIRECRAWL_API_KEY=your_firecrawl_api_key_here
```

**Application Configuration:**
```
AUTH_PASSWORD=your_secure_password_here
DEFAULT_MODEL=qwen-3-235b-a22b-instruct-2507
NODE_ENV=production
PYTHONUNBUFFERED=1
```

### Step 4: Deploy

Railway will automatically:
1. Detect the `railway.toml` and `nixpacks.toml` configuration
2. Install both Node.js and Python dependencies
3. Build the Next.js application
4. Start the application

### Step 5: Access Your Application

1. Railway will provide you with a URL (e.g., `https://your-app.railway.app`)
2. Visit the URL and log in with your `AUTH_PASSWORD`
3. Start conducting AI-powered research!

## Configuration Files

This repository includes the following Railway-specific files:

- `railway.toml` - Railway deployment configuration
- `nixpacks.toml` - Multi-language build configuration (Node.js + Python)
- `Procfile` - Process definition (backup)
- `.env.example` - Environment variables template

## API Keys Setup

### Cerebras API Key
1. Go to [Cerebras Cloud Platform](https://cloud.cerebras.ai/)
2. Sign up for an account
3. Generate an API key
4. Add it as `CEREBRAS_API_KEY`

### Tavily API Key
1. Go to [Tavily](https://tavily.com/)
2. Sign up for an account
3. Generate an API key
4. Add it as `TAVILY_API_KEY`

### Firecrawl API Key
1. Go to [Firecrawl](https://firecrawl.dev/)
2. Sign up for an account
3. Generate an API key
4. Add it as `FIRECRAWL_API_KEY`

## Troubleshooting

### Build Failures

If your deployment fails, check the Railway logs:
1. Go to your project dashboard
2. Click on **Deployments**
3. View the build/deploy logs

Common issues:
- Missing environment variables
- Invalid API keys
- Network connectivity during build

### Python Dependencies

If Python packages fail to install:
1. Check that `requirements.txt` is properly formatted
2. Verify that Railway has access to PyPI
3. Check the build logs for specific errors

### Performance

Railway provides:
- **Memory**: 512MB default (upgradeable)
- **CPU**: Shared compute (upgradeable)
- **Storage**: Ephemeral filesystem
- **Bandwidth**: Unlimited

## Scaling

For production use:
1. Upgrade your Railway plan for more resources
2. Consider using Railway's database add-ons if needed
3. Set up monitoring and alerts
4. Configure custom domains

## Support

- **Railway Documentation**: [docs.railway.app](https://docs.railway.app)
- **Railway Community**: [Discord](https://discord.gg/railway)
- **Application Issues**: Open an issue in this repository