# Multi-stage build for Next.js + Python
FROM python:3.11-slim as python-base

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python files
COPY research_system.py config.py ./
COPY api/ ./api/

FROM node:18-slim as node-base

# Install Python in the Node.js image
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

# Copy Python installation from python-base
COPY --from=python-base /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=python-base /usr/local/bin /usr/local/bin
COPY --from=python-base /app /app

# Set up Node.js app
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# Copy Next.js app
COPY . .
RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]