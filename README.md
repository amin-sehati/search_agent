# AI Research Assistant

A Next.js web application for conducting comprehensive research using AI agents, with real-time progress updates and multiple data sources.

## Features

- **Interactive Web Interface**: Clean, modern UI built with Next.js and Tailwind CSS
- **Real-time Progress Updates**: Live streaming of research progress using Server-Sent Events
- **Multi-source Research**: Integrates Tavily and Firecrawl APIs for comprehensive data gathering
- **AI-Powered Analysis**: Uses GPT-4 for planning, analysis, and report generation
- **Responsive Design**: Works on desktop and mobile devices
- **Export Functionality**: Save research results as JSON files

## Architecture

- **Frontend**: Next.js 14 with TypeScript and Tailwind CSS
- **Backend**: Next.js API Routes with direct Python integration
- **AI Framework**: LangGraph for orchestrating research agents
- **Data Sources**: Tavily (web search) and Firecrawl (content extraction)

## Deployment

### Railway (Recommended)

1. **Fork/Clone** this repository
2. **Connect to Railway**: 
   - Go to [railway.app](https://railway.app)
   - Click "Deploy from GitHub repo"
   - Select your forked repository
3. **Set Environment Variables** in Railway dashboard:
   - `GOOGLE_API_KEY`: Your Cerebras API key
   - `TAVILY_API_KEY`: Your Tavily API key  
   - `FIRECRAWL_API_KEY`: Your Firecrawl API key
   - `AUTH_PASSWORD`: Your login password
   - `DEFAULT_MODEL`: gemini-2.5-flash
4. **Deploy**: Railway will automatically build and deploy using the `railway.toml` and `nixpacks.toml` configuration

### Vercel (Alternative)

1. **Fork/Clone** this repository
2. **Connect to Vercel**: Import your repository to Vercel
3. **Set Environment Variables** in Vercel dashboard:
   - `GOOGLE_API_KEY`: Your Cerebras API key
   - `TAVILY_API_KEY`: Your Tavily API key  
   - `FIRECRAWL_API_KEY`: Your Firecrawl API key
   - `AUTH_PASSWORD`: Your login password
4. **Deploy**: Vercel will automatically build and deploy

### Local Development

1. **Install Dependencies**:
   ```bash
   # Python dependencies
   pip install -r requirements.txt
   
   # Node.js dependencies
   npm install
   ```

2. **Set Environment Variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start Development Server**:
   ```bash
   # Option 1: Use the helper script
   python dev.py
   
   # Option 2: Start directly
   npm run dev
   ```

4. **Access Application**: Open http://localhost:3000

## Usage

1. **Enter Research Query**: Type your research question in the text area
2. **Start Research**: Click "Start Research" to begin the workflow
3. **Monitor Progress**: Watch real-time updates as the system:
   - Plans research questions
   - Searches multiple data sources
   - Analyzes and synthesizes results
4. **Review Results**: Browse through:
   - Generated research questions
   - Executive summary
   - Full detailed report
   - Source citations
5. **Export**: Save results as JSON for future reference

## Research Workflow

1. **Planning Agent**: Breaks down the query into focused research questions
2. **Tavily Agent**: Performs web searches for each question
3. **Firecrawl Agent**: Extracts detailed content from relevant pages
4. **Analysis Agent**: Synthesizes findings into comprehensive reports

## Dependencies

Total package size optimized for Vercel deployment (< 200MB):

**Frontend** (~50KB):
- Next.js 14
- React 18
- Tailwind CSS
- Lucide React (icons)

**Backend** (~120MB):
- Next.js API Routes
- Python research system
- LangGraph
- Cerebras (LLaMA-4)
- Async HTTP libraries

## Configuration

The application requires three API keys:

- **Cerebras**: For LLaMA-4 analysis and report generation
- **Tavily**: For web search capabilities
- **Firecrawl**: For content extraction and scraping

## Security

- Environment variables are securely managed
- API keys are never exposed to the frontend
- CORS is properly configured for API access

## Performance

- Streaming responses for real-time updates
- Optimized bundle size for fast loading
- Efficient async processing
- Responsive UI with loading states