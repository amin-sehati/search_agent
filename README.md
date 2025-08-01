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
- **Backend**: FastAPI with async research workflow
- **AI Framework**: LangGraph for orchestrating research agents
- **Data Sources**: Tavily (web search) and Firecrawl (content extraction)

## Deployment

### Vercel (Recommended)

1. **Fork/Clone** this repository
2. **Connect to Vercel**: Import your repository to Vercel
3. **Set Environment Variables** in Vercel dashboard:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `TAVILY_API_KEY`: Your Tavily API key  
   - `FIRECRAWL_API_KEY`: Your Firecrawl API key
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

3. **Start Development Servers**:
   ```bash
   # Option 1: Run both servers with one command
   python dev.py
   
   # Option 2: Run separately in different terminals
   # Terminal 1: Start FastAPI backend  
   cd api && python main.py
   
   # Terminal 2: Start Next.js frontend
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
- FastAPI
- LangGraph
- OpenAI client
- Async HTTP libraries

## Configuration

The application requires three API keys:

- **OpenAI**: For GPT-4 analysis and report generation
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