from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json
import logging
from datetime import datetime
from typing import AsyncGenerator, Dict, Any
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research_system import CompanyResearcher
from config import Config

app = FastAPI(title="AI Research Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchRequest(BaseModel):
    query: str


class ProgressEvent:
    def __init__(self):
        self.events = []

    def add_event(self, step: str, message: str, step_number: int = None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        event_data = {
            "timestamp": timestamp,
            "step": step,
            "message": message,
            "step_number": step_number,
            "progress": (step_number / 4) * 100 if step_number else 0,
        }
        self.events.append(event_data)


class StreamingProgressHandler:
    def __init__(self, progress_event: ProgressEvent):
        self.progress_event = progress_event
        self.current_step = 0
        self.total_steps = 4

    def update_progress(self, step_name: str, message: str, step_number: int = None):
        if step_number:
            self.current_step = step_number

        self.progress_event.add_event(step_name, message, step_number)


class CustomStreamingResearcher(CompanyResearcher):
    def __init__(self, *args, progress_handler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_handler = progress_handler

    async def conduct_research(self, query: str):
        if self.progress_handler:
            self.progress_handler.update_progress(
                "User Input", "🎯 Processing market/topic query...", 1
            )

        # Run company discovery
        result = await self.conduct_company_research(query)

        if self.progress_handler:
            self.progress_handler.update_progress(
                "Complete", "✅ Company discovery completed!", 2
            )

        return result


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/config/validate")
async def validate_config():
    try:
        config_status = Config.validate_config()
        return {
            "valid": config_status["valid"],
            "issues": config_status.get("issues", []),
        }
    except Exception as e:
        logger.error(f"Config validation error: {e}")
        return {"valid": False, "issues": [str(e)]}


@app.post("/research")
async def start_research(request: ResearchRequest):
    try:
        config_status = Config.validate_config()
        if not config_status["valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Configuration error: {config_status['issues']}",
            )

        researcher = CompanyResearcher(
            openai_api_key=Config.OPENAI_API_KEY,
            tavily_api_key=Config.TAVILY_API_KEY,
            firecrawl_api_key=Config.FIRECRAWL_API_KEY,
            model="gpt-4o",
        )

        result = await researcher.conduct_company_research(request.query)

        # Convert Company objects to dictionaries for JSON serialization
        serialized_companies = []
        for company in result.get("companies_list", []):
            if hasattr(company, "__dict__"):
                serialized_companies.append(company.__dict__)
            else:
                serialized_companies.append(company)

        response_data = {
            "query": result["original_query"],
            "market_topic": result.get("market_topic", ""),
            "companies": serialized_companies,
            "total_companies": len(serialized_companies),
            "timestamp": datetime.now().isoformat(),
        }

        return response_data

    except Exception as e:
        logger.error(f"Research error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/stream")
async def stream_research(request: ResearchRequest):
    async def generate_progress() -> AsyncGenerator[str, None]:
        try:
            config_status = Config.validate_config()
            if not config_status["valid"]:
                error_msg = f"Configuration error: {config_status['issues']}"
                yield f"data: {json.dumps({'error': error_msg, 'type': 'error'})}\n\n"
                return

            progress_event = ProgressEvent()
            progress_handler = StreamingProgressHandler(progress_event)

            researcher = CustomStreamingResearcher(
                openai_api_key=Config.OPENAI_API_KEY,
                tavily_api_key=Config.TAVILY_API_KEY,
                firecrawl_api_key=Config.FIRECRAWL_API_KEY,
                model="gpt-4o",
                progress_handler=progress_handler,
            )

            # Start research in background
            research_task = asyncio.create_task(
                researcher.conduct_research(request.query)
            )

            # Stream progress updates
            last_event_count = 0
            while not research_task.done():
                current_event_count = len(progress_event.events)
                if current_event_count > last_event_count:
                    for event in progress_event.events[last_event_count:]:
                        yield f"data: {json.dumps({'type': 'progress', 'data': event})}\n\n"
                    last_event_count = current_event_count

                await asyncio.sleep(0.5)

            # Get final result
            result = await research_task

            # Send any remaining progress events
            current_event_count = len(progress_event.events)
            if current_event_count > last_event_count:
                for event in progress_event.events[last_event_count:]:
                    yield f"data: {json.dumps({'type': 'progress', 'data': event})}\n\n"

            # Serialize companies
            serialized_companies = []
            for company in result.get("companies_list", []):
                if hasattr(company, "__dict__"):
                    serialized_companies.append(company.__dict__)
                else:
                    serialized_companies.append(company)

            # Send company discovery result
            final_data = {
                "query": result["original_query"],
                "market_topic": result.get("market_topic", ""),
                "companies": serialized_companies,
                "total_companies": len(serialized_companies),
                "timestamp": datetime.now().isoformat(),
                "awaiting_user_input": True,
                "step": "company_review"
            }

            yield f"data: {json.dumps({'type': 'company_discovery', 'data': final_data})}\n\n"

        except Exception as e:
            logger.error(f"Streaming research error: {e}")
            yield f"data: {json.dumps({'error': str(e), 'type': 'error'})}\n\n"

    return StreamingResponse(
        generate_progress(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
