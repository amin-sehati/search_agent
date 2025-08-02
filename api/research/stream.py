from http.server import BaseHTTPRequestHandler
import json
import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator
import os
import sys

# Add project root to path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from research_system import CompanyResearcher, Company
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class CustomStreamingCompanyResearcher(CompanyResearcher):
    def __init__(self, *args, progress_handler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_handler = progress_handler

    async def conduct_company_research_streaming(self, query: str):
        if self.progress_handler:
            self.progress_handler.update_progress(
                "User Input", "🎯 Processing market/topic query...", 1
            )

        # Run the initial discovery workflow
        discovery_state = await self.conduct_company_research(query)

        if self.progress_handler:
            self.progress_handler.update_progress(
                "Company Discovery", "✅ Company list created", 2
            )

        # Return the discovery state for user review
        return discovery_state

    async def continue_company_research_streaming(self, state, user_companies=None):
        if self.progress_handler:
            self.progress_handler.update_progress(
                "Company Info", "🔍 Gathering detailed company information...", 3
            )

        # Continue with detailed info gathering
        final_state = await self.continue_with_company_info(state, user_companies)

        if self.progress_handler:
            self.progress_handler.update_progress(
                "Complete", "🎉 Company research completed!", 4
            )

        return final_state


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Parse request body
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode("utf-8"))

            query = request_data.get("query")
            if not query:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Query is required"}).encode())
                return

            # Set up streaming response
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            # Run async research
            asyncio.run(self.stream_research(query))

        except Exception as e:
            logger.error(f"Handler error: {e}")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    async def stream_research(self, query: str):
        try:
            # Add detailed logging for Vercel debugging
            logger.info("=== VERCEL ENVIRONMENT DEBUG ===")
            logger.info(f"TAVILY_API_KEY present: {bool(os.getenv('TAVILY_API_KEY'))}")
            logger.info(f"TAVILY_API_KEY length: {len(os.getenv('TAVILY_API_KEY', ''))}")
            logger.info(f"FIRECRAWL_API_KEY present: {bool(os.getenv('FIRECRAWL_API_KEY'))}")
            logger.info(f"OPENAI_API_KEY present: {bool(os.getenv('OPENAI_API_KEY'))}")
            
            config_status = Config.validate_config()
            if not config_status["valid"]:
                error_msg = f"Configuration error: {config_status['issues']}"
                logger.error(f"Config validation failed: {error_msg}")
                self.wfile.write(
                    f"data: {json.dumps({'error': error_msg, 'type': 'error'})}\n\n".encode()
                )
                return

            progress_event = ProgressEvent()
            progress_handler = StreamingProgressHandler(progress_event)

            researcher = CustomStreamingCompanyResearcher(
                openai_api_key=Config.OPENAI_API_KEY,
                tavily_api_key=Config.TAVILY_API_KEY,
                firecrawl_api_key=Config.FIRECRAWL_API_KEY,
                model="gpt-4o",
                progress_handler=progress_handler,
            )

            # Start company discovery in background
            discovery_task = asyncio.create_task(researcher.conduct_company_research_streaming(query))

            # Stream progress updates for discovery phase
            last_event_count = 0
            while not discovery_task.done():
                current_event_count = len(progress_event.events)
                if current_event_count > last_event_count:
                    for event in progress_event.events[last_event_count:]:
                        self.wfile.write(
                            f"data: {json.dumps({'type': 'progress', 'data': event})}\n\n".encode()
                        )
                        self.wfile.flush()
                    last_event_count = current_event_count

                await asyncio.sleep(0.5)

            # Get discovery result
            discovery_result = await discovery_task

            # Send any remaining progress events
            current_event_count = len(progress_event.events)
            if current_event_count > last_event_count:
                for event in progress_event.events[last_event_count:]:
                    self.wfile.write(
                        f"data: {json.dumps({'type': 'progress', 'data': event})}\n\n".encode()
                    )
                    self.wfile.flush()

            # Serialize companies for frontend
            serialized_companies = []
            for company in discovery_result.get("companies_list", []):
                if hasattr(company, "__dict__"):
                    serialized_companies.append(company.__dict__)
                else:
                    serialized_companies.append(company)

            # Send company discovery result for user review
            discovery_data = {
                "query": discovery_result["original_query"],
                "market_topic": discovery_result["market_topic"],
                "companies": serialized_companies,
                "total_companies": len(serialized_companies),
                "tavily_source_count": discovery_result.get("tavily_source_count", 0),
                "firecrawl_source_count": discovery_result.get("firecrawl_source_count", 0),
                "total_sources": discovery_result.get("total_sources", 0),
                "timestamp": datetime.now().isoformat(),
                "awaiting_user_input": True,
                "step": "company_review"
            }

            self.wfile.write(
                f"data: {json.dumps({'type': 'company_discovery', 'data': discovery_data})}\n\n".encode()
            )
            self.wfile.flush()

        except Exception as e:
            logger.error(f"Streaming research error: {e}")
            self.wfile.write(
                f"data: {json.dumps({'error': str(e), 'type': 'error'})}\n\n".encode()
            )
