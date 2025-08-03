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

            state = request_data.get("state")
            user_companies_data = request_data.get("user_companies", [])
            
            if not state:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "State is required"}).encode())
                return

            # Set up streaming response
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            # Convert user companies data back to Company objects
            user_companies = []
            for company_data in user_companies_data:
                company = Company(
                    name=company_data.get("name", ""),
                    description=company_data.get("description", ""),
                    reasoning=company_data.get("reasoning", ""),
                    year_established=company_data.get("year_established"),
                    still_in_business=company_data.get("still_in_business"),
                    history=company_data.get("history", ""),
                    future_roadmap=company_data.get("future_roadmap", "")
                )
                user_companies.append(company)

            # Run async company info gathering
            asyncio.run(self.stream_company_info(state, user_companies))

        except Exception as e:
            logger.error(f"Handler error: {e}")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    async def stream_company_info(self, state, user_companies):
        try:
            config_status = Config.validate_config()
            if not config_status["valid"]:
                error_msg = f"Configuration error: {config_status['issues']}"
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

            # Start company info gathering in background
            info_task = asyncio.create_task(
                researcher.continue_company_research_streaming(state, user_companies)
            )

            # Stream progress updates
            last_event_count = 0
            while not info_task.done():
                current_event_count = len(progress_event.events)
                if current_event_count > last_event_count:
                    for event in progress_event.events[last_event_count:]:
                        self.wfile.write(
                            f"data: {json.dumps({'type': 'progress', 'data': event})}\n\n".encode()
                        )
                        self.wfile.flush()
                    last_event_count = current_event_count

                await asyncio.sleep(0.5)

            # Get final result
            final_result = await info_task

            # Send any remaining progress events
            current_event_count = len(progress_event.events)
            if current_event_count > last_event_count:
                for event in progress_event.events[last_event_count:]:
                    self.wfile.write(
                        f"data: {json.dumps({'type': 'progress', 'data': event})}\n\n".encode()
                    )
                    self.wfile.flush()

            # Serialize SearchResult objects
            def serialize_search_results(results):
                return [res.__dict__ for res in results]

            # Send final result with company pages
            final_data = {
                "query": final_result["original_query"],
                "market_topic": final_result["market_topic"],
                "company_pages": final_result.get("final_company_pages", {}),
                "total_companies": len(final_result.get("final_company_pages", {})),
                "timestamp": datetime.now().isoformat(),
                "company_discovery_tavily": serialize_search_results(final_result.get("company_discovery_tavily", [])),
                "company_discovery_firecrawl": serialize_search_results(final_result.get("company_discovery_firecrawl", [])),
            }

            self.wfile.write(
                f"data: {json.dumps({'type': 'complete', 'data': final_data})}\n\n".encode()
            )
            self.wfile.flush()

        except Exception as e:
            logger.error(f"Streaming company info error: {e}")
            self.wfile.write(
                f"data: {json.dumps({'error': str(e), 'type': 'error'})}\n\n".encode()
            )
