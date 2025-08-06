import asyncio
import json
import logging
import os
import sys
import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Annotated, TypedDict
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.types import interrupt, Command

# from langchain_openai import ChatOpenAI
from langchain_cerebras import ChatCerebras
from langchain.schema import HumanMessage, AIMessage
from tavily import TavilyClient
from firecrawl import AsyncFirecrawlApp, ScrapeOptions

load_dotenv()

# Global configuration parameters
TAVILY_DEFAULT_MAX_RESULTS = 3
FIRECRAWL_DEFAULT_MAX_RESULTS = 3
INITIAL_TAVILY_RESULTS = 3
DETAIL_TAVILY_RESULTS = 2
INITIAL_FIRECRAWL_RESULTS = 3
DETAIL_FIRECRAWL_RESULTS = 2
CUSTOM_COMPANY_TAVILY_RESULTS = 2
CUSTOM_COMPANY_FIRECRAWL_RESULTS = 2
BATCH_SIZE = 4
MAX_COMPANY_EXTRACT_CONTENT_LENGTH = 1000
PROFILE_MAX_SOURCES = 10
PROFILE_MAX_CONTENT_LENGTH = 24000
CUSTOM_COMPANY_CONTENT_SUMMARY_LENGTH = 10000
COMPANY_EXTRACT_SOURCE_CONTENT_LENGTH = 10000
COMPANY_EXTRACT_CONTENT_LENGTH = 8000
COMPANY_BOILERPLATE_PER_SOURCE = 500
TAVILY_SUMMARY_MAX_LENGTH = 4000
FIRECRAWL_SUMMARY_MAX_LENGTH = 4000
SUMMARY_CHUNK_SIZE = 2000


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    title: str
    url: str
    description: str
    content: str = ""
    source: str = ""


class Company(BaseModel):
    name: str = Field(..., description="Complete company name.")
    description: str = Field(
        ...,
        description="Detailed description of what the company does, its business model, key products/services, target market, customer base, market position, competitive advantages, and headquarters location.",
    )
    reasoning: str = Field(
        ..., description="Why this company is relevant to the market."
    )
    year_established: str = Field(
        ...,
        description="The year the company was established. 'Unknown' if not available.",
    )
    still_in_business: bool = Field(
        ..., description="Whether the company is still in business."
    )
    history: str = Field(
        ...,
        description="Key history, milestones, achievements, funding rounds, founders, key leadership, recent developments, and reasons for going out of business if applicable.",
    )
    future_roadmap: Optional[str] = Field(
        default="", description="Future roadmap of the company, if available."
    )
    sources: List[SearchResult] = Field(default_factory=list, exclude=True)

    @field_validator("future_roadmap", mode="before")
    @classmethod
    def validate_future_roadmap(cls, v):
        return v or ""


class CompanyResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    original_query: str
    market_topic: str
    company_discovery_tavily: List[SearchResult]
    company_discovery_firecrawl: List[SearchResult]
    company_discovery_llm: List[SearchResult]
    companies_list: List[Company]
    user_modified_companies: List[Company]
    detailed_company_info: Dict[str, List[SearchResult]]
    final_company_pages: Dict[str, str]
    current_step: str
    awaiting_user_input: bool
    user_added_companies: List[Company]
    approved_companies: List[Company]
    pending_custom_companies: List[str]
    human_approval_completed: bool


class TavilyRetriever:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = TavilyClient(api_key=api_key)

    async def search(
        self, query: str, max_results: int = TAVILY_DEFAULT_MAX_RESULTS
    ) -> List[SearchResult]:
        logger.info(f"Tavily API search query: {query}")
        response = self.client.search(
            query=query,
            max_results=min(max_results, TAVILY_DEFAULT_MAX_RESULTS),
            search_depth="advanced",
            include_raw_content=True,
        )
        return self._format_tavily_results(response)

    def _format_tavily_results(self, data: Dict) -> List[SearchResult]:
        return [
            SearchResult(
                title=item.get("title"),
                url=item.get("url"),
                description=item.get("content"),
                content=item.get("raw_content"),
                source="tavily",
            )
            for item in data.get("results")
        ]


class FirecrawlRetriever:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.app = AsyncFirecrawlApp(api_key=api_key)

    async def search(
        self, query: str, max_results: int = FIRECRAWL_DEFAULT_MAX_RESULTS
    ) -> List[SearchResult]:
        logger.info(f"Firecrawl search query: {query}")
        search_result = await self.app.search(
            query,
            limit=min(max_results, FIRECRAWL_DEFAULT_MAX_RESULTS),
            scrape_options=ScrapeOptions(formats=["markdown"]),
        )
        return self._format_firecrawl_results(search_result)

    def _format_firecrawl_results(self, response: dict) -> List[SearchResult]:
        results = response.get("data")
        return [
            SearchResult(
                title=item.get("title"),
                url=item.get("url"),
                description=item.get("description"),
                content=item.get("markdown"),
                source="firecrawl",
            )
            for item in results
        ]


class UserInputAgent:
    def __init__(self, llm: ChatCerebras, stream_progress_callback):
        self.llm = llm
        self.stream_progress = stream_progress_callback

    async def process_user_input(self, state: CompanyResearchState) -> Dict[str, Any]:
        self.stream_progress(
            "Market Topic Extraction",
            "Writing market description based on your query...",
            5,
        )
        query = state["original_query"]
        market_extraction_prompt = f"""Based on the user's query: "{query}" which sets the market for the research, write a short description of the market concisely and
        precisely that would help the search agents to find companies in the market. Formulat your answer as a query that would be asked to the search agents to 
        find companies in the market. Answer with the query only, no other text. (answer in less than 200 characters)"""

        response = await self.llm.ainvoke(
            [HumanMessage(content=market_extraction_prompt)]
        )
        market_topic = response.content.strip()
        self.stream_progress(
            "Market Topic Extraction", f"Market topic identified: {market_topic}", 10
        )

        return {
            "market_topic": market_topic,
            "current_step": "company_discovery",
            "messages": [AIMessage(content=f"Market/topic identified: {market_topic}")],
        }


class TavilyCompanyDiscoveryAgent:
    def __init__(
        self, retriever: TavilyRetriever, llm: ChatCerebras, stream_progress_callback
    ):
        self.retriever = retriever
        self.llm = llm
        self.stream_progress = stream_progress_callback

    async def discover_companies(self, state: CompanyResearchState) -> Dict[str, Any]:
        self.stream_progress(
            "Company Discovery",
            "Starting comprehensive company research with Tavily...",
            10,
        )
        market_topic = state["market_topic"]
        company_query = state["original_query"]
        all_results = await self.retriever.search(
            company_query, max_results=INITIAL_TAVILY_RESULTS
        )

        if not all_results:
            logger.warning("Tavily returned no results.")

        self.stream_progress(
            "Company Discovery",
            f"Tavily found {len(all_results)} company sources.",
            15,
        )

        return {
            "company_discovery_tavily": all_results,
            "messages": [
                AIMessage(
                    content=f"Tavily gathered detailed information from {len(all_results)} sources."
                )
            ],
        }


class FirecrawlCompanyDiscoveryAgent:
    def __init__(
        self, retriever: FirecrawlRetriever, llm: ChatCerebras, stream_progress_callback
    ):
        self.retriever = retriever
        self.llm = llm
        self.stream_progress = stream_progress_callback

    async def discover_companies(self, state: CompanyResearchState) -> Dict[str, Any]:
        self.stream_progress(
            "Company Discovery",
            "Starting comprehensive company research with Firecrawl...",
            25,
        )
        market_topic = state["market_topic"]
        company_query = state["original_query"]
        initial_results = await self.retriever.search(
            company_query, max_results=INITIAL_FIRECRAWL_RESULTS
        )
        if not initial_results:
            logger.warning("Firecrawl returned no results.")

        self.stream_progress(
            "Company Discovery",
            f"Firecrawl found {len(initial_results)} sources.",
            27,
        )
        all_results = initial_results
        self.stream_progress(
            "Company Discovery",
            f"Firecrawl gathered information from {len(all_results)} sources.",
            30,
        )

        return {
            "company_discovery_firecrawl": all_results,
            "messages": [
                AIMessage(
                    content=f"Firecrawl gathered detailed information from {len(all_results)} sources."
                )
            ],
        }


class TavilySummarizer:
    def __init__(self, llm: ChatCerebras, stream_progress_callback):
        self.llm = llm
        self.stream_progress = stream_progress_callback

    async def summarize_results(
        self, tavily_results: List[SearchResult], market_topic: str
    ) -> SearchResult:
        if not tavily_results:
            return SearchResult(
                title="No Tavily Results",
                url="internal://no-results",
                description="No Tavily search results available",
                content="",
                source="tavily_summary",
            )

        self.stream_progress(
            "Content Summarization",
            f"Summarizing {len(tavily_results)} Tavily search results...",
            32,
        )

        # Combine all Tavily content
        combined_content = ""
        for i, result in enumerate(tavily_results):
            chunk = f"Source {i+1}: {result.title}\nURL: {result.url}\nDescription: {result.description}\nContent: {result.content[:SUMMARY_CHUNK_SIZE]}\n\n---\n\n"
            combined_content += chunk

        # Truncate if too long
        if len(combined_content) > TAVILY_SUMMARY_MAX_LENGTH * 2:
            combined_content = (
                combined_content[: TAVILY_SUMMARY_MAX_LENGTH * 2]
                + "\n\n[Content truncated due to length]"
            )

        prompt = f"""
        Analyze and summarize the following Tavily search results for the {market_topic} market.
        
        Focus on extracting:
        1. Company names and their business descriptions
        2. Market trends and industry insights
        3. Key players and their market positions
        4. Business models and competitive advantages
        5. Financial information and funding details
        6. Recent developments and news
        
        Content to analyze:
        {combined_content}
        
        Provide a comprehensive summary that preserves all company-specific information while condensing redundant details.
        Organize the summary to highlight individual companies and their characteristics.
        Maximum length: {TAVILY_SUMMARY_MAX_LENGTH} characters.
        """

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        summary_content = response.content.strip()

        return SearchResult(
            title=f"Tavily Search Summary: {market_topic} Market Analysis",
            url="internal://tavily-summary",
            description=f"AI-generated summary of {len(tavily_results)} Tavily search results for {market_topic}",
            content=summary_content,
            source="tavily_summary",
        )


class FirecrawlSummarizer:
    def __init__(self, llm: ChatCerebras, stream_progress_callback):
        self.llm = llm
        self.stream_progress = stream_progress_callback

    async def summarize_results(
        self, firecrawl_results: List[SearchResult], market_topic: str
    ) -> SearchResult:
        if not firecrawl_results:
            return SearchResult(
                title="No Firecrawl Results",
                url="internal://no-results",
                description="No Firecrawl search results available",
                content="",
                source="firecrawl_summary",
            )

        self.stream_progress(
            "Content Summarization",
            f"Summarizing {len(firecrawl_results)} Firecrawl search results...",
            36,
        )

        # Combine all Firecrawl content
        combined_content = ""
        for i, result in enumerate(firecrawl_results):
            chunk = f"Source {i+1}: {result.title}\nURL: {result.url}\nDescription: {result.description}\nContent: {result.content[:SUMMARY_CHUNK_SIZE]}\n\n---\n\n"
            combined_content += chunk

        # Truncate if too long
        if len(combined_content) > FIRECRAWL_SUMMARY_MAX_LENGTH * 2:
            combined_content = (
                combined_content[: FIRECRAWL_SUMMARY_MAX_LENGTH * 2]
                + "\n\n[Content truncated due to length]"
            )

        prompt = f"""
        Analyze and summarize the following Firecrawl search results for the {market_topic} market.
        
        Focus on extracting:
        1. Company names and their business descriptions
        2. Detailed company profiles and operations
        3. Market positioning and strategies
        4. Technology and product information
        5. Leadership and organizational details
        6. Historical context and evolution
        
        Content to analyze:
        {combined_content}
        
        Provide a comprehensive summary that preserves all company-specific information while condensing redundant details.
        Organize the summary to highlight individual companies and their unique characteristics.
        Maximum length: {FIRECRAWL_SUMMARY_MAX_LENGTH} characters.
        """

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        summary_content = response.content.strip()

        return SearchResult(
            title=f"Firecrawl Search Summary: {market_topic} Market Deep Dive",
            url="internal://firecrawl-summary",
            description=f"AI-generated summary of {len(firecrawl_results)} Firecrawl search results for {market_topic}",
            content=summary_content,
            source="firecrawl_summary",
        )


class LLMKnowledgeAgent:
    def __init__(self, llm: ChatCerebras, stream_progress_callback):
        self.llm = llm
        self.stream_progress = stream_progress_callback

    async def discover_companies(self, state: CompanyResearchState) -> Dict[str, Any]:
        self.stream_progress(
            "Enhanced Company Discovery",
            "Gathering comprehensive company information from LLM knowledge...",
            15,
        )
        market_topic = state["market_topic"]

        prompt = f"""
        Based on your knowledge, provide detailed information about real companies (including startups and established businesses) 
        that operate in the {market_topic} market space. For each company, provide comprehensive details including:
        1. Company name
        2. Detailed description of what they do and their business model
        3. Year established (if known)
        4. Current business status (active, acquired, closed, etc.)
        5. Key history and milestones
        6. Market position and competitive advantages
        7. Future roadmap (if available)
        8. Why they're relevant to this market
        9. Notable achievements or funding rounds (if applicable)
        10. The reason that they went out of business (if applicable)

        Focus on both well-known companies and emerging players. Provide up to 8 companies with rich detail.
        Format your response as detailed company profiles with clear sections for each company.
        If you don't know specific information (like exact founding year), indicate "Information not available" rather than guessing.
        """

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        llm_content = response.content

        # Convert LLM response into SearchResult format for consistency
        results = [
            SearchResult(
                title=f"LLM Knowledge: Detailed Company Profiles in {market_topic}",
                url="llm://comprehensive-knowledge-base",
                description=f"Comprehensive company profiles generated from LLM knowledge for {market_topic} market",
                content=llm_content,
                source="llm_knowledge",
            )
        ]

        self.stream_progress(
            "Company Discovery",
            f"LLM provided comprehensive company profiles for {market_topic} market.",
            22,
        )

        return {
            "company_discovery_llm": results,
            "messages": [
                AIMessage(
                    content=f"LLM provided detailed company profiles for {market_topic} market."
                )
            ],
        }


class CustomCompanyResearchAgent:
    def __init__(
        self,
        tavily_retriever: TavilyRetriever,
        firecrawl_retriever: FirecrawlRetriever,
        llm: ChatCerebras,
        stream_progress_callback,
    ):
        self.tavily_retriever = tavily_retriever
        self.firecrawl_retriever = firecrawl_retriever
        self.llm = llm
        self.stream_progress = stream_progress_callback

    async def research_custom_companies(
        self, state: CompanyResearchState
    ) -> Dict[str, Any]:
        custom_company_names = state["pending_custom_companies"]
        if not custom_company_names:
            # No custom companies to research, proceed to final synthesis
            return {
                "current_step": "final_synthesis",
                "messages": [AIMessage(content="No custom companies to research.")],
            }

        self.stream_progress(
            "Custom Company Research",
            f"Starting detailed research for {len(custom_company_names)} custom companies...",
            56,
        )

        market_topic = state["market_topic"]
        custom_companies = []
        all_custom_sources = []

        for i, company_name in enumerate(custom_company_names):
            self.stream_progress(
                "Custom Company Research",
                f"Researching {company_name} ({i + 1}/{len(custom_company_names)})...",
                56 + int(20 * (i / len(custom_company_names))),
            )

            # Gather comprehensive information for custom company
            custom_company_search_result = await self._research_single_custom_company(
                company_name, market_topic
            )
            all_custom_sources.extend(custom_company_search_result)

            # Create basic company profile
            company_profile = await self._create_company_profile(
                company_name, custom_company_search_result, market_topic
            )
            custom_companies.append(company_profile)

        # Add custom companies to approved companies
        approved_companies = state["approved_companies"]
        all_approved_companies = approved_companies + custom_companies

        # Update detailed company info with custom company sources
        detailed_company_info = state["detailed_company_info"]
        for i, company in enumerate(custom_companies):
            company_sources = [
                src
                for src in all_custom_sources
                if company.name.lower() in src.content.lower()
                or company.name.lower() in src.title.lower()
            ]
            if not company_sources:
                logger.warning(
                    f"No specific sources found for custom company: {company.name}"
                )
            detailed_company_info[company.name] = company_sources

        self.stream_progress(
            "Custom Company Research",
            f"Completed research for {len(custom_companies)} custom companies.",
            75,
        )

        return {
            "approved_companies": all_approved_companies,
            "detailed_company_info": detailed_company_info,
            "pending_custom_companies": [],
            "current_step": "final_synthesis",
            "messages": [
                AIMessage(
                    content=f"Completed detailed research for {len(custom_companies)} custom companies."
                )
            ],
        }

    async def _research_single_custom_company(
        self, company_name: str, market_topic: str
    ) -> List[SearchResult]:
        """Research a single custom company using both Tavily and Firecrawl"""
        # Search queries for comprehensive information
        query = f"""Find all information such as business profile, history, and market position on company called "{company_name}" in the {market_topic} market"""

        all_results = []

        # Parallel search with both retrievers
        tavily_task = self.tavily_retriever.search(
            query, max_results=CUSTOM_COMPANY_TAVILY_RESULTS
        )
        firecrawl_task = self.firecrawl_retriever.search(
            query, max_results=CUSTOM_COMPANY_FIRECRAWL_RESULTS
        )

        tavily_results, firecrawl_results = await asyncio.gather(
            tavily_task, firecrawl_task, return_exceptions=True
        )

        # Handle results safely
        if not isinstance(tavily_results, Exception):
            all_results.extend(tavily_results)
        if not isinstance(firecrawl_results, Exception):
            all_results.extend(firecrawl_results)

        return all_results

    async def _create_company_profile(
        self, company_name: str, sources: List[SearchResult], market_topic: str
    ) -> Company:
        """Create a company profile from research sources"""
        content_summary = ""

        # Use LLM to extract company information
        prompt = f"""
        Based on the following research about {company_name}, extract comprehensive company information:
        
        Research Content: {content_summary[:CUSTOM_COMPANY_CONTENT_SUMMARY_LENGTH]}
        
        Provide detailed information including:
        1. Complete company name
        2. Detailed description of what they do and their business model
        3. Year established (if known)
        4. Current business status (active, acquired, closed, etc.)
        5. Key history and milestones
        6. Market position and competitive advantages
        7. Why they're relevant to the {market_topic} market
        8. Notable achievements or funding rounds (if applicable)
        9. The reason that they went out of business (if applicable)
        10. Headquarters location (if available)
        11. Founders and key leadership (if available)
        12. Key products/services offered
        13. Target market and customer base
        14. Recent developments or news (if available)
        
        Format your response with clear sections for each piece of information.
        If specific information is not available in the research content, indicate "Information not available from the provided sources."
        Keep descriptions concise but informative and focus on accuracy.
        """

        structured_llm = self.llm.with_structured_output(Company)
        company_profile = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        company_profile.sources = sources
        return company_profile


class CompanyListSynthesizer:
    def __init__(self, llm: ChatCerebras, stream_progress_callback):
        self.llm = llm
        self.stream_progress = stream_progress_callback
        self.tavily_summarizer = TavilySummarizer(llm, stream_progress_callback)
        self.firecrawl_summarizer = FirecrawlSummarizer(llm, stream_progress_callback)

    async def synthesize_company_list(
        self, state: CompanyResearchState
    ) -> Dict[str, Any]:
        self.stream_progress(
            "Company Synthesis", "Synthesizing company list from search results...", 35
        )
        tavily_results = state["company_discovery_tavily"]
        firecrawl_results = state["company_discovery_firecrawl"]
        llm_results = state["company_discovery_llm"]
        market_topic = state["market_topic"]
        user_added_companies = state["user_added_companies"]

        # Create summarized versions of the raw search results
        tavily_summary = await self.tavily_summarizer.summarize_results(
            tavily_results, market_topic
        )
        firecrawl_summary = await self.firecrawl_summarizer.summarize_results(
            firecrawl_results, market_topic
        )

        # Combine summaries with LLM knowledge for synthesis
        summarized_sources = [tavily_summary, firecrawl_summary] + llm_results
        all_sources = (
            tavily_results + firecrawl_results + llm_results
        )  # Keep original sources for streaming

        self.stream_progress(
            "Company Synthesis",
            f"Created summaries from {len(all_sources)} sources. Using {len(summarized_sources)} summarized sources for analysis.",
            40,
        )

        discovered_companies = await self._extract_companies(
            market_topic, summarized_sources
        )
        self.stream_progress(
            "Company Synthesis",
            f"Extracted {len(discovered_companies)} companies from sources.",
            45,
        )

        # Stream individual company results
        for i, company in enumerate(discovered_companies):
            company_event = {
                "type": "company_found",
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "name": company.name,
                    "description": company.description,
                    "reasoning": company.reasoning,
                    "year_established": company.year_established,
                    "still_in_business": company.still_in_business,
                    "history": company.history,
                    "future_roadmap": company.future_roadmap or "",
                    "index": i + 1,
                    "total": len(discovered_companies),
                },
            }
            import json

            print(json.dumps(company_event))

        # After company synthesis, stream the raw source content
        self.stream_progress(
            "Source Content",
            "Streaming raw research sources...",
            50,
        )

        # Stream Tavily content
        for i, result in enumerate(tavily_results):
            content_event = {
                "type": "source_content",
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "source": "tavily",
                    "title": result.title,
                    "url": result.url,
                    "description": result.description,
                    "content": result.content,
                    "index": i + 1,
                    "total": len(tavily_results),
                },
            }
            print(json.dumps(content_event))

        # Stream Firecrawl content
        for i, result in enumerate(firecrawl_results):
            content_event = {
                "type": "source_content",
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "source": "firecrawl",
                    "title": result.title,
                    "url": result.url,
                    "description": result.description,
                    "content": result.content,
                    "index": i + 1,
                    "total": len(firecrawl_results),
                },
            }
            print(json.dumps(content_event))

        # Stream LLM content
        for i, result in enumerate(llm_results):
            content_event = {
                "type": "source_content",
                "data": {
                    "timestamp": datetime.now().isoformat(),
                    "source": "llm_knowledge",
                    "title": result.title,
                    "url": result.url,
                    "description": result.description,
                    "content": result.content,
                    "index": i + 1,
                    "total": len(llm_results),
                },
            }
            print(json.dumps(content_event))

        # Stream summary content
        summary_event = {
            "type": "source_content",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "source": "tavily_summary",
                "title": tavily_summary.title,
                "url": tavily_summary.url,
                "description": tavily_summary.description,
                "content": tavily_summary.content,
                "index": 1,
                "total": 1,
            },
        }
        print(json.dumps(summary_event))

        summary_event = {
            "type": "source_content",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "source": "firecrawl_summary",
                "title": firecrawl_summary.title,
                "url": firecrawl_summary.url,
                "description": firecrawl_summary.description,
                "content": firecrawl_summary.content,
                "index": 1,
                "total": 1,
            },
        }
        print(json.dumps(summary_event))
        all_companies = discovered_companies + user_added_companies

        unique_companies = list(
            {
                company.name.lower().strip(): company for company in all_companies
            }.values()
        )
        if not unique_companies:
            self.stream_progress(
                "Company Synthesis", "Failed to extract any companies.", 50
            )
            raise Exception("Failed to extract any companies from research data.")

        self.stream_progress(
            "Company Synthesis", f"Found {len(unique_companies)} unique companies.", 50
        )

        # Since we now have detailed information, prepare for final synthesis
        detailed_company_info = {}
        for company in unique_companies:
            # Associate relevant sources with each company
            company_sources = []
            company_name_lower = company.name.lower()
            company_words = company_name_lower.split()

            # First, add summarized sources (they contain concentrated info)
            for source in summarized_sources:
                source_text = f"{source.content.lower()} {source.title.lower()} {source.description.lower()}"
                if company_name_lower in source_text or any(
                    word in source_text for word in company_words if len(word) > 2
                ):
                    company_sources.append(source)

            # Then add relevant original sources
            for source in all_sources:
                if source in summarized_sources:
                    continue  # Skip duplicates

                source_text = f"{source.content.lower()} {source.title.lower()} {source.description.lower()}"
                if company_name_lower in source_text or any(
                    word in source_text for word in company_words if len(word) > 2
                ):
                    company_sources.append(source)

            # If still no sources, use broader matching
            if not company_sources:
                logger.warning(
                    f"No direct sources found for company: {company.name}, trying broader matching..."
                )
                # Use all sources as fallback for broader analysis
                company_sources = (
                    summarized_sources + all_sources[:3]
                )  # Limit fallback sources

            detailed_company_info[company.name] = company_sources

        return {
            "companies_list": unique_companies,
            "detailed_company_info": detailed_company_info,
            "current_step": "final_synthesis",
            "awaiting_user_input": False,
            "tavily_source_count": len(tavily_results),
            "firecrawl_source_count": len(firecrawl_results),
            "llm_source_count": len(llm_results),
            "total_sources": len(all_sources),
            "messages": [
                AIMessage(
                    content=f"Found {len(unique_companies)} companies with detailed information in {market_topic} market."
                )
            ],
        }

    async def _extract_companies(
        self, market_topic: str, sources: List[SearchResult]
    ) -> List[Company]:
        valid_sources = [s for s in sources if s.content]
        if not valid_sources:
            return []

        num_sources = len(valid_sources)

        # Estimate total boilerplate length to calculate available content length
        boilerplate_per_source = COMPANY_BOILERPLATE_PER_SOURCE  # Approximate length of "Source: ...\nURL: ...\nContent: \n\n"
        total_boilerplate_length = num_sources * boilerplate_per_source

        available_content_length = (
            COMPANY_EXTRACT_CONTENT_LENGTH - total_boilerplate_length
        )
        if available_content_length <= 0:
            # If boilerplate is too long, just use titles and URLs
            content_per_source = 0
        else:
            content_per_source = available_content_length // num_sources

        content_chunks = []
        for s in valid_sources:
            content_slice = s.content[:content_per_source]
            chunk = f"Source: {s.title}\nURL: {s.url}\nContent: {content_slice}\n\n"
            content_chunks.append(chunk)

        # if not content_chunks:
        #     raise Exception(
        #         f"No content available to extract companies for market: {market_topic}"
        #     )

        combined_content = "".join(content_chunks)
        prompt = f"""
        Extract companies from the generated report for {market_topic}:
        {combined_content}
        
        Find real companies operating in this market space using the content provided. Focus on actual companies mentioned in the sources
        that are directly related to the market segment of {market_topic}. Maximum 10 companies.
        
        For each company, extract comprehensive information and fit it into the Company structure:
        - name: Complete company name
        - description: Detailed description including business model, key products/services, target market, customer base, market position, competitive advantages, and headquarters location
        - reasoning: Why this company is relevant to the {market_topic} market
        - year_established: Year the company was established (use "Unknown" if not available)
        - still_in_business: Whether the company is currently active/operational
        - history: Key history, milestones, achievements, funding rounds, founders, key leadership, recent developments, and reasons for going out of business if applicable
        
        Use all relevant content from the sources to provide comprehensive company profiles.
        If specific information is not available in the research content, indicate "Information not available from the provided sources."
        """

        # Create a Pydantic model for multiple companies
        class CompanyList(BaseModel):
            companies: List[Company] = Field(
                ..., description="List of extracted companies"
            )

        structured_llm = self.llm.with_structured_output(CompanyList)
        company_list_response = await structured_llm.ainvoke(
            [HumanMessage(content=prompt)]
        )

        return company_list_response.companies


class FinalCompanySynthesizer:
    def __init__(self, llm: ChatCerebras, stream_progress_callback):
        self.llm = llm
        self.stream_progress = stream_progress_callback

    async def create_company_pages(self, state: CompanyResearchState) -> Dict[str, Any]:
        self.stream_progress(
            "Final Synthesis", "Starting final synthesis of company profiles...", 60
        )
        companies = (
            state["approved_companies"]
            or state["user_modified_companies"]
            or state["companies_list"]
        )
        detailed_info = state["detailed_company_info"]
        final_pages = {}
        total_companies = len(companies)
        batch_size = BATCH_SIZE

        async def create_single_page(company, index):
            self.stream_progress(
                "Final Synthesis",
                f"Generating profile for {company.name} ({index + 1}/{total_companies})...",
                60 + int(40 * (index / total_companies)),
            )
            sources = detailed_info[company.name]
            page_content = await self._create_company_page(company, sources)
            return company.name, page_content

        for i in range(0, total_companies, batch_size):
            batch = companies[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[create_single_page(c, i + j) for j, c in enumerate(batch)]
            )
            for name, content in batch_results:
                final_pages[name] = content

        self.stream_progress(
            "Final Synthesis", "All company profiles have been generated.", 100
        )
        return {"final_company_pages": final_pages, "current_step": "complete"}

    async def _create_company_page(
        self, company: Company, sources: List[SearchResult]
    ) -> str:
        MAX_CONTENT_LENGTH = PROFILE_MAX_CONTENT_LENGTH
        content_chunks = []
        total_length = 0

        for source in sources:
            content_text = source.content
            if not content_text:
                continue
            if not content_text.strip():
                continue

            # Calculate remaining space and truncate content if needed
            base_chunk = f"Source: {source.title}\nURL: {source.url}\nContent: "
            ending = "\n\n---\n\n"
            remaining_space = (
                MAX_CONTENT_LENGTH - total_length - len(base_chunk) - len(ending)
            )

            if remaining_space <= 0:
                break

            # Truncate content if it's too long
            truncated_content = (
                content_text[:remaining_space]
                if len(content_text) > remaining_space
                else content_text
            )
            chunk = base_chunk + truncated_content + ending

            content_chunks.append(chunk)
            total_length += len(chunk)

        if not content_chunks:
            # Try to use description or title if no content is available
            for source in sources:
                if source.description or source.title:
                    chunk = f"Source: {source.title}\nURL: {source.url}\nContent: {source.description or source.title}\n\n---\n\n"
                    content_chunks.append(chunk)
                    break

        if not content_chunks:
            raise Exception(
                f"No source content available to generate profile for: {company.name}. Total sources: {len(sources)}"
            )

        combined_content = "".join(content_chunks)
        prompt = f"""
        Generate a comprehensive company profile in Markdown format for "{company.name}".
        
        Company Name: {company.name}
        Initial Description: {company.description}
        Company Status: {"Active" if company.still_in_business else "Inactive"}
        Year Established: {company.year_established}
        Market Relevance: {company.reasoning}
        
        Collected Research Data:
        ---
        {combined_content}
        ---
        
        Instructions:
        1. Create a detailed profile using the following Markdown structure.
        2. Extract and synthesize ALL relevant information from the "Collected Research Data" to create comprehensive sections.
        3. Provide detailed, substantive content for each section based on the research data.
        4. Include specific details like numbers, dates, locations, products, services, funding, leadership, etc.
        5. If information is not available for a section, write "Information not available from the provided sources."
        6. Do NOT invent information - only use what's provided in the research data.
        7. Make each section as detailed and informative as possible using the available data.

        Required Markdown Structure:
        # {company.name}
        
        ## Company Overview
        [Provide a comprehensive overview including business model, main activities, target market, and overall company description. Include headquarters location, company size, and organizational structure if available.]
        
        ## Business Status & Operations
        [Detail current operational status, business health, recent performance, and operational scope. Include information about markets served, geographical presence, and business scale.]
        
        ## Products & Services
        [Describe in detail the company's offerings, product lines, services, technology platforms, and key features. Include pricing models, target customers, and competitive advantages.]
        
        ## Market Position & Strategy
        [Explain the company's position in its market, competitive landscape, market share, strategic initiatives, partnerships, and business strategy.]
        
        ## Financial Information & Funding
        [Include revenue figures, funding rounds, investor information, valuation, financial performance, and business metrics if available.]
        
        ## Leadership & Team
        [Detail key executives, founders, leadership team, company culture, and organizational information.]
        
        ## Company History & Milestones
        [Provide chronological history including founding story, major milestones, acquisitions, expansions, pivot points, and significant developments.]
        
        ## Recent Developments & News
        [Include latest news, recent announcements, product launches, strategic moves, and current initiatives.]
        
        ## Future Plans & Roadmap
        [Detail future plans, strategic roadmap, upcoming products, expansion plans, and growth strategies if available.]
        
        ## Sources
        [List all source URLs used as bulleted list.]
        """

        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()


class CompanyResearcher:
    def __init__(
        self,
        cerebras_api_key: str,
        tavily_api_key: str,
        firecrawl_api_key: str,
        model: str = "gpt-oss-120b",
    ):
        self.llm = ChatCerebras(model=model, temperature=0)
        self.tavily_retriever = TavilyRetriever(tavily_api_key)
        self.firecrawl_retriever = FirecrawlRetriever(firecrawl_api_key)
        self.user_input_agent = UserInputAgent(self.llm, self._stream_progress)
        self.tavily_discovery_agent = TavilyCompanyDiscoveryAgent(
            self.tavily_retriever, self.llm, self._stream_progress
        )
        self.firecrawl_discovery_agent = FirecrawlCompanyDiscoveryAgent(
            self.firecrawl_retriever, self.llm, self._stream_progress
        )
        self.llm_knowledge_agent = LLMKnowledgeAgent(self.llm, self._stream_progress)
        self.company_synthesizer = CompanyListSynthesizer(
            self.llm, self._stream_progress
        )
        self.custom_company_agent = CustomCompanyResearchAgent(
            self.tavily_retriever,
            self.firecrawl_retriever,
            self.llm,
            self._stream_progress,
        )
        self.final_synthesizer = FinalCompanySynthesizer(
            self.llm, self._stream_progress
        )
        self.checkpointer = (
            MemorySaver()
        )  # Required for human-in-the-loop functionality
        self.workflow = self._build_workflow()

    def _stream_progress(self, step: str, message: str, progress: int):
        event = {
            "type": "progress",
            "data": {
                "timestamp": datetime.now().isoformat(),
                "step": step,
                "message": message,
                "progress": progress,
            },
        }
        print(json.dumps(event))

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(CompanyResearchState)
        workflow.add_node("user_input", self.user_input_agent.process_user_input)
        workflow.add_node(
            "tavily_discovery", self.tavily_discovery_agent.discover_companies
        )
        workflow.add_node(
            "firecrawl_discovery", self.firecrawl_discovery_agent.discover_companies
        )
        workflow.add_node("llm_knowledge", self.llm_knowledge_agent.discover_companies)
        workflow.add_node(
            "company_synthesis", self.company_synthesizer.synthesize_company_list
        )
        workflow.add_node(
            "custom_company_research",
            self.custom_company_agent.research_custom_companies,
        )
        workflow.add_node(
            "final_synthesis", self.final_synthesizer.create_company_pages
        )

        workflow.set_entry_point("user_input")
        workflow.add_edge("user_input", "tavily_discovery")
        workflow.add_edge("user_input", "firecrawl_discovery")
        workflow.add_edge("user_input", "llm_knowledge")
        workflow.add_edge(
            ["tavily_discovery", "firecrawl_discovery", "llm_knowledge"],
            "company_synthesis",
        )
        workflow.add_edge("company_synthesis", "final_synthesis")
        workflow.add_edge("custom_company_research", "final_synthesis")
        workflow.add_edge("final_synthesis", END)
        return workflow.compile(checkpointer=self.checkpointer)

    async def conduct_company_research(
        self,
        query: str,
        user_companies: List[Company] = None,
    ) -> Dict[str, Any]:
        initial_state = {
            "messages": [HumanMessage(content=f"Research query: {query}")],
            "original_query": query,
            "market_topic": "",
            "company_discovery_tavily": [],
            "company_discovery_firecrawl": [],
            "company_discovery_llm": [],
            "companies_list": [],
            "user_modified_companies": user_companies or [],
            "detailed_company_info": {},
            "final_company_pages": {},
            "current_step": "user_input",
            "awaiting_user_input": False,
            "user_added_companies": user_companies or [],
            "approved_companies": [],
            "pending_custom_companies": [],
            "human_approval_completed": False,
        }
        # Create a configuration with thread_id for checkpointer
        config = {
            "configurable": {"thread_id": f"research_session_{id(initial_state)}"}
        }
        return await self.workflow.ainvoke(initial_state, config=config)

    async def add_custom_company(
        self, company_name: str, current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add a custom company to the research and generate its profile"""
        # Update the state with the new custom company
        current_state["pending_custom_companies"] = [company_name]
        current_state["current_step"] = "custom_company_research"

        # Use the existing custom company research agent
        updated_state = await self.custom_company_agent.research_custom_companies(
            current_state
        )

        # Generate final synthesis for the updated companies
        final_state = await self.final_synthesizer.create_company_pages(updated_state)

        return final_state

    async def continue_with_company_info(
        self, state: Dict[str, Any], user_modified_companies: List[Company] = None
    ) -> Dict[str, Any]:
        if user_modified_companies:
            state["approved_companies"] = (
                user_modified_companies  # Use approved_companies instead
            )
            # Associate sources with modified companies
            detailed_info = state["detailed_company_info"]
            for company in user_modified_companies:
                if company.name not in detailed_info:
                    detailed_info[company.name] = (
                        []
                    )  # Will be handled by final synthesizer
            state["detailed_company_info"] = detailed_info

        state["awaiting_user_input"] = False
        state["human_approval_completed"] = True
        state["pending_custom_companies"] = []
        state["current_step"] = "final_synthesis"

        # Create simplified workflow that just does final synthesis
        info_workflow = StateGraph(CompanyResearchState)
        info_workflow.add_node(
            "final_synthesis", self.final_synthesizer.create_company_pages
        )
        info_workflow.set_entry_point("final_synthesis")
        info_workflow.add_edge("final_synthesis", END)
        compiled_workflow = info_workflow.compile(checkpointer=self.checkpointer)

        # Create a configuration with thread_id for checkpointer
        config = {"configurable": {"thread_id": f"continue_session_{id(state)}"}}
        return await compiled_workflow.ainvoke(state, config=config)


async def main():
    parser = argparse.ArgumentParser(description="Conduct company research.")
    parser.add_argument("query", type=str, nargs="?", help="The research query.")
    args = parser.parse_args()

    if args.query:
        query = args.query
    else:
        query = input("Please enter your research query: ")

    from config import Config

    researcher = CompanyResearcher(
        cerebras_api_key=Config.CEREBRAS_API_KEY,
        tavily_api_key=Config.TAVILY_API_KEY,
        firecrawl_api_key=Config.FIRECRAWL_API_KEY,
        model=Config.DEFAULT_MODEL,
    )

    final_state = await researcher.conduct_company_research(query)

    result = {
        "type": "complete",
        "data": {
            "query": query,
            "market_topic": final_state["market_topic"],
            "company_pages": final_state["final_company_pages"],
            "total_companies": len(final_state.get("companies_list", [])),
            "timestamp": datetime.now().isoformat(),
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
