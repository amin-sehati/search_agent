import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Annotated, TypedDict
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from tavily import TavilyClient
from firecrawl import FirecrawlApp

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    content: str = ""
    score: float = 0.0
    source: str = ""
    published_date: Optional[str] = None


@dataclass
class Company:
    name: str
    description: str
    reasoning: str
    year_established: Optional[str] = None
    still_in_business: Optional[bool] = None
    history: str = ""
    future_roadmap: str = ""
    sources: List[SearchResult] = None

    def __post_init__(self):
        if self.sources is None:
            self.sources = []


@dataclass
class ResearchContext:
    query: str
    sources: List[SearchResult]
    summary: str = ""
    report: str = ""
    citations: List[str] = None

    def __post_init__(self):
        if self.citations is None:
            self.citations = []


class CompanyResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    original_query: str
    market_topic: str
    company_discovery_tavily: List[SearchResult]
    company_discovery_firecrawl: List[SearchResult]
    companies_list: List[Company]
    user_modified_companies: List[Company]
    detailed_company_info: Dict[str, List[SearchResult]]
    final_company_pages: Dict[str, str]
    current_step: str
    awaiting_user_input: bool


# Pydantic models for structured LLM outputs
class MarketTopicExtraction(BaseModel):
    """Structured output for market topic extraction"""

    market_topic: str = Field(
        ...,
        description="A clear, specific description of the market, problem, or topic that companies would be addressing",
    )

    @field_validator("market_topic")
    @classmethod
    def validate_market_topic(cls, v):
        if len(v.strip()) < 5:
            raise ValueError("Market topic must be at least 5 characters long")
        return v.strip()


class CompanyInfo(BaseModel):
    """Structured output for individual company information"""

    name: str = Field(..., description="Company name")
    description: str = Field(
        ..., description="Brief description of what the company does"
    )
    reasoning: str = Field(
        ..., description="Why this company is relevant to the market"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError("Company name must be at least 2 characters long")
        return v.strip()

    @field_validator("description")
    @classmethod
    def validate_description(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Company description must be at least 10 characters long")
        return v.strip()

    @field_validator("reasoning")
    @classmethod
    def validate_reasoning(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Company reasoning must be at least 10 characters long")
        return v.strip()


class CompanyExtraction(BaseModel):
    """Structured output for company extraction from sources"""

    companies: List[CompanyInfo] = Field(
        ...,
        description="List of companies extracted from the research sources",
        min_length=1,
        max_length=10,
    )

    @field_validator("companies")
    @classmethod
    def validate_companies(cls, v):
        if len(v) == 0:
            raise ValueError("At least one company must be extracted")
        return v


class CompanyProfile(BaseModel):
    """Structured output for detailed company profile generation"""

    company_name: str = Field(..., description="Name of the company")
    overview: str = Field(..., description="Company overview section")
    status: str = Field(..., description="Business status section")
    market_position: str = Field(..., description="Market position section")
    key_facts: str = Field(..., description="Key facts section")
    sources: List[SearchResult] = Field(
        ..., description="Sources used to generate the profile"
    )

    @field_validator("company_name")
    @classmethod
    def validate_company_name(cls, v):
        if len(v.strip()) < 2:
            raise ValueError("Company name must be at least 2 characters long")
        return v.strip()

    @field_validator("overview", "status", "market_position", "key_facts")
    @classmethod
    def validate_sections(cls, v):
        if len(v.strip()) < 20:
            raise ValueError("Each section must be at least 20 characters long")
        return v.strip()

    @field_validator("sources")
    @classmethod
    def validate_sources(cls, v):
        if not v:
            raise ValueError("At least one source must be provided")
        return v


class BaseRetriever(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key

    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        pass

    @abstractmethod
    async def get_content(self, url: str) -> str:
        pass


class TavilyRetriever(BaseRetriever):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.client = TavilyClient(api_key=api_key)
        self.failure_count = 0
        self.max_failures = 3
        self.circuit_open = False

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        # Circuit breaker check
        if self.circuit_open:
            logger.warning("Tavily circuit breaker is open, skipping search")
            return []

        # Validate API key
        if not self.api_key or self.api_key.strip() == "":
            logger.error("Tavily API key is empty or not configured")
            self._handle_failure()
            return []

        # Log for debugging
        logger.info(f"Tavily API search query: {query}")
        logger.info(f"Max results: {min(max_results, 5)}")

        # Try multiple search configurations
        search_configs = [
            # Attempt 1: Basic search with raw content
            {
                "query": query,
                "max_results": min(max_results, 5),
                "search_depth": "basic",
                "include_raw_content": True,
                "include_answer": False,
            },
            # Attempt 2: Advanced search with raw content
            {
                "query": query,
                "max_results": min(max_results, 5),
                "search_depth": "advanced",
                "include_raw_content": True,
                "include_answer": False,
            },
            # Attempt 3: Simple search as fallback
            {
                "query": query,
                "max_results": min(max_results, 5),
            },
        ]

        for attempt_num, search_kwargs in enumerate(search_configs, 1):
            try:
                logger.info(
                    f"Tavily search attempt {attempt_num}/{len(search_configs)}"
                )

                # Use the official Tavily client
                response = self.client.search(**search_kwargs)

                if response and response.get("results"):
                    self.failure_count = 0  # Reset on success
                    formatted_results = self._format_tavily_results(response)
                    logger.info(
                        f"Tavily API success on attempt {attempt_num}: {len(formatted_results)} results"
                    )

                    # If we get results, return immediately
                    if len(formatted_results) > 0:
                        return formatted_results
                    else:
                        logger.warning(
                            f"Tavily attempt {attempt_num} returned 0 results, trying next attempt"
                        )
                        continue
                else:
                    logger.warning(f"Tavily attempt {attempt_num} returned no results")
                    continue

            except Exception as e:
                logger.warning(f"Tavily attempt {attempt_num} error: {e}")
                if attempt_num == len(search_configs):
                    logger.warning("All Tavily attempts failed with exceptions")
                    self._handle_failure()
                    return []

        # If we reach here, all attempts returned 0 results
        logger.info(
            "All Tavily attempts returned 0 results - continuing without Tavily data"
        )
        return []

    def _handle_failure(self):
        """Handle API failures and circuit breaker logic"""
        self.failure_count += 1
        if self.failure_count >= self.max_failures:
            self.circuit_open = True
            logger.warning(
                f"Tavily circuit breaker opened after {self.failure_count} failures"
            )

    def _format_tavily_results(self, data: Dict) -> List[SearchResult]:
        results = []
        for item in data.get("results", []):
            result = SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                content=item.get("raw_content", ""),
                score=item.get("score", 0.0),
                source="tavily",
                published_date=item.get("published_date"),
            )
            results.append(result)
        return results

    async def get_content(self, url: str) -> str:
        # Not implemented for this retriever
        return ""


class FirecrawlRetriever(BaseRetriever):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.app = FirecrawlApp(api_key=api_key)
        self.failure_count = 0
        self.max_failures = 3

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:

        # Log for debugging
        logger.info(f"Firecrawl search query: {query}")
        logger.info(f"Max results: {min(max_results, 5)}")

        try:
            # Use the official Firecrawl client
            search_params = {"query": query, "limit": min(max_results, 5)}

            response = self.app.search(**search_params)

            if response and response.data:
                self.failure_count = 0  # Reset on success
                formatted_results = self._format_firecrawl_results(response)
                logger.info(
                    f"Firecrawl search success: {len(formatted_results)} results"
                )
                return formatted_results
            else:
                logger.warning("Firecrawl returned no results")
                return []

        except Exception as e:
            logger.error(f"Firecrawl search error: {e}")
            self._handle_failure()
            return []

    def _handle_failure(self):
        """Handle API failures and circuit breaker logic"""
        self.failure_count += 1
        if self.failure_count >= self.max_failures:
            self.circuit_open = True
            logger.warning(
                f"Firecrawl circuit breaker opened after {self.failure_count} failures"
            )

    def _format_firecrawl_results(self, response) -> List[SearchResult]:
        results = []
        for item in response.data if hasattr(response, "data") else []:
            result = SearchResult(
                title=getattr(item, "title", ""),
                url=getattr(item, "url", ""),
                snippet=getattr(item, "description", "")[:500],
                content=getattr(item, "description", ""),
                score=1.0,
                source="firecrawl",
            )
            results.append(result)
        return results

    async def get_content(self, url: str) -> str:
        try:
            # Use the official Firecrawl client for scraping
            scrape_params = {"url": url, "formats": ["json", "markdown"]}

            response = self.app.scrape_url(**scrape_params)

            return response["data"]["json"]

        except Exception as e:
            logger.error(f"Firecrawl scrape error for {url}: {e}")
            return ""


class UserInputAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def process_user_input(self, state: CompanyResearchState) -> Dict[str, Any]:
        print("\n🎯 USER INPUT AGENT: Processing market/topic query...")
        print("=" * 60)

        query = state["original_query"]
        print(f"📝 Processing query: '{query}'", flush=True)

        market_extraction_prompt = f"""
        Extract the market, problem, or topic from this user query: "{query}"
        This will be used to search for companies operating in this space. Keep it short and concise (less than 300 characters)but 
        preserve the nuance of the user's query.
        """

        print(
            "🤖 Calling LLM for market topic extraction with structured output...",
            flush=True,
        )

        # Use structured output with Pydantic model
        llm_with_structured_output = self.llm.with_structured_output(
            MarketTopicExtraction
        )

        try:
            response = await llm_with_structured_output.ainvoke(
                [HumanMessage(content=market_extraction_prompt)]
            )
            market_topic = response.market_topic
            print(
                f"✅ Market/Topic extracted and validated: {market_topic}", flush=True
            )
        except Exception as e:
            print(
                f"❌ Structured output failed, falling back to text parsing: {e}",
                flush=True,
            )
            # Fallback to regular LLM call
            response = await self.llm.ainvoke(
                [HumanMessage(content=market_extraction_prompt)]
            )
            market_topic = response.content.strip()
            print(f"⚠️ Fallback market topic: {market_topic}", flush=True)

        print(f"✅ Market/Topic identified: {market_topic}")
        print("=" * 60)

        return {
            "market_topic": market_topic,
            "current_step": "company_discovery",
            "messages": [AIMessage(content=f"Market/topic identified: {market_topic}")],
        }


class CompanyDiscoveryPlannerAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def plan_research(self, state: CompanyResearchState) -> Dict[str, Any]:
        print("\n🎯 COMPANY DISCOVERY PLANNER: Creating company search queries...")
        print("=" * 60)

        market_topic = state["market_topic"]

        planning_prompt = f"""
        You are a company discovery planner. Your task is to create 4-5 specific search queries to find companies that operate in this market/topic: "{market_topic}"
        
        Create search queries that will help find:
        1. Companies directly solving this problem or operating in this market
        2. Both successful and failed companies in this space
        3. Startups, established companies, and competitors
        
        Each query should be:
        - Specific and targeted for finding company names and information
        - Designed to yield results about companies, not just general information
        - Focused on finding 10+ companies per query
        
        Format your response as a JSON list of strings, like this:
        ["Query 1", "Query 2", "Query 3"]
        
        Company Discovery Queries:
        """

        response = await self.llm.ainvoke([HumanMessage(content=planning_prompt)])

        try:
            queries = json.loads(response.content.strip())
            if not isinstance(queries, list):
                queries = [f"companies in {market_topic} market"]  # Fallback
        except:
            queries = [f"companies in {market_topic} market"]  # Fallback

        print(f"✅ Generated {len(queries)} company discovery queries:")
        for i, query in enumerate(queries, 1):
            print(f"   {i}. {query}")
        print("=" * 60)

        logger.info(f"Generated {len(queries)} company discovery queries")

        return {
            "company_discovery_queries": queries,
            "messages": [
                AIMessage(
                    content=f"Company discovery plan created with {len(queries)} search queries."
                )
            ],
            "current_step": "company_search",
        }

    async def _validate_standalone_questions(
        self, questions: List[str], original_query: str
    ) -> List[str]:
        """Validate and improve questions to ensure they are truly standalone"""

        validation_prompt = f"""
        Original Query: {original_query}
        
        Review the following research questions and ensure each one is COMPLETELY STANDALONE and SELF-CONTAINED.
        
        Questions to validate:
        {json.dumps(questions, indent=2)}
        
        For each question, check if it:
        1. Makes complete sense when read in isolation (without the original query)
        2. Contains all necessary context, keywords, and entities
        3. Avoids pronouns and vague references
        4. Is specific enough for effective web searching
        
        If any question is NOT standalone, rewrite it to be fully self-contained while maintaining its research intent.
        
        Return the validated/improved questions as a JSON list.
        
        Validated Questions:
        """

        try:
            response = await self.llm.ainvoke([HumanMessage(content=validation_prompt)])
            validated = json.loads(response.content.strip())
            if isinstance(validated, list) and len(validated) > 0:
                return validated
        except Exception as e:
            logger.warning(f"Question validation failed: {e}")

        # Fallback: return original questions
        return questions


class TavilyCompanyDiscoveryAgent:
    def __init__(self, retriever: TavilyRetriever, llm: ChatOpenAI):
        self.retriever = retriever
        self.llm = llm

    async def discover_companies(self, state: CompanyResearchState) -> Dict[str, Any]:
        print("\n🔍 TAVILY COMPANY DISCOVERY: Finding companies...")
        print("=" * 60)

        market_topic = state["market_topic"]

        # Create company discovery query
        company_query = (
            f"companies startups businesses operating in {market_topic} market space"
        )

        print(f"   Searching for companies: {company_query}")
        results = await self.retriever.search(company_query, max_results=10)
        print(f"   ✓ Found {len(results)} company sources")

        # If Tavily returns no results, log warning but continue (Firecrawl can still provide results)
        if len(results) == 0:
            logger.warning(
                "Tavily returned no results - continuing with Firecrawl only"
            )
            print("   ⚠️  Tavily returned no results - will rely on Firecrawl")

        print(f"✅ Tavily company discovery complete: {len(results)} sources")
        print("=" * 60)

        logger.info(f"Tavily company discovery completed with {len(results)} sources")

        return {
            "company_discovery_tavily": results,
            "messages": [
                AIMessage(content=f"Tavily found {len(results)} company sources.")
            ],
        }


class FirecrawlCompanyDiscoveryAgent:
    def __init__(self, retriever: FirecrawlRetriever, llm: ChatOpenAI):
        self.retriever = retriever
        self.llm = llm

    async def discover_companies(self, state: CompanyResearchState) -> Dict[str, Any]:
        print("\n🕷️  FIRECRAWL COMPANY DISCOVERY: Finding companies...")
        print("=" * 60)

        market_topic = state["market_topic"]

        # Create company discovery query
        company_query = (
            f"companies startups businesses operating in {market_topic} market space"
        )

        print(f"   Searching for companies: {company_query}")
        results = await self.retriever.search(company_query, max_results=10)
        print(f"   ✓ Found {len(results)} company sources")

        # If Firecrawl fails completely, raise an error
        if len(results) == 0:
            logger.error("Firecrawl returned no results - API is not functioning")
            print("   ❌ Firecrawl API failed - cannot continue")
            raise Exception("Firecrawl search API failed to return results")

        print(f"✅ Firecrawl company discovery complete: {len(results)} sources")
        print("=" * 60)

        logger.info(
            f"Firecrawl company discovery completed with {len(results)} sources"
        )

        return {
            "company_discovery_firecrawl": results,
            "messages": [
                AIMessage(content=f"Firecrawl found {len(results)} company sources.")
            ],
        }


class CompanyListSynthesizer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def _deduplicate_sources(self, sources: List[SearchResult]) -> List[SearchResult]:
        seen_urls = set()
        unique_sources = []

        for source in sources:
            if source.url not in seen_urls and source.url:
                seen_urls.add(source.url)
                unique_sources.append(source)

        return unique_sources

    async def synthesize_company_list(
        self, state: CompanyResearchState
    ) -> Dict[str, Any]:
        print("\n🧠 COMPANY LIST SYNTHESIZER: Creating company list...")
        print("=" * 60)

        tavily_results = state.get("company_discovery_tavily", [])
        firecrawl_results = state.get("company_discovery_firecrawl", [])
        market_topic = state["market_topic"]

        print(f"   📊 Processing {len(tavily_results)} Tavily sources")
        print(f"   📊 Processing {len(firecrawl_results)} Firecrawl sources")

        # Combine and deduplicate sources
        all_sources = tavily_results + firecrawl_results
        unique_sources = self._deduplicate_sources(all_sources)

        print(f"   🔄 Deduplicated to {len(unique_sources)} unique sources")
        print("   📝 Extracting companies...")
        logger.info(f"Starting company extraction from {len(unique_sources)} sources")

        # Extract companies from sources
        try:
            companies = await self._extract_companies(market_topic, unique_sources)
            print(f"   ✅ Extracted {len(companies)} companies")
        except Exception as e:
            logger.error(f"Company extraction failed: {e}")
            companies = []

        # Error if no companies found - core function must work
        if not companies:
            logger.error("No companies extracted from search results")
            raise Exception(
                "Failed to extract companies from research data - core function failure"
            )

        print(f"✅ Company list synthesis complete! Found {len(companies)} companies")
        print("=" * 60)

        logger.info(f"Company synthesis completed with {len(companies)} companies")

        return {
            "companies_list": companies,
            "current_step": "user_review",
            "awaiting_user_input": True,
            "tavily_source_count": len(tavily_results),
            "firecrawl_source_count": len(firecrawl_results),
            "total_sources": len(unique_sources),
            "messages": [
                AIMessage(
                    content=f"Found {len(companies)} companies in {market_topic} market."
                )
            ],
        }

    async def _extract_companies(
        self, market_topic: str, sources: List[SearchResult]
    ) -> List[Company]:
        """Extract company information from search results with optimized processing and structured output"""

        # Optimized content limits to reduce processing time
        MAX_CONTENT_LENGTH = 4000
        MAX_SNIPPET_LENGTH = 300
        content_chunks = []
        total_length = 0

        print(
            f"📊 Processing {len(sources)} sources for company extraction...",
            flush=True,
        )

        for source in sources[:8]:
            # Use best available content: full content > snippet > empty
            if source.content and source.content.strip():
                # Use full content but truncate if too long
                content_text = (
                    source.content[: MAX_SNIPPET_LENGTH * 3]
                    if len(source.content) > MAX_SNIPPET_LENGTH * 3
                    else source.content
                )
            elif source.snippet and source.snippet.strip():
                # Use snippet (which contains Tavily's enhanced content) - allow more length
                content_text = (
                    source.snippet[: MAX_SNIPPET_LENGTH * 2]
                    if len(source.snippet) > MAX_SNIPPET_LENGTH * 2
                    else source.snippet
                )
            else:
                content_text = ""

            chunk = f"Source: {source.title}\nURL: {source.url}\nContent: {content_text}\n\n"

            if total_length + len(chunk) > MAX_CONTENT_LENGTH:
                break

            content_chunks.append(chunk)
            total_length += len(chunk)

        combined_content = "".join(content_chunks)

        if not combined_content.strip():
            print("❌ No content available for company extraction", flush=True)
            logger.warning("No content available for company extraction")
            return []

        print(
            f"📝 Content prepared: {len(content_chunks)} chunks, {total_length} characters",
            flush=True,
        )

        # Optimized prompt for structured output
        prompt = f"""
        Extract companies from this {market_topic} market data:

        {combined_content}

        Find real companies operating in this market space. Focus on actual companies mentioned in the sources.
        Provide their names, brief descriptions, and why they're relevant to this market.
        Maximum 8 companies.
        """

        try:
            print(
                "🤖 Calling LLM for company extraction with structured output...",
                flush=True,
            )
            logger.info(
                f"Extracting companies from {len(content_chunks)} sources, {total_length} characters"
            )

            # Use structured output with Pydantic model
            llm_with_structured_output = self.llm.with_structured_output(
                CompanyExtraction
            )

            response = await asyncio.wait_for(
                llm_with_structured_output.ainvoke([HumanMessage(content=prompt)]),
                timeout=45.0,
            )

            print(
                f"✅ Structured company extraction successful: {len(response.companies)} companies",
                flush=True,
            )

            # Convert Pydantic models to dataclass objects
            companies = []
            for company_info in response.companies:
                company = Company(
                    name=company_info.name,
                    description=company_info.description,
                    reasoning=company_info.reasoning,
                )
                companies.append(company)
                print(f"  📋 Extracted: {company_info.name}", flush=True)

            logger.info(f"Successfully extracted {len(companies)} companies")
            return companies

        except asyncio.TimeoutError:
            print("⏰ Company extraction timed out after 45 seconds", flush=True)
            logger.error("Company extraction timed out after 45 seconds")
            return []
        except Exception as e:
            print(f"❌ Structured company extraction failed: {e}", flush=True)
            logger.error(f"Error extracting companies: {e}")

            # Fallback to original JSON parsing method
            print("🔄 Falling back to JSON parsing method...", flush=True)
            try:
                response = await asyncio.wait_for(
                    self.llm.ainvoke(
                        [
                            HumanMessage(
                                content=prompt
                                + "\n\nReturn as JSON array with fields: name, description, reasoning"
                            )
                        ]
                    ),
                    timeout=30.0,
                )

                # Clean and parse JSON response
                response_content = response.content.strip()
                if response_content.startswith("```json"):
                    response_content = response_content[7:]
                if response_content.endswith("```"):
                    response_content = response_content[:-3]

                companies_data = json.loads(response_content.strip())

                if not isinstance(companies_data, list):
                    logger.error("Response is not a list format")
                    return []

                companies = []
                for item in companies_data:
                    if isinstance(item, dict) and item.get("name"):
                        company = Company(
                            name=item.get("name", ""),
                            description=item.get("description", ""),
                            reasoning=item.get("reasoning", ""),
                        )
                        companies.append(company)

                print(
                    f"✅ Fallback extraction successful: {len(companies)} companies",
                    flush=True,
                )
                logger.info(
                    f"Fallback: Successfully extracted {len(companies)} companies"
                )
                return companies

            except Exception as fallback_error:
                print(
                    f"❌ Fallback extraction also failed: {fallback_error}", flush=True
                )
                logger.error(f"Fallback extraction error: {fallback_error}")
                return []


class CompanyInfoGatheringAgent:
    def __init__(
        self,
        tavily_retriever: TavilyRetriever,
        firecrawl_retriever: FirecrawlRetriever,
        llm: ChatOpenAI,
    ):
        self.tavily_retriever = tavily_retriever
        self.firecrawl_retriever = firecrawl_retriever
        self.llm = llm

    async def gather_company_info(self, state: CompanyResearchState) -> Dict[str, Any]:
        print("\n🔍 COMPANY INFO GATHERING: Researching company details...")
        print("=" * 60)

        companies = state.get(
            "user_modified_companies", state.get("companies_list", [])
        )
        detailed_info = {}

        # Process companies in parallel batches to avoid timeouts
        BATCH_SIZE = 3  # Process 3 companies at a time

        async def research_company(company):
            try:
                print(f"   Researching: {company.name}")

                # Create optimized search query
                query = f"{company.name} company business"

                # Use asyncio.gather for parallel API calls with timeout
                try:
                    tavily_task = self.tavily_retriever.search(
                        query, max_results=5
                    )  # Reduced from 10
                    firecrawl_task = self.firecrawl_retriever.search(
                        query, max_results=5
                    )  # Reduced from 10

                    # Add timeout to prevent hanging
                    tavily_results, firecrawl_results = await asyncio.wait_for(
                        asyncio.gather(
                            tavily_task, firecrawl_task, return_exceptions=True
                        ),
                        timeout=45.0,  # 45 second timeout per company
                    )

                    # Handle exceptions
                    if isinstance(tavily_results, Exception):
                        logger.warning(
                            f"Tavily search failed for {company.name}: {tavily_results}"
                        )
                        tavily_results = []
                    if isinstance(firecrawl_results, Exception):
                        logger.warning(
                            f"Firecrawl search failed for {company.name}: {firecrawl_results}"
                        )
                        firecrawl_results = []

                except asyncio.TimeoutError:
                    logger.warning(f"Search timeout for {company.name}, using fallback")
                    tavily_results, firecrawl_results = [], []

                all_results = tavily_results + firecrawl_results

                print(f"     ✓ Found {len(all_results)} sources for {company.name}")
                return company.name, all_results

            except Exception as e:
                logger.error(f"Error researching {company.name}: {e}")
                return company.name, []

        # Process companies in batches
        for i in range(0, len(companies), BATCH_SIZE):
            batch = companies[i : i + BATCH_SIZE]
            print(
                f"   Processing batch {i//BATCH_SIZE + 1}/{(len(companies) + BATCH_SIZE - 1)//BATCH_SIZE}"
            )

            try:
                # Process batch with timeout
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*[research_company(company) for company in batch]),
                    timeout=120.0,  # 2 minutes per batch
                )

                # Store results
                for company_name, results in batch_results:
                    detailed_info[company_name] = results

            except asyncio.TimeoutError:
                logger.error(
                    f"Batch {i//BATCH_SIZE + 1} timed out, processing individually"
                )

                # Fallback: process individually with shorter timeout
                for company in batch:
                    try:
                        company_name, results = await asyncio.wait_for(
                            research_company(company), timeout=30.0
                        )
                        detailed_info[company_name] = results
                    except asyncio.TimeoutError:
                        logger.warning(f"Individual timeout for {company.name}")
                        detailed_info[company.name] = []

        print(f"✅ Company info gathering complete for {len(companies)} companies")
        print("=" * 60)

        return {
            "detailed_company_info": detailed_info,
            "current_step": "final_synthesis",
            "messages": [
                AIMessage(
                    content=f"Gathered detailed information for {len(companies)} companies."
                )
            ],
        }


class FinalCompanySynthesizer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def create_company_pages(self, state: CompanyResearchState) -> Dict[str, Any]:
        print("\n📄 FINAL SYNTHESIS: Creating company pages...")
        print("=" * 60)

        companies = state.get(
            "user_modified_companies", state.get("companies_list", [])
        )
        detailed_info = state["detailed_company_info"]
        final_pages = {}

        # Process company pages in parallel batches
        BATCH_SIZE = 4  # Process 4 company pages at a time

        async def create_single_page(company):
            try:
                print(f"   Creating page: {company.name}")
                company_sources = detailed_info.get(company.name, [])

                # Add timeout for page creation
                page_content = await asyncio.wait_for(
                    self._create_company_page(company, company_sources),
                    timeout=45.0,  # 45 seconds per page
                )

                print(
                    f"     ✓ Page created for {company.name} with {len(company_sources)} sources"
                )
                return company.name, page_content

            except asyncio.TimeoutError:
                logger.warning(f"Page creation timeout for {company.name}")
                return (
                    company.name,
                    f"# {company.name}\n\nPage creation timed out. Basic info: {company.description}",
                )
            except Exception as e:
                logger.error(f"Error creating page for {company.name}: {e}")
                return (
                    company.name,
                    f"# {company.name}\n\nError creating page: {company.description}",
                )

        # Process in batches to avoid overwhelming the system
        for i in range(0, len(companies), BATCH_SIZE):
            batch = companies[i : i + BATCH_SIZE]
            print(
                f"   Processing page batch {i//BATCH_SIZE + 1}/{(len(companies) + BATCH_SIZE - 1)//BATCH_SIZE}"
            )

            try:
                # Process batch with timeout
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*[create_single_page(company) for company in batch]),
                    timeout=120.0,  # 2 minutes per batch
                )

                # Store results
                for company_name, page_content in batch_results:
                    final_pages[company_name] = page_content

            except asyncio.TimeoutError:
                logger.error(
                    f"Page batch {i//BATCH_SIZE + 1} timed out, processing individually"
                )

                # Fallback: process individually
                for company in batch:
                    try:
                        company_name, page_content = await asyncio.wait_for(
                            create_single_page(company), timeout=30.0
                        )
                        final_pages[company_name] = page_content
                    except asyncio.TimeoutError:
                        logger.warning(f"Individual page timeout for {company.name}")
                        final_pages[company.name] = (
                            f"# {company.name}\n\nTimeout creating detailed page."
                        )

        print(f"✅ Final synthesis complete! Created {len(final_pages)} company pages")
        print("=" * 60)

        return {
            "final_company_pages": final_pages,
            "current_step": "complete",
            "awaiting_user_input": False,
            "messages": [
                AIMessage(
                    content=f"Created detailed pages for {len(companies)} companies."
                )
            ],
        }

    async def _create_company_page(
        self, company: Company, sources: List[SearchResult]
    ) -> str:
        """Create a comprehensive page for a single company with structured output"""

        # Limit content to prevent timeouts
        MAX_SOURCES = 5
        MAX_CONTENT_LENGTH = 2000

        content_chunks = []
        total_length = 0

        print(
            f"📄 Creating profile for {company.name} with {len(sources)} sources...",
            flush=True,
        )

        for source in sources[:MAX_SOURCES]:
            # Use best available content: full content > snippet > empty
            if source.content and source.content.strip():
                # Use full content but limit length for processing efficiency
                content_text = (
                    source.content[:800]
                    if len(source.content) > 800
                    else source.content
                )
            elif source.snippet and source.snippet.strip():
                # Use snippet (which contains Tavily's enhanced content) - allow more length
                content_text = (
                    source.snippet[:600]
                    if len(source.snippet) > 600
                    else source.snippet
                )
            else:
                content_text = ""

            chunk = f"Source: {source.title}\nContent: {content_text}\n\n"

            if total_length + len(chunk) > MAX_CONTENT_LENGTH:
                break

            content_chunks.append(chunk)
            total_length += len(chunk)

        combined_content = "".join(content_chunks)

        # Simplified prompt for structured output
        prompt = f"""
        Create a company profile for "{company.name}":

        Basic Info: {company.description}
        Market Role: {company.reasoning}

        Research Data:
        {combined_content}

        Available Sources:
        {chr(10).join([f"- {source.title} ({source.url})" for source in sources[:10]])}

        Create a comprehensive profile with proper markdown formatting. Include the most relevant sources that were used to generate each section of the profile.
        """

        try:
            print(f"🤖 Generating structured profile for {company.name}...", flush=True)

            # Use structured output with Pydantic model
            llm_with_structured_output = self.llm.with_structured_output(CompanyProfile)

            response = await asyncio.wait_for(
                llm_with_structured_output.ainvoke([HumanMessage(content=prompt)]),
                timeout=35.0,
            )

            # Convert structured response to markdown
            sources_section = ""
            if response.sources:
                sources_section = "\n\n## Sources\n" + "\n".join(
                    [
                        f"- [{source.title}]({source.url})"
                        for source in response.sources[:10]
                    ]
                )

            markdown_content = f"""# {response.company_name}

## Overview
{response.overview}

## Status
{response.status}

## Market Position
{response.market_position}

## Key Facts
{response.key_facts}{sources_section}

---
*Research conducted on {datetime.now().strftime('%Y-%m-%d')}*"""

            print(f"✅ Structured profile created for {company.name}", flush=True)
            return markdown_content

        except asyncio.TimeoutError:
            print(f"⏰ Profile creation timeout for {company.name}", flush=True)
            logger.warning(f"Page creation timeout for {company.name}")
            return f"# {company.name}\n\n## Overview\n{company.description}\n\n## Market Role\n{company.reasoning}\n\n*Detailed analysis timed out*"
        except Exception as e:
            print(
                f"❌ Structured profile creation failed for {company.name}: {e}",
                flush=True,
            )
            logger.error(f"Error creating company page for {company.name}: {e}")

            # Fallback to simple text generation
            print(
                f"🔄 Falling back to simple text generation for {company.name}...",
                flush=True,
            )
            try:
                simple_prompt = f"""
                Create a markdown company profile for "{company.name}":
                
                Description: {company.description}
                Market Role: {company.reasoning}
                
                Use markdown headers and keep it concise.
                """

                response = await asyncio.wait_for(
                    self.llm.ainvoke([HumanMessage(content=simple_prompt)]),
                    timeout=20.0,
                )
                print(f"✅ Fallback profile created for {company.name}", flush=True)
                return response.content.strip()
            except Exception as fallback_error:
                print(
                    f"❌ Fallback profile creation also failed for {company.name}: {fallback_error}",
                    flush=True,
                )
                return f"# {company.name}\n\n## Overview\n{company.description}\n\n## Market Role\n{company.reasoning}"

    async def _generate_summary(self, query: str, sources: List[SearchResult]) -> str:
        if not sources:
            return "No sources found for the given query."

        content_chunks = []
        for source in sources[:10]:  # Limit for token management
            chunk = f"Source: {source.title}\nURL: {source.url}\nContent: {source.snippet}\n\n"
            content_chunks.append(chunk)

        combined_content = "".join(content_chunks)

        prompt = f"""
        Based on the following research sources about "{query}", provide a summary of the key findings:

        {combined_content}

        Please provide a well-structured company profile that:
        1. Identifies the main themes and findings
        2. Highlights any consensus or disagreements across sources
        3. Notes any gaps or limitations in the available information
        4. Organizes information in a logical, coherent manner

        Company Profile:
        """

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error generating summary of research findings."

    async def _generate_report(
        self, query: str, sources: List[SearchResult], summary: str
    ) -> str:
        if not sources:
            return "No research data available to generate report."

        # Generate citations
        citations = []
        for i, source in enumerate(sources, 1):
            citation = f"[{i}] {source.title} - {source.url}"
            citations.append(citation)

        prompt = f"""
        Write a comprehensive research report about "{query}" based on the following research data:

        RESEARCH SUMMARY:
        {summary}

        DETAILED SOURCES:
        """

        for i, source in enumerate(sources, 1):
            prompt += f"\n[{i}] {source.title}\nURL: {source.url}\nContent: {source.snippet}\n"

        prompt += f"""

        Please write a detailed research report that:
        1. Provides a clear introduction to the topic
        2. Organizes findings into logical sections with headers
        3. Includes specific details and evidence from the sources
        4. Uses proper citations in the format [number] referencing the sources above
        5. Concludes with key takeaways and implications
        6. Formats the output in markdown

        The report should be comprehensive, well-structured, and professionally written.

        RESEARCH REPORT:
        """

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            report = response.content.strip()

            # Add citations section
            citations_section = "\n\n## References\n\n" + "\n".join(citations)
            report += citations_section

            return report
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return "Error generating research report."


class CompanyResearcher:
    def __init__(
        self,
        openai_api_key: str,
        tavily_api_key: str,
        firecrawl_api_key: str,
        model: str = "gpt-4o",
    ):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model=model,
            temperature=0,
            request_timeout=30.0,  # Reduced from 60 seconds
            max_retries=1,  # Reduced from 2 retries
        )

        # Initialize retrievers
        self.tavily_retriever = TavilyRetriever(tavily_api_key)
        self.firecrawl_retriever = FirecrawlRetriever(firecrawl_api_key)

        # Initialize agents
        self.user_input_agent = UserInputAgent(self.llm)
        self.tavily_discovery_agent = TavilyCompanyDiscoveryAgent(
            self.tavily_retriever, self.llm
        )
        self.firecrawl_discovery_agent = FirecrawlCompanyDiscoveryAgent(
            self.firecrawl_retriever, self.llm
        )
        self.company_synthesizer = CompanyListSynthesizer(self.llm)
        self.company_info_agent = CompanyInfoGatheringAgent(
            self.tavily_retriever, self.firecrawl_retriever, self.llm
        )
        self.final_synthesizer = FinalCompanySynthesizer(self.llm)

        # Build the graph
        self.workflow = self._build_workflow()

        # Add checkpoint capability
        self.checkpoint_file = None

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(CompanyResearchState)

        # Add nodes
        workflow.add_node("user_input", self.user_input_agent.process_user_input)
        workflow.add_node(
            "tavily_discovery", self.tavily_discovery_agent.discover_companies
        )
        workflow.add_node(
            "firecrawl_discovery", self.firecrawl_discovery_agent.discover_companies
        )
        workflow.add_node(
            "company_synthesis", self.company_synthesizer.synthesize_company_list
        )
        workflow.add_node(
            "company_info_gathering", self.company_info_agent.gather_company_info
        )
        workflow.add_node(
            "final_synthesis", self.final_synthesizer.create_company_pages
        )

        # Define the flow
        workflow.set_entry_point("user_input")
        workflow.add_edge("user_input", "tavily_discovery")
        workflow.add_edge("user_input", "firecrawl_discovery")
        workflow.add_edge(
            ["tavily_discovery", "firecrawl_discovery"], "company_synthesis"
        )

        # Add conditional edge for user review step
        def should_proceed_to_info_gathering(state):
            return (
                "company_info_gathering"
                if not state.get("awaiting_user_input", False)
                else END
            )

        workflow.add_conditional_edges(
            "company_synthesis", should_proceed_to_info_gathering
        )
        workflow.add_edge("company_info_gathering", "final_synthesis")
        workflow.add_edge("final_synthesis", END)

        return workflow.compile()

    async def conduct_company_research(self, query: str) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print("🚀 COMPANY RESEARCH WORKFLOW STARTING")
        print("=" * 80)
        print(f"📋 Query: {query}")
        print(
            "🏗️  Workflow: User Input → [Tavily & Firecrawl Discovery] → Company List → User Review → Info Gathering → Final Pages"
        )
        print("=" * 80)

        logger.info(f"Starting company research for query: {query}")

        initial_state: CompanyResearchState = {
            "messages": [HumanMessage(content=f"Research query: {query}")],
            "original_query": query,
            "market_topic": "",
            "company_discovery_tavily": [],
            "company_discovery_firecrawl": [],
            "companies_list": [],
            "user_modified_companies": [],
            "detailed_company_info": {},
            "final_company_pages": {},
            "current_step": "user_input",
            "awaiting_user_input": False,
        }

        # Run initial discovery workflow with checkpoint
        try:
            discovery_state = await asyncio.wait_for(
                self.workflow.ainvoke(initial_state),
                timeout=240.0,  # 4 minutes for discovery phase
            )

            # Save checkpoint after discovery
            checkpoint_file = self.save_checkpoint(
                discovery_state, "discovery_complete"
            )
            discovery_state["checkpoint_file"] = checkpoint_file

            print("\n" + "=" * 80)
            print("🎉 COMPANY DISCOVERY COMPLETED")
            print("=" * 80)

            logger.info("Company discovery completed")
            return discovery_state

        except asyncio.TimeoutError:
            logger.error("Company discovery phase timed out after 4 minutes")
            raise Exception(
                "Discovery phase timeout - please retry with a more specific query"
            )

    async def continue_with_company_info(
        self, state: Dict[str, Any], user_modified_companies: List[Company] = None
    ) -> Dict[str, Any]:
        """Continue workflow with user-modified company list"""
        print("\n" + "=" * 80)
        print("🔄 CONTINUING WITH COMPANY INFO GATHERING")
        print("=" * 80)

        if user_modified_companies:
            state["user_modified_companies"] = user_modified_companies

        state["awaiting_user_input"] = False
        state["current_step"] = "company_info_gathering"

        # Create a new workflow that starts from company info gathering
        info_workflow = StateGraph(CompanyResearchState)
        info_workflow.add_node(
            "company_info_gathering", self.company_info_agent.gather_company_info
        )
        info_workflow.add_node(
            "final_synthesis", self.final_synthesizer.create_company_pages
        )

        info_workflow.set_entry_point("company_info_gathering")
        info_workflow.add_edge("company_info_gathering", "final_synthesis")
        info_workflow.add_edge("final_synthesis", END)

        compiled_workflow = info_workflow.compile()

        try:
            # Add timeout for the final processing phase
            final_state = await asyncio.wait_for(
                compiled_workflow.ainvoke(state),
                timeout=240.0,  # 4 minutes for info gathering and synthesis
            )

            # Save final checkpoint
            self.save_checkpoint(final_state, "research_complete")

            print("\n" + "=" * 80)
            print("🎉 COMPLETE COMPANY RESEARCH WORKFLOW FINISHED")
            print("=" * 80)

            logger.info("Complete company research workflow completed")
            return final_state

        except asyncio.TimeoutError:
            logger.error("Company info gathering phase timed out after 4 minutes")
            # Save partial results checkpoint
            self.save_checkpoint(state, "partial_results")
            raise Exception(
                "Info gathering timeout - partial results saved to checkpoint"
            )

    def save_company_research(self, state: Dict[str, Any], filename: str) -> None:
        """Save company research results (disabled - files only available through UI download)"""
        # File saving is disabled - research results are only available through UI download
        logger.info(
            f"File saving disabled - research results available only through UI download: {filename}"
        )
        return

    def save_checkpoint(self, state: Dict[str, Any], step: str) -> str:
        """Save checkpoint for resume capability (disabled - no file saving)"""
        # Checkpoint saving is disabled - no files are saved to repository
        logger.info(
            f"Checkpoint saving disabled - no files saved to repository: {step}"
        )
        return f"checkpoint_{step}_disabled"

    def load_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """Load checkpoint to resume processing"""
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)

            logger.info(
                f"Loaded checkpoint from {checkpoint_data['step']} at {checkpoint_data['timestamp']}"
            )
            return self._deserialize_state(checkpoint_data["state"])

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return None

    def _serialize_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize state for checkpoint storage"""
        serialized = state.copy()

        # Convert Company objects to dictionaries
        if "companies_list" in serialized:
            serialized["companies_list"] = [
                company.__dict__ if hasattr(company, "__dict__") else company
                for company in serialized["companies_list"]
            ]

        if "user_modified_companies" in serialized:
            serialized["user_modified_companies"] = [
                company.__dict__ if hasattr(company, "__dict__") else company
                for company in serialized["user_modified_companies"]
            ]

        # Convert SearchResult objects to dictionaries
        for key in ["company_discovery_tavily", "company_discovery_firecrawl"]:
            if key in serialized:
                serialized[key] = [
                    result.__dict__ if hasattr(result, "__dict__") else result
                    for result in serialized[key]
                ]

        # Handle detailed_company_info
        if "detailed_company_info" in serialized:
            for company_name, results in serialized["detailed_company_info"].items():
                serialized["detailed_company_info"][company_name] = [
                    result.__dict__ if hasattr(result, "__dict__") else result
                    for result in results
                ]

        # Handle messages (convert to string representations to avoid JSON serialization issues)
        if "messages" in serialized:
            serialized["messages"] = [
                {
                    "type": msg.__class__.__name__,
                    "content": (
                        str(msg.content) if hasattr(msg, "content") else str(msg)
                    ),
                }
                for msg in serialized["messages"]
            ]

        return serialized

    def _deserialize_state(self, serialized_state: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize state from checkpoint"""
        state = serialized_state.copy()

        # Convert dictionaries back to Company objects
        if "companies_list" in state:
            state["companies_list"] = [
                Company(**company) if isinstance(company, dict) else company
                for company in state["companies_list"]
            ]

        if "user_modified_companies" in state:
            state["user_modified_companies"] = [
                Company(**company) if isinstance(company, dict) else company
                for company in state["user_modified_companies"]
            ]

        # Convert dictionaries back to SearchResult objects
        for key in ["company_discovery_tavily", "company_discovery_firecrawl"]:
            if key in state:
                state[key] = [
                    SearchResult(**result) if isinstance(result, dict) else result
                    for result in state[key]
                ]

        # Handle detailed_company_info
        if "detailed_company_info" in state:
            for company_name, results in state["detailed_company_info"].items():
                state["detailed_company_info"][company_name] = [
                    SearchResult(**result) if isinstance(result, dict) else result
                    for result in results
                ]

        return state


async def main():
    import sys
    import json

    print("🚀 Starting AI Research System", flush=True)
    print(f"⚙️  Python version: {sys.version}", flush=True)
    print(f"📋 Command line arguments: {sys.argv}", flush=True)

    # Import and use config
    from config import Config

    print("🔧 Loading configuration...", flush=True)
    # Validate configuration
    config_status = Config.validate_config()
    if not config_status["valid"]:
        print(
            f"❌ Configuration validation failed: {config_status['issues']}", flush=True
        )
        logger.error(f"Configuration issues: {config_status['issues']}")
        raise Exception(f"Configuration validation failed: {config_status['issues']}")

    print("✅ Configuration validated successfully", flush=True)
    print(
        f"🔑 API Keys configured: OpenAI={bool(Config.OPENAI_API_KEY)}, Tavily={bool(Config.TAVILY_API_KEY)}, Firecrawl={bool(Config.FIRECRAWL_API_KEY)}",
        flush=True,
    )

    # Create Company researcher
    print("🏗️  Initializing Company Researcher...", flush=True)
    researcher = CompanyResearcher(
        openai_api_key=Config.OPENAI_API_KEY,
        tavily_api_key=Config.TAVILY_API_KEY,
        firecrawl_api_key=Config.FIRECRAWL_API_KEY,
        model="gpt-4o",
    )
    print("✅ Company Researcher initialized", flush=True)

    # Check command line arguments
    print(f"🔍 Parsing command line arguments (total: {len(sys.argv)})", flush=True)
    if len(sys.argv) < 2:
        print("❌ Insufficient arguments provided", flush=True)
        raise Exception(
            "Usage: python research_system.py <query> or python research_system.py --company-info <data>"
        )

    mode = sys.argv[1]
    print(f"🎯 Detected mode: {mode}", flush=True)

    if mode == "--company-info":
        print("🏢 Starting Company Info Research Mode", flush=True)
        if len(sys.argv) < 3:
            print("❌ Company info mode requires data argument", flush=True)
            raise Exception("Company info mode requires data argument")

        # Handle company info research
        try:
            print("📊 Parsing company info data...", flush=True)
            data = json.loads(sys.argv[2])
            state = data.get("state")
            user_companies = data.get("user_companies")

            print(
                f"📋 Received state keys: {list(state.keys()) if state else 'None'}",
                flush=True,
            )
            print(
                f"🏭 Number of companies to research: {len(user_companies) if user_companies else 0}",
                flush=True,
            )

            if not state or not user_companies:
                print("❌ Invalid company info data format", flush=True)
                raise Exception("Invalid company info data format")

            # Convert to Company objects if needed
            companies = []
            print("🔄 Converting companies to internal format...", flush=True)
            for i, comp in enumerate(user_companies):
                if isinstance(comp, dict):
                    company = Company(
                        name=comp.get("name", ""),
                        description=comp.get("description", ""),
                        reasoning=comp.get("reasoning", ""),
                    )
                    companies.append(company)
                    print(f"  ✅ Company {i+1}: {company.name}", flush=True)
                else:
                    companies.append(comp)
                    print(
                        f"  ✅ Company {i+1}: {comp.name if hasattr(comp, 'name') else 'Unknown'}",
                        flush=True,
                    )

            # Continue with company info gathering
            print("🏗️  Building research state...", flush=True)
            state_dict = {
                "user_modified_companies": companies,
                "companies_list": companies,
                "market_topic": state.get("market_topic", ""),
                "original_query": state.get("query", ""),
                "messages": [],
                "company_discovery_tavily": state.get("company_discovery_tavily", []),
                "company_discovery_firecrawl": state.get(
                    "company_discovery_firecrawl", []
                ),
                "detailed_company_info": {},
                "final_company_pages": {},
                "current_step": "company_info_gathering",
                "awaiting_user_input": False,
            }

            print(f"🎯 Market topic: {state_dict['market_topic']}", flush=True)
            print(f"🔍 Original query: {state_dict['original_query']}", flush=True)
            print(
                f"📊 Tavily raw data: {len(state_dict['company_discovery_tavily'])} results",
                flush=True,
            )
            print(
                f"🕷️  Firecrawl raw data: {len(state_dict['company_discovery_firecrawl'])} results",
                flush=True,
            )
            print("🚀 Starting detailed company research...", flush=True)

            final_state = await researcher.continue_with_company_info(state_dict)

            print("✅ Company research completed successfully", flush=True)
            print(
                f"📄 Generated pages for {len(final_state.get('final_company_pages', {}))} companies",
                flush=True,
            )

            # Output final result as JSON
            result = {
                "type": "complete",
                "data": {
                    "query": state.get("query"),
                    "market_topic": state.get("market_topic"),
                    "company_pages": final_state.get("final_company_pages", {}),
                    "total_companies": len(companies),
                    "timestamp": datetime.now().isoformat(),
                    "company_discovery_tavily": [
                        res.__dict__
                        for res in final_state.get("company_discovery_tavily", [])
                    ],
                    "company_discovery_firecrawl": [
                        res.__dict__
                        for res in final_state.get("company_discovery_firecrawl", [])
                    ],
                },
            }
            print("📤 Sending final result to UI...", flush=True)
            print(json.dumps(result))

        except Exception as e:
            print(f"❌ Company info research failed: {str(e)}", flush=True)
            error_result = {
                "type": "error",
                "data": {
                    "message": f"Company info research failed: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                },
            }
            print(json.dumps(error_result))
            raise
    else:
        # Handle regular company discovery
        query = mode  # First argument is the query
        print(f"🔍 Starting Company Discovery Mode for query: '{query}'", flush=True)

        try:
            print("🎯 Initiating company research workflow...", flush=True)
            discovery_state = await researcher.conduct_company_research(query)

            print("✅ Company discovery completed successfully", flush=True)

            # Output discovery result as JSON
            companies = discovery_state["companies_list"]
            tavily_count = len(discovery_state.get("company_discovery_tavily", []))
            firecrawl_count = len(
                discovery_state.get("company_discovery_firecrawl", [])
            )

            print(f"📊 Discovery Results Summary:", flush=True)
            print(f"  🏢 Companies found: {len(companies)}", flush=True)
            print(f"  🔍 Tavily sources: {tavily_count}", flush=True)
            print(f"  🕷️  Firecrawl sources: {firecrawl_count}", flush=True)
            print(f"  📋 Market topic: {discovery_state['market_topic']}", flush=True)

            for i, company in enumerate(companies, 1):
                print(
                    f"    {i}. {company.name}: {company.description[:60]}...",
                    flush=True,
                )

            result = {
                "type": "company_discovery",
                "data": {
                    "query": query,
                    "market_topic": discovery_state["market_topic"],
                    "companies": [
                        {
                            "name": company.name,
                            "description": company.description,
                            "reasoning": company.reasoning,
                        }
                        for company in companies
                    ],
                    "total_companies": len(companies),
                    "tavily_source_count": tavily_count,
                    "firecrawl_source_count": firecrawl_count,
                    "total_sources": tavily_count + firecrawl_count,
                    "timestamp": datetime.now().isoformat(),
                    "awaiting_user_input": True,
                    "step": "company_review",
                },
            }
            print("📤 Sending discovery result to UI...", flush=True)
            print(json.dumps(result))

        except Exception as e:
            print(f"❌ Company research failed: {str(e)}", flush=True)
            error_result = {
                "type": "error",
                "data": {
                    "message": f"Company research failed: {str(e)}",
                    "timestamp": datetime.now().isoformat(),
                },
            }
            print(json.dumps(error_result))
            raise


if __name__ == "__main__":
    asyncio.run(main())
