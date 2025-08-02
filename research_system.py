import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union, Annotated, TypedDict
from datetime import datetime
import aiohttp
import openai
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import operator

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
        self.base_url = "https://api.tavily.com"
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
            
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"
        }
        
        # Use minimal payload confirmed to work
        payload = {
            "api_key": self.api_key,
            "query": query,
            "max_results": min(max_results, 5),
        }
        
        # Log for debugging in Vercel (mask API key)
        debug_payload = payload.copy()
        if self.api_key:
            debug_payload["api_key"] = f"{self.api_key[:8]}..." if len(self.api_key) > 8 else "****"
        logger.info(f"Tavily API request to {self.base_url}/search with payload: {debug_payload}")
        logger.info(f"Tavily API key length: {len(self.api_key) if self.api_key else 0}")

        # For Vercel debugging: log the exact environment
        logger.info(f"Running in Vercel: {os.environ.get('VERCEL', 'false')}")
        logger.info(f"API base URL: {self.base_url}")
        
        # Try multiple approaches for better Vercel compatibility
        attempts = [
            # Attempt 1: Minimal payload (most likely to work in Vercel)
            payload,
            # Attempt 2: Add search_depth for robustness
            {**payload, "search_depth": "basic"},
        ]
        
        for attempt_num, attempt_payload in enumerate(attempts, 1):
            try:
                logger.info(f"Tavily API attempt {attempt_num}/{len(attempts)}")
                
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                    async with session.post(
                        f"{self.base_url}/search", headers=headers, json=attempt_payload
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            self.failure_count = 0  # Reset on success
                            logger.info(f"Tavily API success on attempt {attempt_num}")
                            return self._format_tavily_results(data)
                        else:
                            # Get detailed error information
                            try:
                                error_data = await response.text()
                                logger.warning(f"Tavily attempt {attempt_num} failed: {response.status} - {error_data}")
                            except:
                                logger.warning(f"Tavily attempt {attempt_num} failed: {response.status}")
                            
                            # Don't handle as failure yet, try next attempt
                            if attempt_num == len(attempts):
                                self._handle_failure()
                                return []
                                
            except Exception as e:
                logger.warning(f"Tavily attempt {attempt_num} error: {e}")
                if attempt_num == len(attempts):
                    self._handle_failure()
                    return []
        
        return []
    
    def _handle_failure(self):
        """Handle API failures and circuit breaker logic"""
        self.failure_count += 1
        if self.failure_count >= self.max_failures:
            self.circuit_open = True
            logger.warning(f"Tavily circuit breaker opened after {self.failure_count} failures")

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
        return ""


class FirecrawlRetriever(BaseRetriever):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.firecrawl.dev/v0"
        self.failure_count = 0
        self.max_failures = 3
        self.circuit_open = False

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        # Circuit breaker check
        if self.circuit_open:
            logger.warning("Firecrawl circuit breaker is open, skipping search")
            return []
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "query": query,
            "pageOptions": {"onlyMainContent": True, "includeHtml": False},
            "limit": min(max_results, 5),  # Limit results
        }

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20)) as session:
                async with session.post(
                    f"{self.base_url}/search", headers=headers, json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.failure_count = 0  # Reset on success
                        return self._format_firecrawl_results(data)
                    else:
                        logger.error(f"Firecrawl search failed: {response.status}")
                        self._handle_failure()
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
            logger.warning(f"Firecrawl circuit breaker opened after {self.failure_count} failures")

    def _format_firecrawl_results(self, data: Dict) -> List[SearchResult]:
        results = []
        for item in data.get("data", []):
            result = SearchResult(
                title=item.get("metadata", {}).get("title", ""),
                url=item.get("metadata", {}).get("sourceURL", ""),
                snippet=item.get("extract", "")[:500],
                content=item.get("markdown", ""),
                score=1.0,
                source="firecrawl",
            )
            results.append(result)
        return results

    async def get_content(self, url: str) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "url": url,
            "pageOptions": {"onlyMainContent": True, "includeHtml": False},
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/scrape", headers=headers, json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data", {}).get("markdown", "")
                    else:
                        logger.error(f"Firecrawl scrape failed: {response.status}")
                        return ""
        except Exception as e:
            logger.error(f"Firecrawl scrape error: {e}")
            return ""


class UserInputAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def process_user_input(self, state: CompanyResearchState) -> Dict[str, Any]:
        print("\n🎯 USER INPUT AGENT: Processing market/topic query...")
        print("=" * 60)

        query = state["original_query"]

        market_extraction_prompt = f"""
        Extract the market, problem, or topic from this user query: "{query}"
        
        Provide a clear, specific description of the market/problem/topic that companies would be addressing.
        This will be used to search for companies operating in this space.
        
        Return just the market/problem/topic description, nothing else.
        """

        response = await self.llm.ainvoke(
            [HumanMessage(content=market_extraction_prompt)]
        )
        market_topic = response.content.strip()

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

        # If Tavily fails completely, create a fallback message but don't fail the whole workflow
        if len(results) == 0:
            logger.warning("Tavily returned no results - this may indicate API issues")
            print("   ⚠️  Tavily API may be experiencing issues, continuing with Firecrawl only")

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

        # Fallback if no companies found
        if not companies:
            logger.warning("No companies extracted, creating fallback list")
            companies = [
                Company(
                    name="Example Company",
                    description="A company operating in this market space",
                    reasoning="Placeholder company for demonstration purposes",
                )
            ]

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
        """Extract company information from search results with optimized processing"""

        # Optimized content limits to reduce processing time
        MAX_CONTENT_LENGTH = 4000  # Reduced from 8000
        MAX_SNIPPET_LENGTH = 300   # Reduced from 500
        content_chunks = []
        total_length = 0

        for source in sources[:8]:  # Reduced from 10 sources
            # Truncate snippet if too long
            snippet = source.snippet[:MAX_SNIPPET_LENGTH] if source.snippet else ""
            chunk = f"Source: {source.title}\nURL: {source.url}\nContent: {snippet}\n\n"

            if total_length + len(chunk) > MAX_CONTENT_LENGTH:
                break

            content_chunks.append(chunk)
            total_length += len(chunk)

        combined_content = "".join(content_chunks)

        if not combined_content.strip():
            logger.warning("No content available for company extraction")
            return []

        # Optimized prompt for faster processing
        prompt = f"""
        Extract companies from this {market_topic} market data. Return ONLY a JSON array:

        {combined_content}

        Format: [{{
            "name": "Company Name",
            "description": "Brief description",
            "reasoning": "Market relevance"
        }}]

        Max 8 companies. Real companies only.
        """

        try:
            logger.info(
                f"Extracting companies from {len(content_chunks)} sources, {total_length} characters"
            )

            # Reduced timeout for faster failure detection
            response = await asyncio.wait_for(
                self.llm.ainvoke([HumanMessage(content=prompt)]),
                timeout=30.0,  # Reduced from 60 seconds
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

            logger.info(f"Successfully extracted {len(companies)} companies")
            return companies

        except asyncio.TimeoutError:
            logger.error("Company extraction timed out after 30 seconds")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in company extraction: {e}")
            logger.error(
                f"Response content: {response.content[:300] if 'response' in locals() else 'No response'}"
            )
            return []
        except Exception as e:
            logger.error(f"Error extracting companies: {e}")
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
                    tavily_task = self.tavily_retriever.search(query, max_results=5)  # Reduced from 10
                    firecrawl_task = self.firecrawl_retriever.search(query, max_results=5)  # Reduced from 10
                    
                    # Add timeout to prevent hanging
                    tavily_results, firecrawl_results = await asyncio.wait_for(
                        asyncio.gather(tavily_task, firecrawl_task, return_exceptions=True),
                        timeout=45.0  # 45 second timeout per company
                    )
                    
                    # Handle exceptions
                    if isinstance(tavily_results, Exception):
                        logger.warning(f"Tavily search failed for {company.name}: {tavily_results}")
                        tavily_results = []
                    if isinstance(firecrawl_results, Exception):
                        logger.warning(f"Firecrawl search failed for {company.name}: {firecrawl_results}")
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
            batch = companies[i:i + BATCH_SIZE]
            print(f"   Processing batch {i//BATCH_SIZE + 1}/{(len(companies) + BATCH_SIZE - 1)//BATCH_SIZE}")
            
            try:
                # Process batch with timeout
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*[research_company(company) for company in batch]),
                    timeout=120.0  # 2 minutes per batch
                )
                
                # Store results
                for company_name, results in batch_results:
                    detailed_info[company_name] = results
                    
            except asyncio.TimeoutError:
                logger.error(f"Batch {i//BATCH_SIZE + 1} timed out, processing individually")
                
                # Fallback: process individually with shorter timeout
                for company in batch:
                    try:
                        company_name, results = await asyncio.wait_for(
                            research_company(company),
                            timeout=30.0
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
                    timeout=45.0  # 45 seconds per page
                )
                
                print(f"     ✓ Page created for {company.name} with {len(company_sources)} sources")
                return company.name, page_content
                
            except asyncio.TimeoutError:
                logger.warning(f"Page creation timeout for {company.name}")
                return company.name, f"# {company.name}\n\nPage creation timed out. Basic info: {company.description}"
            except Exception as e:
                logger.error(f"Error creating page for {company.name}: {e}")
                return company.name, f"# {company.name}\n\nError creating page: {company.description}"

        # Process in batches to avoid overwhelming the system
        for i in range(0, len(companies), BATCH_SIZE):
            batch = companies[i:i + BATCH_SIZE]
            print(f"   Processing page batch {i//BATCH_SIZE + 1}/{(len(companies) + BATCH_SIZE - 1)//BATCH_SIZE}")
            
            try:
                # Process batch with timeout
                batch_results = await asyncio.wait_for(
                    asyncio.gather(*[create_single_page(company) for company in batch]),
                    timeout=120.0  # 2 minutes per batch
                )
                
                # Store results
                for company_name, page_content in batch_results:
                    final_pages[company_name] = page_content
                    
            except asyncio.TimeoutError:
                logger.error(f"Page batch {i//BATCH_SIZE + 1} timed out, processing individually")
                
                # Fallback: process individually
                for company in batch:
                    try:
                        company_name, page_content = await asyncio.wait_for(
                            create_single_page(company),
                            timeout=30.0
                        )
                        final_pages[company_name] = page_content
                    except asyncio.TimeoutError:
                        logger.warning(f"Individual page timeout for {company.name}")
                        final_pages[company.name] = f"# {company.name}\n\nTimeout creating detailed page."

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
        """Create a comprehensive page for a single company with optimized processing"""

        # Limit content to prevent timeouts
        MAX_SOURCES = 5
        MAX_CONTENT_LENGTH = 2000
        
        content_chunks = []
        total_length = 0
        
        for source in sources[:MAX_SOURCES]:
            snippet = source.snippet[:400] if source.snippet else ""  # Limit snippet length
            chunk = f"Source: {source.title}\nContent: {snippet}\n\n"
            
            if total_length + len(chunk) > MAX_CONTENT_LENGTH:
                break
                
            content_chunks.append(chunk)
            total_length += len(chunk)

        combined_content = "".join(content_chunks)

        # Simplified prompt for faster processing
        prompt = f"""
        Create a company profile for "{company.name}":

        Basic Info: {company.description}
        Market Role: {company.reasoning}

        Research Data:
        {combined_content}

        Include:
        ## Overview
        ## Status  
        ## Market Position
        ## Key Facts

        Keep concise. Use markdown format.
        """

        try:
            # Reduced timeout for faster page creation
            response = await asyncio.wait_for(
                self.llm.ainvoke([HumanMessage(content=prompt)]),
                timeout=25.0  # 25 seconds per page
            )
            return response.content.strip()
        except asyncio.TimeoutError:
            logger.warning(f"Page creation timeout for {company.name}")
            return f"# {company.name}\n\n## Overview\n{company.description}\n\n## Market Role\n{company.reasoning}\n\n*Detailed analysis timed out*"
        except Exception as e:
            logger.error(f"Error creating company page for {company.name}: {e}")
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
            max_retries=1,         # Reduced from 2 retries
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
                timeout=240.0  # 4 minutes for discovery phase
            )
            
            # Save checkpoint after discovery
            checkpoint_file = self.save_checkpoint(discovery_state, "discovery_complete")
            discovery_state["checkpoint_file"] = checkpoint_file

            print("\n" + "=" * 80)
            print("🎉 COMPANY DISCOVERY COMPLETED")
            print("=" * 80)

            logger.info("Company discovery completed")
            return discovery_state
            
        except asyncio.TimeoutError:
            logger.error("Company discovery phase timed out after 4 minutes")
            raise Exception("Discovery phase timeout - please retry with a more specific query")

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
                timeout=240.0  # 4 minutes for info gathering and synthesis
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
            raise Exception("Info gathering timeout - partial results saved to checkpoint")

    def save_company_research(self, state: Dict[str, Any], filename: str) -> None:
        """Save company research results"""
        companies_data = []
        for company in state.get("companies_list", []):
            companies_data.append(
                {
                    "name": company.name,
                    "description": company.description,
                    "reasoning": company.reasoning,
                    "year_established": company.year_established,
                    "still_in_business": company.still_in_business,
                    "history": company.history,
                    "future_roadmap": company.future_roadmap,
                }
            )

        research_data = {
            "query": state["original_query"],
            "market_topic": state["market_topic"],
            "timestamp": datetime.now().isoformat(),
            "companies": companies_data,
            "company_pages": state.get("final_company_pages", {}),
            "total_companies": len(companies_data),
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(research_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Company research saved to {filename}")
    
    def save_checkpoint(self, state: Dict[str, Any], step: str) -> str:
        """Save checkpoint for resume capability"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_filename = f"checkpoint_{step}_{timestamp}.json"
        
        checkpoint_data = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "state": self._serialize_state(state)
        }
        
        with open(checkpoint_filename, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Checkpoint saved: {checkpoint_filename}")
        return checkpoint_filename
    
    def load_checkpoint(self, checkpoint_file: str) -> Dict[str, Any]:
        """Load checkpoint to resume processing"""
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"Loaded checkpoint from {checkpoint_data['step']} at {checkpoint_data['timestamp']}")
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
    # Import and use config
    from config import Config

    # Validate configuration
    config_status = Config.validate_config()
    if not config_status["valid"]:
        logger.error(f"Configuration issues: {config_status['issues']}")
        return

    # Create Company researcher
    researcher = CompanyResearcher(
        openai_api_key=Config.OPENAI_API_KEY,
        tavily_api_key=Config.TAVILY_API_KEY,
        firecrawl_api_key=Config.FIRECRAWL_API_KEY,
        model="gpt-4o",
    )

    # Conduct company research
    query = "ride sharing market like Uber"
    discovery_state = await researcher.conduct_company_research(query)

    # Print company discovery results
    print("=" * 80)
    print("MARKET/TOPIC IDENTIFIED")
    print("=" * 80)
    print(discovery_state["market_topic"])

    print("\n" + "=" * 80)
    print("COMPANIES FOUND")
    print("=" * 80)
    companies = discovery_state["companies_list"]
    for i, company in enumerate(companies, 1):
        print(f"{i}. {company.name}")
        print(f"   Description: {company.description}")
        print(f"   Reasoning: {company.reasoning}")
        print()

    # Simulate user continuing with all companies (in real app, user would modify the list)
    print("=" * 80)
    print("CONTINUING WITH DETAILED COMPANY INFO...")
    print("=" * 80)

    final_state = await researcher.continue_with_company_info(discovery_state)

    # Print final results
    print("\n" + "=" * 80)
    print("COMPANY PAGES CREATED")
    print("=" * 80)
    for company_name, page_content in final_state["final_company_pages"].items():
        print(f"\n--- {company_name} ---")
        print(page_content[:500] + "..." if len(page_content) > 500 else page_content)

    # Save research
    researcher.save_company_research(final_state, "company_research_output.json")


if __name__ == "__main__":
    asyncio.run(main())
