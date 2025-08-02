import asyncio
import json
import logging
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

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        headers = {"Content-Type": "application/json"}
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": True,
            "include_raw_content": True,
            "max_results": max_results,
            "include_domains": [],
            "exclude_domains": [],
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/search", headers=headers, json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_tavily_results(data)
                    else:
                        logger.error(f"Tavily search failed: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return []

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

    async def search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "query": query,
            "pageOptions": {"onlyMainContent": True, "includeHtml": False},
            "limit": max_results,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/search", headers=headers, json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_firecrawl_results(data)
                    else:
                        logger.error(f"Firecrawl search failed: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Firecrawl search error: {e}")
            return []

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

        # Extract companies from sources
        companies = await self._extract_companies(market_topic, unique_sources)

        print(f"✅ Company list synthesis complete! Found {len(companies)} companies")
        print("=" * 60)

        logger.info(f"Company synthesis completed with {len(companies)} companies")

        return {
            "companies_list": companies,
            "current_step": "user_review",
            "awaiting_user_input": True,
            "messages": [
                AIMessage(
                    content=f"Found {len(companies)} companies in {market_topic} market."
                )
            ],
        }

    async def _extract_companies(
        self, market_topic: str, sources: List[SearchResult]
    ) -> List[Company]:
        """Extract company information from search results"""

        content_chunks = []
        for source in sources:
            chunk = f"Source: {source.title}\nURL: {source.url}\nContent: {source.snippet}\n\n"
            content_chunks.append(chunk)

        combined_content = "".join(content_chunks)

        prompt = f"""
        Based on the following search results about companies in "{market_topic}", extract a list of companies that are directly addressing this market/problem.

        {combined_content}

        For each company, provide:
        1. Company name
        2. Brief description of what they do
        3. Reasoning for why they address this market/problem

        Format your response as JSON with this structure:
        [
          {{
            "name": "Company Name",
            "description": "Brief description of the company",
            "reasoning": "Why this company addresses the market/problem"
          }}
        ]

        Focus on real companies mentioned in the sources. Only include companies that clearly operate in or address the specified market/topic.
        """

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            companies_data = json.loads(response.content.strip())

            companies = []
            for item in companies_data:
                company = Company(
                    name=item.get("name", ""),
                    description=item.get("description", ""),
                    reasoning=item.get("reasoning", ""),
                )
                companies.append(company)

            return companies
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

        for i, company in enumerate(companies, 1):
            print(f"   Researching {i}/{len(companies)}: {company.name}")

            # Create detailed search query for this company
            query = f"{company.name} company history establishment year business status roadmap"

            # Parallel searches
            tavily_results = await self.tavily_retriever.search(query, max_results=10)
            firecrawl_results = await self.firecrawl_retriever.search(
                query, max_results=10
            )

            all_results = tavily_results + firecrawl_results
            detailed_info[company.name] = all_results

            print(
                f"     ✓ Found {len(all_results)} sources ({len(tavily_results)} Tavily + {len(firecrawl_results)} Firecrawl)"
            )

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

        for i, company in enumerate(companies, 1):
            print(f"   Creating page {i}/{len(companies)}: {company.name}")

            company_sources = detailed_info.get(company.name, [])
            page_content = await self._create_company_page(company, company_sources)
            final_pages[company.name] = page_content

            print(f"     ✓ Page created with {len(company_sources)} sources")

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
        """Create a comprehensive page for a single company"""

        content_chunks = []
        for source in sources:
            chunk = f"Source: {source.title}\nURL: {source.url}\nContent: {source.snippet}\n\n"
            content_chunks.append(chunk)

        combined_content = "".join(content_chunks)

        prompt = f"""
        Create a comprehensive company profile for "{company.name}" based on the following information:

        INITIAL COMPANY INFO:
        Description: {company.description}
        Market Relevance: {company.reasoning}

        DETAILED RESEARCH SOURCES:
        {combined_content}

        Create a very detailed and comprehensive company profile that includes:
        1. **Company Overview** - What the company does
        2. **Year Established** - When was it founded (if available)
        3. **Business Status** - Is it still in business, acquired, shut down, etc.
        4. **Market Operations** - How it operates in the specified market
        5. **Company History** - Key milestones, evolution, major events
        6. **Future Roadmap** - Plans, vision, direction (if available)
        7. **Market Position** - How it fits in the competitive landscape

        Format the response in markdown with proper headers and structure.
        If information is not available, state "Information not available" for that section.
        Cite sources where relevant using [Source Title - URL] format.

        COMPANY PROFILE:
        """

        try:
            response = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"Error creating company page for {company.name}: {e}")
            return f"# {company.name}\n\nError creating company profile."

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
        self.llm = ChatOpenAI(api_key=openai_api_key, model=model, temperature=0.3)

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

        # Run initial discovery workflow
        discovery_state = await self.workflow.ainvoke(initial_state)

        print("\n" + "=" * 80)
        print("🎉 COMPANY DISCOVERY COMPLETED")
        print("=" * 80)

        logger.info("Company discovery completed")
        return discovery_state

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

        final_state = await compiled_workflow.ainvoke(state)

        print("\n" + "=" * 80)
        print("🎉 COMPLETE COMPANY RESEARCH WORKFLOW FINISHED")
        print("=" * 80)

        logger.info("Complete company research workflow completed")
        return final_state

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
