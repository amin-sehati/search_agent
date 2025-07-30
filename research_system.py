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
class ResearchContext:
    query: str
    sources: List[SearchResult]
    summary: str = ""
    report: str = ""
    citations: List[str] = None

    def __post_init__(self):
        if self.citations is None:
            self.citations = []


class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    original_query: str
    research_questions: List[str]
    tavily_results: List[SearchResult]
    firecrawl_results: List[SearchResult]
    all_sources: List[SearchResult]
    summary: str
    report: str
    next_action: str


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


class PlannerAgent:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    async def plan_research(self, state: ResearchState) -> Dict[str, Any]:
        print("\n🎯 PLANNER AGENT: Breaking down research query...")
        print("=" * 60)

        query = state["original_query"]

        planning_prompt = f"""
        You are a research planner. Your task is to break down the following research query into 3-5 specific, focused questions that can be researched independently.
        
        Original Query: {query}
        
        Please provide 3-5 specific research questions that together will comprehensively address the original query. Each question MUST be:
        1. STANDALONE and SELF-CONTAINED - fully understandable without any additional context
        2. Include ALL necessary keywords, entities, and context from the original query
        3. Specific and focused enough to yield targeted search results
        4. Researchable through web search engines
        5. Complementary to the other questions to provide comprehensive coverage
        
        CRITICAL REQUIREMENTS:
        - Each question must make complete sense on its own when read in isolation
        - Include specific company names, technologies, time periods, or other key terms from the original query
        - Avoid pronouns (it, they, these, etc.) - use specific nouns instead
        - Avoid references to "the above" or "mentioned" concepts
        - Each question should be a complete, standalone research inquiry
        
        Example of GOOD standalone questions:
        - "What are the main reasons why ride-sharing companies like Uber have failed in international markets?"
        - "Which transportation startups similar to Uber shut down between 2010-2020 and what caused their failure?"
        
        Example of BAD non-standalone questions:
        - "What caused their failure?" (unclear what "their" refers to)
        - "Which ones shut down?" (unclear what "ones" means)
        
        Format your response as a JSON list of strings, like this:
        ["Question 1", "Question 2", "Question 3"]
        
        Research Questions:
        """

        response = await self.llm.ainvoke([HumanMessage(content=planning_prompt)])

        try:
            questions = json.loads(response.content.strip())
            if not isinstance(questions, list):
                questions = [query]  # Fallback
        except:
            questions = [query]  # Fallback

        # Validate and improve questions to ensure they are standalone
        validated_questions = await self._validate_standalone_questions(
            questions, query
        )

        print(f"✅ Generated {len(validated_questions)} standalone research questions:")
        for i, question in enumerate(validated_questions, 1):
            print(f"   {i}. {question}")
        print("=" * 60)

        logger.info(f"Generated {len(questions)} research questions")

        return {
            "research_questions": validated_questions,
            "messages": [
                AIMessage(
                    content=f"Research plan created with {len(validated_questions)} standalone questions."
                )
            ],
            "next_action": "execute_research",
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


class TavilyExecutionAgent:
    def __init__(self, retriever: TavilyRetriever, llm: ChatOpenAI):
        self.retriever = retriever
        self.llm = llm

    async def execute_research(self, state: ResearchState) -> Dict[str, Any]:
        print("\n🔍 TAVILY AGENT: Executing research...")
        print("=" * 60)

        questions = state["research_questions"]
        all_results = []

        for i, question in enumerate(questions, 1):
            print(
                f"   Searching question {i}: {question[:80]}{'...' if len(question) > 80 else ''}"
            )
            results = await self.retriever.search(question, max_results=5)
            all_results.extend(results)
            print(f"   ✓ Found {len(results)} sources")

        print(f"✅ Tavily research complete: {len(all_results)} total sources")
        print("=" * 60)

        logger.info(f"Tavily agent completed research with {len(all_results)} sources")

        # Return only the specific updates this agent is responsible for
        return {
            "tavily_results": all_results,
            "messages": [
                AIMessage(content=f"Tavily agent found {len(all_results)} sources.")
            ],
        }


class FirecrawlExecutionAgent:
    def __init__(self, retriever: FirecrawlRetriever, llm: ChatOpenAI):
        self.retriever = retriever
        self.llm = llm

    async def execute_research(self, state: ResearchState) -> Dict[str, Any]:
        print("\n🕷️  FIRECRAWL AGENT: Executing research...")
        print("=" * 60)

        questions = state["research_questions"]
        all_results = []

        for i, question in enumerate(questions, 1):
            print(
                f"   Searching question {i}: {question[:80]}{'...' if len(question) > 80 else ''}"
            )
            results = await self.retriever.search(question, max_results=5)
            all_results.extend(results)
            print(f"   ✓ Found {len(results)} sources")

        print(f"✅ Firecrawl research complete: {len(all_results)} total sources")
        print("=" * 60)

        logger.info(
            f"Firecrawl agent completed research with {len(all_results)} sources"
        )

        # Return only the specific updates this agent is responsible for
        return {
            "firecrawl_results": all_results,
            "messages": [
                AIMessage(content=f"Firecrawl agent found {len(all_results)} sources.")
            ],
        }


class ResearchAnalyzer:
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

    async def analyze_and_synthesize(self, state: ResearchState) -> Dict[str, Any]:
        print("\n🧠 ANALYZER: Synthesizing research results...")
        print("=" * 60)

        tavily_results = state.get("tavily_results", [])
        firecrawl_results = state.get("firecrawl_results", [])

        print(f"   📊 Processing {len(tavily_results)} Tavily sources")
        print(f"   📊 Processing {len(firecrawl_results)} Firecrawl sources")

        # Combine and deduplicate sources
        all_sources = tavily_results + firecrawl_results
        unique_sources = self._deduplicate_sources(all_sources)
        sorted_sources = sorted(unique_sources, key=lambda x: x.score, reverse=True)

        final_sources = sorted_sources[:20]  # Limit to top 20

        print(f"   🔄 Deduplicated to {len(final_sources)} unique sources")
        print("   📝 Generating summary...")

        # Generate summary
        summary = await self._generate_summary(state["original_query"], final_sources)

        print("   📄 Generating final report...")

        # Generate report
        report = await self._generate_report(
            state["original_query"], final_sources, summary
        )

        print("✅ Analysis and synthesis complete!")
        print("=" * 60)

        logger.info(f"Analysis completed with {len(final_sources)} final sources")

        return {
            "all_sources": final_sources,
            "summary": summary,
            "report": report,
            "messages": [
                AIMessage(content="Research analysis and synthesis completed.")
            ],
            "next_action": "complete",
        }

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

        Please provide a well-structured summary that:
        1. Identifies the main themes and findings
        2. Highlights any consensus or disagreements across sources
        3. Notes any gaps or limitations in the available information
        4. Organizes information in a logical, coherent manner

        Summary:
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


class LangGraphResearcher:
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
        self.planner = PlannerAgent(self.llm)
        self.tavily_agent = TavilyExecutionAgent(self.tavily_retriever, self.llm)
        self.firecrawl_agent = FirecrawlExecutionAgent(
            self.firecrawl_retriever, self.llm
        )
        self.analyzer = ResearchAnalyzer(self.llm)

        # Build the graph
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(ResearchState)

        # Add nodes
        workflow.add_node("planner", self.planner.plan_research)
        workflow.add_node("tavily_executor", self.tavily_agent.execute_research)
        workflow.add_node("firecrawl_executor", self.firecrawl_agent.execute_research)
        workflow.add_node("analyzer", self.analyzer.analyze_and_synthesize)

        # Define the flow
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "tavily_executor")
        workflow.add_edge("planner", "firecrawl_executor")
        workflow.add_edge(["tavily_executor", "firecrawl_executor"], "analyzer")
        workflow.add_edge("analyzer", END)

        return workflow.compile()

    async def conduct_research(self, query: str) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print("🚀 LANGGRAPH RESEARCH WORKFLOW STARTING")
        print("=" * 80)
        print(f"📋 Query: {query}")
        print("🏗️  Workflow: Planner → [Tavily & Firecrawl] → Analyzer")
        print("=" * 80)

        logger.info(f"Starting LangGraph research for query: {query}")

        initial_state: ResearchState = {
            "messages": [HumanMessage(content=f"Research query: {query}")],
            "original_query": query,
            "research_questions": [],
            "tavily_results": [],
            "firecrawl_results": [],
            "all_sources": [],
            "summary": "",
            "report": "",
            "next_action": "plan",
        }

        final_state = await self.workflow.ainvoke(initial_state)

        print("\n" + "=" * 80)
        print("🎉 LANGGRAPH RESEARCH WORKFLOW COMPLETED")
        print("=" * 80)

        logger.info("LangGraph research completed")
        return final_state

    def save_research(self, state: Dict[str, Any], filename: str) -> None:
        research_data = {
            "query": state["original_query"],
            "timestamp": datetime.now().isoformat(),
            "research_questions": state["research_questions"],
            "summary": state["summary"],
            "report": state["report"],
            "sources": [asdict(source) for source in state["all_sources"]],
            "total_sources": len(state["all_sources"]),
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(research_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Research saved to {filename}")


async def main():
    # Import and use config
    from config import Config

    # Validate configuration
    config_status = Config.validate_config()
    if not config_status["valid"]:
        logger.error(f"Configuration issues: {config_status['issues']}")
        return

    # Create LangGraph researcher
    researcher = LangGraphResearcher(
        openai_api_key=Config.OPENAI_API_KEY,
        tavily_api_key=Config.TAVILY_API_KEY,
        firecrawl_api_key=Config.FIRECRAWL_API_KEY,
        model="gpt-4o",
    )

    # Conduct research using LangGraph
    query = "Companies like Uber that failed in the past and died"
    final_state = await researcher.conduct_research(query)

    # Print results
    print("=" * 80)
    print("RESEARCH QUESTIONS")
    print("=" * 80)
    for i, question in enumerate(final_state["research_questions"], 1):
        print(f"{i}. {question}")

    print("\n" + "=" * 80)
    print("RESEARCH SUMMARY")
    print("=" * 80)
    print(final_state["summary"])

    print("\n" + "=" * 80)
    print("FULL REPORT")
    print("=" * 80)
    print(final_state["report"])

    print("\n" + "=" * 80)
    print("SOURCES FOUND")
    print("=" * 80)
    print(f"Total sources: {len(final_state['all_sources'])}")
    print(f"Tavily sources: {len(final_state['tavily_results'])}")
    print(f"Firecrawl sources: {len(final_state['firecrawl_results'])}")

    # Save research
    researcher.save_research(final_state, "research_output.json")


if __name__ == "__main__":
    asyncio.run(main())
