import streamlit as st
import asyncio
import sys
import os
from datetime import datetime
import json
from research_system import LangGraphResearcher
from config import Config

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 AI Research Assistant")
st.markdown("Get comprehensive research reports with real-time progress updates")

if "research_state" not in st.session_state:
    st.session_state.research_state = None
if "research_complete" not in st.session_state:
    st.session_state.research_complete = False
if "progress_messages" not in st.session_state:
    st.session_state.progress_messages = []


class StreamlitProgressHandler:
    def __init__(self, progress_container, status_container):
        self.progress_container = progress_container
        self.status_container = status_container
        self.current_step = 0
        self.total_steps = 4

    def update_progress(self, step_name, message, step_number=None):
        if step_number:
            self.current_step = step_number

        progress = self.current_step / self.total_steps
        self.progress_container.progress(
            progress, text=f"Step {self.current_step}/{self.total_steps}: {step_name}"
        )

        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.progress_messages.append(f"[{timestamp}] {message}")

        with self.status_container:
            st.text_area(
                "Progress Log",
                value="\n".join(st.session_state.progress_messages[-10:]),
                height=200,
                key=f"progress_log_{len(st.session_state.progress_messages)}",
            )


class CustomLangGraphResearcher(LangGraphResearcher):
    def __init__(self, *args, progress_handler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_handler = progress_handler

    async def conduct_research(self, query: str):
        if self.progress_handler:
            self.progress_handler.update_progress(
                "Planning", "🎯 Breaking down research query...", 1
            )

        initial_state = {
            "messages": [],
            "original_query": query,
            "research_questions": [],
            "tavily_results": [],
            "firecrawl_results": [],
            "all_sources": [],
            "summary": "",
            "report": "",
            "next_action": "plan",
        }

        if self.progress_handler:
            self.progress_handler.update_progress(
                "Planning", "✅ Research questions generated", 1
            )

        state_after_planning = await self.planner.plan_research(initial_state)
        initial_state.update(state_after_planning)

        if self.progress_handler:
            self.progress_handler.update_progress(
                "Research", "🔍 Executing Tavily search...", 2
            )

        tavily_results = await self.tavily_agent.execute_research(initial_state)
        initial_state.update(tavily_results)

        if self.progress_handler:
            self.progress_handler.update_progress(
                "Research", "🕷️ Executing Firecrawl search...", 2
            )

        firecrawl_results = await self.firecrawl_agent.execute_research(initial_state)
        initial_state.update(firecrawl_results)

        if self.progress_handler:
            self.progress_handler.update_progress(
                "Analysis", "🧠 Analyzing and synthesizing results...", 3
            )

        final_results = await self.analyzer.analyze_and_synthesize(initial_state)
        initial_state.update(final_results)

        if self.progress_handler:
            self.progress_handler.update_progress(
                "Complete", "🎉 Research workflow completed!", 4
            )

        return initial_state


def validate_configuration():
    config_status = Config.validate_config()
    if not config_status["valid"]:
        st.error("❌ Configuration Error")
        st.write("Please check your configuration:")
        for issue in config_status["issues"]:
            st.write(f"• {issue}")
        return False
    return True


async def run_research(query, progress_handler):
    try:
        researcher = CustomLangGraphResearcher(
            openai_api_key=Config.OPENAI_API_KEY,
            tavily_api_key=Config.TAVILY_API_KEY,
            firecrawl_api_key=Config.FIRECRAWL_API_KEY,
            model="gpt-4",
            progress_handler=progress_handler,
        )

        result = await researcher.conduct_research(query)
        return result, None
    except Exception as e:
        return None, str(e)


def main():
    with st.sidebar:
        st.header("🔧 Configuration")

        if not validate_configuration():
            st.stop()
        else:
            st.success("✅ Configuration validated")

        st.header("📊 Research Statistics")
        if st.session_state.research_state:
            state = st.session_state.research_state
            st.metric("Total Sources", len(state.get("all_sources", [])))
            st.metric("Tavily Sources", len(state.get("tavily_results", [])))
            st.metric("Firecrawl Sources", len(state.get("firecrawl_results", [])))
            st.metric("Research Questions", len(state.get("research_questions", [])))

    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("🎯 Research Query")
        query = st.text_area(
            "Enter your research question:",
            placeholder="e.g., Companies like Uber that failed in the past and died",
            height=100,
        )

        if st.button("🚀 Start Research", type="primary", disabled=not query.strip()):
            st.session_state.progress_messages = []
            st.session_state.research_complete = False

            progress_container = st.empty()
            status_container = st.empty()

            progress_handler = StreamlitProgressHandler(
                progress_container, status_container
            )

            with st.spinner("Initializing research workflow..."):
                try:
                    result, error = asyncio.run(
                        run_research(query.strip(), progress_handler)
                    )

                    if error:
                        st.error(f"❌ Research failed: {error}")
                    else:
                        st.session_state.research_state = result
                        st.session_state.research_complete = True
                        st.success("✅ Research completed successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")

    with col2:
        st.header("📈 Progress")
        if st.session_state.progress_messages:
            st.text_area(
                "Recent Updates",
                value="\n".join(st.session_state.progress_messages[-5:]),
                height=200,
            )
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            st.info("Progress updates will appear here during research")

    if st.session_state.research_complete and st.session_state.research_state:
        state = st.session_state.research_state

        st.divider()

        tabs = st.tabs(
            ["📋 Research Questions", "📝 Summary", "📄 Full Report", "🔗 Sources"]
        )

        with tabs[0]:
            st.header("Research Questions")
            questions = state.get("research_questions", [])
            for i, question in enumerate(questions, 1):
                st.write(f"**{i}.** {question}")

        with tabs[1]:
            st.header("Research Summary")
            summary = state.get("summary", "")
            if summary:
                st.markdown(summary)
            else:
                st.info("No summary available")

        with tabs[2]:
            st.header("Full Research Report")
            report = state.get("report", "")
            if report:
                st.markdown(report)
            else:
                st.info("No report available")

        with tabs[3]:
            st.header("Sources")
            sources = state.get("all_sources", [])

            if sources:
                for i, source in enumerate(sources, 1):
                    with st.expander(f"[{i}] {source.title}"):
                        st.write(f"**URL:** {source.url}")
                        st.write(f"**Source:** {source.source}")
                        st.write(f"**Score:** {source.score:.2f}")
                        if source.published_date:
                            st.write(f"**Published:** {source.published_date}")
                        st.write("**Content:**")
                        st.write(
                            source.snippet[:500] + "..."
                            if len(source.snippet) > 500
                            else source.snippet
                        )
            else:
                st.info("No sources available")

        st.divider()

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("💾 Save Research"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"research_output_{timestamp}.json"

                research_data = {
                    "query": state["original_query"],
                    "timestamp": datetime.now().isoformat(),
                    "research_questions": state["research_questions"],
                    "summary": state["summary"],
                    "report": state["report"],
                    "sources": [
                        source.__dict__ if hasattr(source, "__dict__") else source
                        for source in state["all_sources"]
                    ],
                    "total_sources": len(state["all_sources"]),
                }

                try:
                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(research_data, f, indent=2, ensure_ascii=False)
                    st.success(f"✅ Research saved to {filename}")
                except Exception as e:
                    st.error(f"❌ Failed to save: {str(e)}")

        with col2:
            if st.button("🔄 New Research"):
                st.session_state.research_state = None
                st.session_state.research_complete = False
                st.session_state.progress_messages = []
                st.rerun()


if __name__ == "__main__":
    main()
