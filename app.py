
import os
import streamlit as st

from typing import TypedDict
from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langgraph.graph import StateGraph, START, END


# ==========================================
# 1. LOAD ENVIRONMENT
# ==========================================

load_dotenv()

st.set_page_config(
    page_title="LangGraph Multi-Agent System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent AI Research System")

st.caption(
    "LangGraph + OpenRouter Free Model | Three AI Agents"
)


# ==========================================
# 2. INITIALIZE OPENROUTER
# ==========================================

def get_llm():

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY missing in .env file."
        )

    return ChatOpenRouter(
        model="openrouter/free",
        api_key=api_key,
        temperature=0.2,
        max_tokens=2500,
        max_retries=2
    )


# ==========================================
# 3. SHARED AGENT STATE
# ==========================================

class AgentState(TypedDict):

    topic: str
    plan: str
    research: str
    report: str


# ==========================================
# 4. RESPONSE TEXT EXTRACTOR
# ==========================================

def get_text(response):

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):
                parts.append(
                    str(item.get("text", ""))
                )

            else:
                parts.append(str(item))

        return "\n".join(parts).strip()

    return str(content).strip()


# ==========================================
# 5. PLANNER AGENT
# ==========================================

def planner_agent(state: AgentState):

    llm = get_llm()

    prompt = f"""
    You are an expert AI Research Planner.

    Research Topic:
    {state['topic']}

    Create a concise research plan.

    Include:

    1. Research objectives
    2. Five important research questions
    3. Main areas to investigate
    4. Practical examples to examine
    5. Suggested report structure

    Important:
    - Keep the plan focused.
    - Use clear headings.
    - Do not write the final report.
    """

    response = llm.invoke(prompt)

    plan = get_text(response)

    if not plan:
        raise ValueError(
            "Planner Agent returned an empty response."
        )

    return {
        "plan": plan
    }


# ==========================================
# 6. RESEARCH AGENT
# ==========================================

def research_agent(state: AgentState):

    if not state.get("plan", "").strip():

        raise ValueError(
            "Research Agent cannot run without a research plan."
        )

    llm = get_llm()

    prompt = f"""
    You are an expert Research Analyst.

    Research Topic:
    {state['topic']}

    Research Plan:
    {state['plan']}

    Prepare useful research findings.

    Required sections:

    1. Overview
    2. Key concepts
    3. Important findings
    4. Practical examples
    5. Benefits
    6. Challenges and limitations
    7. Future opportunities
    8. Conclusion

    Provide at least five meaningful findings.

    Important rules:
    - Explain concepts clearly.
    - Do not invent statistics.
    - Do not invent references or citations.
    - Mention when information needs verification.
    - Return research findings only.
    - Never ask the user to provide research findings.
    """

    response = llm.invoke(prompt)

    research = get_text(response)

    if not research:
        raise ValueError(
            "Research Agent returned an empty response."
        )

    return {
        "research": research
    }


# ==========================================
# 7. WRITER AGENT
# ==========================================

def writer_agent(state: AgentState):

    research = state.get("research", "").strip()

    if not research:

        raise ValueError(
            "Writer Agent cannot run because research findings are missing."
        )

    llm = get_llm()

    prompt = f"""
    You are a Professional Technical Report Writer.

    Topic:
    {state['topic']}

    Research Findings:
    {research}

    Create a complete professional Markdown report.

    Report structure:

    # Title

    ## Executive Summary

    ## Introduction

    ## Main Concepts

    ## Research Findings

    ## Practical Applications

    ## Benefits

    ## Challenges and Limitations

    ## Future Opportunities

    ## Recommendations

    ## Conclusion

    ## Key Takeaways

    Important rules:

    - Use the supplied research findings.
    - Explain the topic in detail.
    - Do not invent statistics.
    - Do not invent citations.
    - Do not ask the user for more information.
    - Return the complete Markdown report only.
    """

    response = llm.invoke(prompt)

    report = get_text(response)

    if not report:
        raise ValueError(
            "Writer Agent returned an empty report."
        )

    return {
        "report": report
    }


# ==========================================
# 8. BUILD LANGGRAPH WORKFLOW
# ==========================================

def build_graph():

    workflow = StateGraph(AgentState)

    workflow.add_node(
        "planner",
        planner_agent
    )

    workflow.add_node(
        "researcher",
        research_agent
    )

    workflow.add_node(
        "writer",
        writer_agent
    )

    workflow.add_edge(
        START,
        "planner"
    )

    workflow.add_edge(
        "planner",
        "researcher"
    )

    workflow.add_edge(
        "researcher",
        "writer"
    )

    workflow.add_edge(
        "writer",
        END
    )

    return workflow.compile()


# ==========================================
# 9. STREAMLIT INTERFACE
# ==========================================

st.subheader("Research Topic")

topic = st.text_input(
    "Enter your research topic",
    placeholder=(
        "Example: Generative AI and Agentic AI "
        "in Software Development"
    )
)

generate = st.button(
    "🚀 Start AI Agents",
    type="primary",
    use_container_width=True
)


# ==========================================
# 10. EXECUTE WORKFLOW
# ==========================================

if generate:

    if not topic.strip():

        st.warning(
            "Please enter a research topic."
        )

    else:

        try:

            app = build_graph()

            initial_state = {
                "topic": topic.strip(),
                "plan": "",
                "research": "",
                "report": ""
            }

            result = {}

            # Planner
            with st.status(
                "Agent 1: Planner is working...",
                expanded=True
            ) as status:

                result = app.invoke(
                    initial_state,
                    config={"recursion_limit": 10}
                )

                if not result.get("plan", "").strip():
                    raise ValueError(
                        "Planner output is missing."
                    )

                if not result.get("research", "").strip():
                    raise ValueError(
                        "Research output is missing."
                    )

                if not result.get("report", "").strip():
                    raise ValueError(
                        "Final report is missing."
                    )

                status.update(
                    label="All three Agents completed successfully!",
                    state="complete",
                    expanded=False
                )

            st.success(
                "All three AI Agents completed their tasks!"
            )

            # Planner output
            st.subheader(
                "📋 1. Planner Agent"
            )

            with st.expander(
                "View Research Plan",
                expanded=False
            ):

                st.markdown(
                    result["plan"]
                )

            st.divider()

            # Research output
            st.subheader(
                "🔍 2. Research Agent"
            )

            with st.expander(
                "View Research Findings",
                expanded=False
            ):

                st.markdown(
                    result["research"]
                )

            st.divider()

            # Writer output
            st.subheader(
                "📝 3. Writer Agent"
            )

            st.markdown(
                result["report"]
            )

            # Download
            st.download_button(
                label="📥 Download Final Report",
                data=result["report"],
                file_name="AI_Research_Report.md",
                mime="text/markdown",
                use_container_width=True
            )

        except Exception as error:

            st.error(
                f"Application Error: {error}"
            )

            st.warning(
                "Check your OpenRouter API key, "
                "free model availability and request limits. "
                "If a model returns an incomplete response, "
                "try again."
            )


# ==========================================
# 11. FOOTER
# ==========================================

st.divider()

st.caption(
    "Project 4 | LangGraph Multi-Agent AI System | OpenRouter"
)
