from __future__ import annotations
import operator
from pathlib import Path
from typing import TypedDict, List, Annotated, Literal, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()
class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description=(
            "One sentence describing what the reader should be able "
            "to do or understand after this section"
        )
    )

    bullets: list[str] = Field(
        ...,
        min_length=3,
        max_length=5,
        description=(
            "3-5 concrete, non-overlapping subpoints to cover "
            "in this section"
        )
    )

    target_words: int = Field(
        ...,
        description="Target word count for this section (120-450)"
    )

    section_type: Literal[
        "intro",
        "core",
        "examples",
        "checklist",
        "common-mistakes",
        "conclusion"
    ] = Field(
        ...,
        description=(
            "Use 'common-mistakes' exactly once in the plan"
        )
    )
class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    key_findings: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool

    mode: Literal[
        "closed_book",
        "hybrid",
        "open_book"
    ]

    queries: List[str] = Field(
        default_factory=list
    )


class EvidencePack(BaseModel):
    evidence: list[EvidenceItem] = Field(
        default_factory=list
    )

class State(TypedDict):
    topic: str

    mode: str
    needs_research: bool
    queries: List[str]

    evidence: List[EvidenceItem]

    plan: Plan

    sections: Annotated[
        list[str],
        operator.add
    ]

    final: str
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

router_prompt = """
You are a research decision router for a technical blog writing agent.

Your job is to analyze the requested blog topic BEFORE the blog is written
and decide whether internet research is needed.

You must return a structured RouterDecision.

Rules:

1. Set needs_research=false and mode="closed_book" when the topic can be
   accurately explained using stable, well-established knowledge and does not
   require current information.

2. Set needs_research=true when the blog would benefit from or require
   information from the internet, especially when the topic involves:

   - latest or current information
   - recent developments or releases
   - current libraries, frameworks, APIs, or tools
   - current documentation
   - pricing, benchmarks, or comparisons
   - current best practices
   - facts that should be verified from reliable sources

3. Use mode="hybrid" when the main concepts are stable but some current
   information, examples, documentation, comparisons, or facts should be
   verified through internet research.

4. Use mode="open_book" when most of the important information required for
   the blog depends on current external information.

5. If needs_research=false:

   - mode MUST be "closed_book"
   - queries MUST be an empty list

6. If needs_research=true:

   - Generate exactly 4 focused search queries.
   - These queries will be passed directly to Tavily.
   - Each query must search for a different useful aspect of the topic.
   - Queries must be specific and concise.
   - Avoid vague queries.

7. Design the 4 queries to cover different research needs:

   - official documentation or technical facts
   - latest developments or releases
   - practical implementation/examples
   - limitations, trade-offs, benchmarks, or best practices

8. Do NOT perform research yourself.

Only decide whether research is needed and generate the search queries.

Return ONLY the RouterDecision schema.
"""

def router_node(state: State) -> dict:

    topic = state["topic"]

    decider = model.with_structured_output(
        RouterDecision
    )

    decision = decider.invoke([
        SystemMessage(
            content=router_prompt
        ),
        HumanMessage(
            content=f"Blog topic: {topic}"
        )
    ])

    print("\n========== ROUTER ==========")
    print("Needs research:", decision.needs_research)
    print("Mode:", decision.mode)
    print("Queries:")

    for q in decision.queries:
        print("-", q)

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries
    }

def route_next(state: State) -> str:

    if state["needs_research"]:
        return "research"

    return "orchestrator"

def _tavily_search(
    query: str,
    max_results: int = 5
) -> List[dict]:

    tool = TavilySearchResults(
        max_results=max_results
    )

    results = tool.invoke({
        "query": query
    })

    normalized = []

    for result in results or []:

        normalized.append({
            "title": result.get("title", ""),

            "url": result.get("url", ""),

            "snippet": (
                result.get("content")
                or result.get("snippet")
                or ""
            ),

            "published_at": (
                result.get("published_date")
                or result.get("published_at")
            )
        })

    return normalized


research_prompt = """
You are a research assistant for a technical blog writing agent.

You will receive search results collected from Tavily.

Your job is to extract reliable and useful evidence for the blog.

Requirements:

1. Use only information supported by the provided search results.

2. Prefer:

   - official documentation
   - official project websites
   - research papers
   - reputable engineering sources
   - trustworthy technical documentation

3. Ignore:

   - duplicate sources
   - irrelevant sources
   - low-quality sources
   - promotional content

4. Extract useful technical information such as:

   - technical facts
   - implementation details
   - practical examples
   - best practices
   - limitations
   - trade-offs
   - performance considerations
   - current information

5. Do not invent information.

6. Preserve the original source URL.

7. For every useful source provide:

   - title
   - URL
   - publication date if available
   - short snippet
   - key findings

8. Do NOT write the final blog.

Return ONLY the EvidencePack schema.
"""


def research_node(state: State) -> dict:

    queries = state.get(
        "queries",
        []
    ) or []

    raw_results: List[dict] = []

    print("\n========== TAVILY RESEARCH ==========")

    for query in queries:

        print("\nSearching:")
        print(query)

        results = _tavily_search(
            query,
            max_results=6
        )

        raw_results.extend(results)

        print(
            f"Results found: {len(results)}"
        )

    if not raw_results:

        print("No research results found.")

        return {
            "evidence": []
        }

    # Convert search results into readable text
    research_text = "\n\n".join(
        [
            (
                f"TITLE: {item['title']}\n"
                f"URL: {item['url']}\n"
                f"DATE: {item.get('published_at')}\n"
                f"CONTENT: {item['snippet']}"
            )
            for item in raw_results
        ]
    )

    extractor = model.with_structured_output(
        EvidencePack
    )

    pack = extractor.invoke([
        SystemMessage(
            content=research_prompt
        ),
        HumanMessage(
            content=(
                f"Blog topic:\n"
                f"{state['topic']}\n\n"
                f"Tavily search results:\n\n"
                f"{research_text}"
            )
        )
    ])

    # Deduplicate by URL
    dedup = {}

    for evidence in pack.evidence:

        if evidence.url:
            dedup[evidence.url] = evidence

    evidence = list(
        dedup.values()
    )

    print(
        f"\nFinal evidence items: {len(evidence)}"
    )

    return {
        "evidence": evidence
    }


# ============================================================
# 9. ORCHESTRATOR
# ============================================================

def orchestrator(state: State) -> dict:

    research_context = ""

    if state.get("evidence"):

        research_context = "\n\n".join(
            [
                (
                    f"Source: {e.title}\n"
                    f"URL: {e.url}\n"
                    f"Findings: {e.key_findings or ''}\n"
                    f"Snippet: {e.snippet or ''}"
                )
                for e in state["evidence"]
            ]
        )

    else:

        research_context = (
            "No external research was performed. "
            "Use stable technical knowledge."
        )

    orchestrator_prompt = """
You are a senior technical writer and developer advocate.

Your job is to create a highly actionable outline for a technical blog post.

Hard requirements:

- Create 5-7 sections.
- Each section must include:
  1. goal
  2. 3-5 concrete non-overlapping bullets
  3. target word count between 120-450
  4. section_type

- Include EXACTLY ONE section with:
  section_type="common-mistakes"

Technical quality:

- Assume the reader is a developer.
- Use correct technical terminology.
- Prefer:

  problem
  ->
  intuition
  ->
  approach
  ->
  implementation
  ->
  trade-offs
  ->
  testing/observability
  ->
  conclusion

- Bullets must be actionable and testable.

Examples:

- Show a minimal working example.
- Explain why a specific approach fails.
- Compare two implementation strategies.
- Add debugging or observability guidance.
- Add performance/cost considerations.
- Add security/privacy considerations when relevant.

Avoid vague bullets such as:

"Explain X"
"Discuss Y"

Every bullet should say what the writer should actually build,
compare, measure, verify, or demonstrate.

Ordering:

- Start with a crisp introduction.
- Build core concepts first.
- Move to implementation/examples.
- Include common mistakes.
- End with practical checklist and next steps.

Use the provided research evidence when relevant.

Do not invent claims that conflict with the evidence.

Output must strictly match the Plan schema.
"""

    plan = model.with_structured_output(
        Plan
    ).invoke([
        SystemMessage(
            content=orchestrator_prompt
        ),
        HumanMessage(
            content=(
                f"Topic:\n{state['topic']}\n\n"
                f"Research mode:\n{state.get('mode')}\n\n"
                f"Research evidence:\n"
                f"{research_context}"
            )
        )
    ])

    print("\n========== PLAN ==========")
    print("Blog title:", plan.blog_title)
    print("Sections:", len(plan.tasks))

    return {
        "plan": plan
    }


# ============================================================
# 10. FANOUT
# ============================================================

def fanout(state: State):

    return [
        Send(
            "worker",
            {
                "task": task,
                "topic": state["topic"],
                "plan": state["plan"],
                "evidence": state.get("evidence", [])
            }
        )
        for task in state["plan"].tasks
    ]


# ============================================================
# 11. WORKER
# ============================================================

def worker(payload: dict) -> dict:

    task = payload["task"]
    topic = payload["topic"]
    plan = payload["plan"]

    evidence = payload.get(
        "evidence",
        []
    )

    bullets_text = (
        "\n- "
        + "\n- ".join(task.bullets)
    )

    blog_title = plan.blog_title

    if evidence:

        research_text = "\n\n".join(
            [
                (
                    f"Source: {item.title}\n"
                    f"URL: {item.url}\n"
                    f"Findings: {item.key_findings or ''}\n"
                    f"Snippet: {item.snippet or ''}"
                )
                for item in evidence
            ]
        )

    else:

        research_text = (
            "No external research available."
        )

    worker_system_prompt = """
You are a senior technical writer and developer advocate.

Write ONE section of a technical blog post in Markdown.

Hard constraints:

- Follow the Goal.
- Cover ALL bullets in order.
- Do not skip bullets.
- Do not merge unrelated bullets.
- Stay close to the target word count (±15%).
- Output ONLY the section content.
- Do not write the blog H1.

Technical quality:

- Be precise.
- Be implementation-oriented.
- Prefer concrete APIs, data structures, protocols,
  algorithms, and exact technical terminology.

When relevant include:

- small correct code snippet
- tiny input/output example
- implementation checklist
- architecture flow
- trade-offs
- edge cases
- failure modes
- debugging guidance

If research evidence is provided:

- Use it for factual claims.
- Do not invent source information.
- Do not add unsupported current claims.

Markdown:

- Start with ## <Section Title>
- Use short paragraphs.
- Use bullet lists when useful.
- Use code fences for code.
- Avoid fluff.
- Avoid marketing language.

Return only the Markdown section.
"""

    section_md = model.invoke([
        SystemMessage(
            content=worker_system_prompt
        ),

        HumanMessage(
            content=(
                f"Blog: {blog_title}\n"
                f"Topic: {topic}\n\n"

                f"Section: {task.title}\n"
                f"Section type: {task.section_type}\n"

                f"Goal:\n"
                f"{task.goal}\n\n"

                f"Target words:\n"
                f"{task.target_words}\n\n"

                f"Bullets:\n"
                f"{bullets_text}\n\n"

                f"Research evidence:\n"
                f"{research_text}\n\n"

                "Return only the section content in Markdown."
            )
        )
    ]).content.strip()

    return {
        "sections": [
            section_md
        ]
    }


# ============================================================
# 12. REDUCER
# ============================================================

def reducer(state: State) -> dict:

    title = state["plan"].blog_title

    body = "\n\n".join(
        state["sections"]
    ).strip()

    final_md = (
        f"# {title}\n\n"
        f"{body}\n"
    )

    filename = (
        title
        .lower()
        .replace(" ", "_")
        .strip()
        + ".md"
    )

    output_path = Path(
        filename
    ).resolve()

    output_path.write_text(
        final_md,
        encoding="utf-8"
    )

    print("\n========== FILE ==========")
    print("FILE PATH:", output_path)
    print("FILE EXISTS:", output_path.exists())
    print("FILE SIZE:", output_path.stat().st_size)

    return {
        "final": final_md
    }
 
graph = StateGraph(State)

graph.add_node(
    "router",
    router_node
)

graph.add_node(
    "research",
    research_node
)

graph.add_node(
    "orchestrator",
    orchestrator
)

graph.add_node(
    "worker",
    worker
)

graph.add_node(
    "reducer",
    reducer
)


# START
graph.add_edge(
    START,
    "router"
)


# Router decides:
# research -> research node
# no research -> orchestrator
graph.add_conditional_edges(
    "router",
    route_next,
    {
        "research": "research",
        "orchestrator": "orchestrator"
    }
)


# Research -> orchestrator
graph.add_edge(
    "research",
    "orchestrator"
)


# Orchestrator -> dynamic workers
graph.add_conditional_edges(
    "orchestrator",
    fanout,
    ["worker"]
)


# Workers -> reducer
graph.add_edge(
    "worker",
    "reducer"
)


# Reducer -> END
graph.add_edge(
    "reducer",
    END
)



app = graph.compile()
out = app.invoke({
    "topic": "Write a blog on self attention"
})


print("\n\n========== FINAL BLOG ==========\n")

print(
    out["final"]
)