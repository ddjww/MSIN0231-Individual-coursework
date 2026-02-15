import streamlit as st
from langchain_community.retrievers import WikipediaRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ======================================================
# CONFIGURATION & CONSTANTS
# ======================================================
TEMPERATURE = 0.2   
MAX_TOKENS = 750  
TOP_K_WIKI = 5
# Limit each document to control prompt length and reduce noise.
MAX_CHARS_PER_DOC = 6000   

# ======================================================
# Page config
# ======================================================
st.set_page_config(
    page_title="Market Research Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("WikiPulse — Your Industry Snapshot Generator")
st.markdown(
    """
     Welcome! This tool helps you quickly build a **500 words industry snapshot** using **Wikipedia** as the only data source.
    **What you will get**
    - A concise industry snapshot powered entirely by Wikipedia.
    - Source transparency, with direct links to the Wikipedia pages used.
    - A structured starting point you can refine, expand, or adapt for deeper analysis.
    """
)

# ======================================================
# SIDEBAR: Settings (Q0)
# ======================================================
st.sidebar.header("Settings")

# Dropdown for selecting the LLM 
model_name = st.sidebar.selectbox(
    "Configuration",
    options=["gpt-4o-mini"], 
    index=0
)

# Text field for entering API key 
api_key = st.sidebar.text_input(
    "Enter OpenAI API Key",
    type="password",
    help="Your key will not be stored permanently."
)

# ======================================================
# HELPER FUNCTIONS
# ======================================================

def get_wikipedia_content(industry_query):
    """
    Top 5 relevant Wikipedia pages.
    """
    retriever = WikipediaRetriever(top_k_results=TOP_K_WIKI)
    docs = retriever.invoke(industry_query)
    return docs

def generate_industry_report(industry, context_text, api_key, model):
    """
    Generate report < 500 words using LLM.
    """
    llm = ChatOpenAI(
        model=model,
        temperature=TEMPERATURE,
        openai_api_key=api_key,
        max_tokens=MAX_TOKENS
    )

    system_msg = (
        "You are a professional market research analyst writing a concise industry briefing for a business analyst at a large corporation. "
        "CRITICAL EVIDENCE RULE: Use ONLY the provided Wikipedia extracts. "
        "Do not refer to the task, the prompt, or the extracts (avoid phrases like 'the extracts provided' or 'the text does not cover'). "
        "Write in a decision-relevant, analytical tone (not an encyclopedia style). No bullet points."
        "Synthesize information across multiple extracts and ensure every analytical claim is cited."
    )

    user_msg = f"""
Write a concise and structured industry overview for: {industry}

HARD CONSTRAINTS (must follow exactly):
- Length: 420–450 words (MUST be < 480).
- Structure: EXACTLY 4 paragraphs.  No headings.  No bullet points.
- Tone: Senior Analyst level. Avoid descriptive "encyclopedia" style; use evaluative language.
- Sources: Use ONLY the Wikipedia extracts below.  If a claim is not explicitly supported, omit it.
- No meta-language: Do not mention the extracts, the task, or limitations (e.g., avoid “the extracts provided”).

CITATIONS (mandatory):
- Use [Source: Page Title] for key claims.
- Each paragraph must include at least one citation.
- Cross-source synthesis: Paragraph 2, Paragraphs 3 and Paragraph 4 MUST each include 2+ citations from different pages.
- Do not invent page titles.

PARAGRAPH PLAN (write exactly these 4 paragraphs):
1) Definition & boundary: define what the industry includes (and excludes) as supported by the extracts.
2) Structure & ecosystem: explain key segments/actors AND how they interact (incumbents vs entrants, partnerships), synthesising across sources.
3) Drivers & dynamics: analyse what is shifting demand, delivery, access, or cost structures. Integrate regional market examples (if present) to illustrate these global trends.
4) Constraints & trade-offs: analyse the most important risks/constraints (e.g., regulation, adoption frictions, trust/security) using evidence from multiple sources, and end with a sharp analytical implication grounded in the extracts.

STYLE (secondary):
- No generic conclusion (avoid “In conclusion/Overall…”).

Wikipedia extracts:
{context_text}
""".strip()

    report = llm.invoke([
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]).content

    return report

# ======================================================
# MAIN APP LOGIC
# ======================================================

# Initialize Session State
def init_state():
    if "steps" not in st.session_state:
        st.session_state.steps = 1
    if "docs" not in st.session_state:
        st.session_state.docs = None
    if "industry" not in st.session_state:
        st.session_state.industry = ""
    if "wiki_context" not in st.session_state:
        st.session_state.wiki_context = ""

init_state()

# --- STEP 1: INPUT (Q1) ---
st.header("Industry Selection")

# Initialize input with session state value
industry_input = st.text_input(
    "Enter an industry to research (e.g., 'Electric Vehicles'):", 
    value=st.session_state.industry
)

# [cite_start]Q1 Check: Check if industry is provided [cite: 52]
if st.button("Generate"):
    if not api_key:
        st.error("Please enter your API Key in the sidebar first.")
    elif not industry_input.strip():
        st.warning("Please enter a industry name.") 
    else:
        st.session_state.industry = industry_input
        st.session_state.steps = 2
        st.session_state.docs = None
        st.session_state.wiki_context = ""
        st.session_state.pop("report_text", None)

        st.rerun()

# --- STEP 2: RETRIEVAL (Q2) ---
st.divider()
st.header("Data Retrieval")

if st.session_state.steps < 2:
    st.caption("Sources will appear here after you enter an industry and click Generate.")
else:
    st.info(f"Searching Wikipedia for: **{st.session_state.industry}**...")
    
    if st.session_state.docs is None:
        with st.spinner("Retrieving Wikipedia pages..."):
            # [关键修改] 使用 try-except 进行异常处理
            try:
                raw_docs = get_wikipedia_content(st.session_state.industry)
                
                # 如果找不到任何页面，报错并停止
                if not raw_docs:
                    st.error("No relevant Wikipedia pages found. Please try a different industry.")
                    st.stop()
                
                # 截取前 TOP_K_WIKI 个 (防止 retriever 返回过多)
                st.session_state.docs = raw_docs[:TOP_K_WIKI]
                
            except Exception as e:
                st.error(f"Error retrieving Wikipedia pages: {e}")
                st.stop()
    
    # Q2 Output: 展示 URL 并根据数量给提示
    if st.session_state.docs:
        num_docs = len(st.session_state.docs)
        
        # [关键修改] 优雅降级：如果少于5个，给 Warning；否则给 Success
        if num_docs < TOP_K_WIKI:
            st.warning(
                f"Only {num_docs} relevant Wikipedia pages were found. "
                "The report will be generated based on the available pages."
            )
        else:
            st.success(f"Found {num_docs} relevant Wikipedia pages.")

        wiki_context = ""
        for i, doc in enumerate(st.session_state.docs):
            source_url = doc.metadata.get("source")
            title = doc.metadata.get("title", "No Title")
            
            # 兜底逻辑：防止 source 为空
            if not source_url:
                safe_title = title.replace(" ", "_")
                source_url = f"https://en.wikipedia.org/wiki/{safe_title}"
            
            st.markdown(f"**{i+1}. [{title}]({source_url})**")
            
            # 安全截断
            clean_content = (doc.page_content or "")[:MAX_CHARS_PER_DOC]
            wiki_context += f"Source: {source_url}\nContent: {clean_content}\n\n"
        
        st.session_state.wiki_context = wiki_context
    
    st.session_state.steps = 3

# --- STEP 3: REPORT (Q3) ---
st.divider()
st.header("Industry Report")

if st.session_state.steps < 3:
    st.caption("Report will appear here after sources are retrieved.")
else:
    if not st.session_state.get("report_text"):
        with st.spinner("Generating report..."):
            try:
                st.session_state.report_text = generate_industry_report(
                    st.session_state.industry,
                    st.session_state.wiki_context,
                    api_key,
                    model_name
                )
            except Exception as e:
                st.error(f"Error generating report: {e}")
                st.session_state.report_text = ""

    if st.session_state.get("report_text"):
        st.write(st.session_state.report_text)