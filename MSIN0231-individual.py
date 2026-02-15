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
    - A structured summary covering **industry overview**, **key themes / sub-areas**, and **geographic notes** (when mentioned).
    - A transparent retrieval step showing the **top Wikipedia pages used**.
    **Designed for:** Business Analysts who need a **quick, reliable starting point**.
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
        "You are a professional market research analyst preparing a concise industry briefing for a business audience."
        "Use ONLY the context retrieved from Wikipedia to answer. "
        "Maintain a professional tone, logical structure, and strong narrative flow. "
        "Do not add external knowledge or assumptions."
    )

    user_msg = f"""
Write a concise and structured industry overview for: {industry}

The report should:
- Clearly define the industry and its scope
- Explain its core structure or main segments where relevant
- Highlight major drivers, characteristics, or dynamics reflected in the extracts
- Briefly note geographic patterns if they are material in the text

Write in structured paragraphs with smooth logical transitions.
Avoid bullet points. Avoid repetition.
Do not introduce external knowledge.
Start directly with the content.

Target length: approximately 470–490 words.
The total length must NOT exceed 500 words.
If you are close to the limit, shorten slightly to stay under 500.

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
            st.success(f"Found {num_docs} relevant Wikipedia pages (max {TOP_K_WIKI}).")

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
    st.rerun()

# --- STEP 3: REPORT (Q3) ---
st.divider()
st.header("Industry Report")

if st.session_state.steps < 3:
    st.caption("Report will appear here after sources are retrieved.")
else:
    if "report_text" not in st.session_state:
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

    if "report_text" in st.session_state:
        st.write(st.session_state.report_text)