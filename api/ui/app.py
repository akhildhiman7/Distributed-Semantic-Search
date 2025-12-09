import streamlit as st
import requests
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer

# Initialize embedder for UI
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Page configuration
st.set_page_config(
    page_title="Paper Search Interface",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .paper-card {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
        background-color: #F9FAFB;
    }
    .score-badge {
        background-color: #10B981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
    }
    .category-badge {
        background-color: #3B82F6;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-size: 0.75rem;
        margin-right: 0.25rem;
    }
    .rate-limit-info {
        background-color: #FEF3C7;
        padding: 0.5rem;
        border-radius: 6px;
        border-left: 4px solid #F59E0B;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_URL = "http://localhost:8000"

# Title
st.markdown('<h1 class="main-header">📚 Academic Paper Search</h1>', unsafe_allow_html=True)
st.markdown("**Semantic search across 510,000+ research papers using vector similarity**")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API status
    st.subheader("API Status")
    try:
        health_response = requests.get(f"{API_URL}/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            if health_data["status"] == "healthy":
                st.success("✅ API is healthy")
                st.info(f"📊 Papers in database: **{health_data['num_papers']:,}**")
            else:
                st.warning("⚠️ API is degraded")
        else:
            st.error("❌ API is unavailable")
    except:
        st.error("❌ Cannot connect to API")
        st.info("Make sure the FastAPI server is running on http://localhost:8000")
    
    # Rate limit info
    st.subheader("Rate Limits")
    st.markdown("""
    <div class="rate-limit-info">
    ⚡ **100 requests per minute**<br>
    Check response headers for:<br>
    • X-RateLimit-Limit<br>
    • X-RateLimit-Remaining<br>
    • X-RateLimit-Reset
    </div>
    """, unsafe_allow_html=True)
    
    # Quick stats
    try:
        stats_response = requests.get(f"{API_URL}/stats")
        if stats_response.status_code == 200:
            stats_data = stats_response.json()
            st.metric("Total Papers", f"{stats_data['num_papers']:,}")
            st.metric("Index Status", "✅ Built" if stats_data["has_index"] else "❌ Not built")
    except:
        pass

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔍 Search Papers", "📥 Insert Paper", "📊 System Info"])

# Tab 1: Search Papers
with tab1:
    st.markdown('<h2 class="sub-header">Search Papers</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input(
            "Search query",
            placeholder="Enter keywords or a research question...",
            help="The search uses semantic similarity, not just keyword matching."
        )
    
    with col2:
        top_k = st.slider("Results to show", 1, 20, 10)
    
    # Advanced filters
    with st.expander("🔧 Advanced Filters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            min_score = st.slider("Minimum similarity score", 0.0, 1.0, 0.3, 0.05)
        
        with col2:
            categories_input = st.text_input(
                "Filter by categories (comma-separated)",
                placeholder="e.g., cs.AI, stat.ML, cs.LG",
                help="arXiv categories like cs.AI (Computer Science - AI)"
            )
            categories = [cat.strip() for cat in categories_input.split(",")] if categories_input else None
    
    # Search button
    if st.button("🔎 Search Papers", type="primary", use_container_width=True):
        if not search_query:
            st.warning("Please enter a search query")
        else:
            with st.spinner("Searching papers..."):
                try:
                    # Prepare request
                    payload = {
                        "query": search_query,
                        "top_k": top_k,
                        "min_score": min_score,
                        "categories": categories
                    }
                    
                    # Make API call
                    start_time = time.time()
                    response = requests.post(f"{API_URL}/search", json=payload)
                    elapsed = time.time() - start_time
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        # Show rate limit headers
                        rate_limit = {
                            "limit": response.headers.get("X-RateLimit-Limit", "N/A"),
                            "remaining": response.headers.get("X-RateLimit-Remaining", "N/A"),
                            "reset": response.headers.get("X-RateLimit-Reset", "N/A")
                        }
                        
                        st.success(f"Found {len(results)} papers in {elapsed:.2f}s")
                        
                        # Rate limit info
                        st.caption(f"Rate limit: {rate_limit['remaining']}/{rate_limit['limit']} requests remaining")
                        
                        if results:
                            # Create dataframe for results
                            df_data = []
                            for paper in results:
                                df_data.append({
                                    "ID": paper["paper_id"],
                                    "Title": paper["title"],
                                    "Abstract": paper["abstract"][:200] + "..." if len(paper["abstract"]) > 200 else paper["abstract"],
                                    "Categories": ", ".join(paper["categories"]),
                                    "Score": paper["score"]
                                })
                            
                            df = pd.DataFrame(df_data)
                            
                            # Display results
                            for i, paper in enumerate(results):
                                with st.container():
                                    st.markdown(f"""
                                    <div class="paper-card">
                                        <div style="display: flex; justify-content: space-between; align-items: start;">
                                            <h3 style="margin: 0; flex: 1;">{paper['title']}</h3>
                                            <span class="score-badge">{paper['score']:.3f}</span>
                                        </div>
                                        <p style="color: #6B7280; margin: 0.5rem 0; font-size: 0.9rem;">
                                            <strong>Paper ID:</strong> {paper['paper_id']}
                                        </p>
                                        <div style="margin: 0.5rem 0;">
                                            {" ".join([f'<span class="category-badge">{cat}</span>' for cat in paper["categories"]])}
                                        </div>
                                        <p style="margin: 0.75rem 0 0 0;">
                                            {paper['abstract'][:300]}...
                                        </p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                    with st.expander(f"View full abstract for Paper {paper['paper_id']}"):
                                        st.write(paper["abstract"])
                                        
                                    st.divider()
                        else:
                            st.info("No papers found matching your search criteria.")
                            
                    elif response.status_code == 429:
                        st.error("⏳ Rate limit exceeded! Please wait before making more requests.")
                        st.json(response.json())
                    else:
                        st.error(f"Search failed: {response.status_code}")
                        st.json(response.json())
                        
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to the API. Make sure it's running on http://localhost:8000")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Tab 2: Insert Paper
with tab2:
    st.markdown('<h2 class="sub-header">Insert New Paper</h2>', unsafe_allow_html=True)
    
    st.info("""
    **Note:** This required a pre-computed 384-dimensional embedding vector.
    Use the embedding generator script to create vectors from paper text.
    """)
    
    with st.form("insert_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            paper_id = st.number_input(
                "Paper ID (INT64)",
                min_value=1,
                max_value=10**12,
                value=1000000,
                help="Must be a unique integer"
            )
            
            title = st.text_input(
                "Title",
                placeholder="Paper title",
                max_chars=500
            )
        
        with col2:
            categories_input = st.text_input(
                "Categories (comma-separated)",
                placeholder="cs.AI, stat.ML, cs.LG",
                help="arXiv-style categories"
            )
            
            vector_input = st.text_area(
                "Embedding Vector (384 comma-separated floats)",
                placeholder="Optional...",
                height=100,
                help="384-dimensional vector as comma-separated values"
            )
        
        abstract = st.text_area(
            "Abstract",
            placeholder="Paper abstract",
            height=150,
            max_chars=4000
        )
        
        submitted = st.form_submit_button("📤 Insert Paper", type="primary", use_container_width=True)
        
        
        
        if submitted:
            # Validate inputs
            errors = []
            if not title:
                errors.append("Title is required")
            if not abstract:
                errors.append("Abstract is required")
            # REMOVE vector validation
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                try:
                    # GENERATE embedding automatically
                    text_to_embed = f"{title} {abstract}"
                    vector = embedder.encode(text_to_embed).tolist()
                    
                    # Parse categories
                    categories = [cat.strip() for cat in categories_input.split(",")] if categories_input else []
                    
                    # Prepare payload
                    payload = {
                        "paper_id": int(paper_id),
                        "title": title,
                        "abstract": abstract,
                        "categories": categories,
                        "vector": vector
                    }
                    
                    with st.spinner("Inserting paper..."):
                        response = requests.post(f"{API_URL}/insert", json=payload)
                        
                        if response.status_code == 200:
                            st.success("✅ Paper inserted successfully!")
                            st.balloons()
                            st.json(response.json())
                        elif response.status_code == 429:
                            st.error("⏳ Rate limit exceeded!")
                            st.json(response.json())
                        else:
                            st.error(f"Insert failed: {response.status_code}")
                            st.json(response.json())
                            
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# Tab 3: System Info
with tab3:
    st.markdown('<h2 class="sub-header">System Information</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("API Endpoints")
        
        endpoints = {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /stats": "Collection statistics",
            "POST /search": "Search papers",
            "POST /insert": "Insert paper",
            "POST /batch_insert": "Insert multiple papers"
        }
        
        for endpoint, description in endpoints.items():
            st.code(f"curl http://localhost:8000{endpoint.split(' ')[1]}", language="bash")
            st.caption(description)
            st.divider()
    
    with col2:
        st.subheader("Current Status")
        
        try:
            # Health status
            health_response = requests.get(f"{API_URL}/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                
                status_emoji = "✅" if health_data["status"] == "healthy" else "⚠️"
                st.metric("API Health", f"{status_emoji} {health_data['status'].upper()}")
                
                cols = st.columns(2)
                with cols[0]:
                    st.metric("Milvus", "Connected" if health_data["milvus_connected"] else "Disconnected")
                with cols[1]:
                    st.metric("Collection", "Loaded" if health_data["collection_loaded"] else "Not loaded")
                
                st.metric("Total Papers", f"{health_data['num_papers']:,}")
                
                # Rate limit info from last response
                if 'X-RateLimit-Remaining' in health_response.headers:
                    st.divider()
                    st.subheader("Rate Limits")
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Limit", health_response.headers['X-RateLimit-Limit'])
                    with cols[1]:
                        st.metric("Remaining", health_response.headers['X-RateLimit-Remaining'])
                    with cols[2]:
                        reset_time = datetime.fromtimestamp(int(health_response.headers['X-RateLimit-Reset']))
                        st.metric("Resets", reset_time.strftime("%H:%M:%S"))
            
            # Stats
            stats_response = requests.get(f"{API_URL}/stats")
            if stats_response.status_code == 200:
                stats_data = stats_response.json()
                st.divider()
                st.subheader("Collection Info")
                st.code(json.dumps(stats_data, indent=2), language="json")
                
        except:
            st.error("Could not retrieve system information")

# Footer
st.divider()
st.caption("""
**Paper Search API** • Powered by Milvus Vector Database & Sentence Transformers  
**Backend**: FastAPI • **Frontend**: Streamlit • **Database**: 510,000+ research papers  
**Rate limiting**: 100 requests/minute • **Vector dimension**: 384
""")