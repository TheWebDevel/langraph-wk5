import streamlit as st
import os
import re
from agents import workflow, AgentState
from vector_store import initialize_vector_store
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

@st.cache_resource
def initialize_app():
    try:
        with st.spinner("Initializing vector database..."):
            initialize_vector_store(force_rebuild=False)
        st.session_state.show_db_success = True
        st.session_state.db_initialized = True
    except Exception as e:
        st.error(f"Failed to initialize vector database: {str(e)}")
        st.info("Please run 'python initialize_db.py' to set up the database manually.")
        st.session_state.db_initialized = False
        return False
    return True

if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False

if not st.session_state.db_initialized:
    db_ready = initialize_app()
else:
    db_ready = True

def sanitize_markdown(text):
    return re.sub(r'(?<!\*)\*(?!\*)', r'\\*', text)

st.set_page_config(page_title="Multi-Agent Support System", layout="wide")

st.title("Multi-Agent Support System")
st.markdown("Ask questions about IT or Finance topics. The system will route your query to the appropriate specialist agent.")

if not db_ready:
    st.stop()

# Show temporary success message
if st.session_state.get("show_db_success", False):
    st.success("Vector database ready!", icon="✅")
    st.session_state.show_db_success = False

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Processing your query..."):
            try:
                initial_state: AgentState = {
                    "messages": [],
                    "query": prompt,
                    "classification": None,
                    "response": None,
                    "agent_used": None,
                    "tool_results": None,
                    "graph_data": None,
                    "final_answer": None,
                    "used_web_search": None
                }

                result = workflow.invoke(initial_state)

                final_answer = result.get("final_answer", "No response generated")
                graph_data = result.get("graph_data")

                st.markdown(sanitize_markdown(final_answer))

                if graph_data and "Graph generated successfully" in str(graph_data):
                    try:
                        with open("temp_graph.html", "r") as f:
                            graph_html = f.read()
                        st.components.html(graph_html, height=400)
                    except:
                        st.info("Graph visualization available")

                st.session_state.messages.append({"role": "assistant", "content": final_answer})

            except Exception as e:
                import traceback
                error_message = f"Error processing query: {str(e)}"
                st.error(error_message)
                st.error(f"Full traceback: {traceback.format_exc()}")
                st.session_state.messages.append({"role": "assistant", "content": error_message})