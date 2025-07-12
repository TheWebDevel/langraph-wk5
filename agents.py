import os
from typing import Dict, List, Any, Optional, TypedDict
from langchain_community.chat_models import BedrockChat
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv
from vector_store import vector_search_impl

load_dotenv()

import boto3
bedrock_client = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION", "us-east-1"))
llm = BedrockChat(
    client=bedrock_client,
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"temperature": 0}
)

class AgentState(TypedDict):
    messages: List[Any]
    query: str
    classification: Optional[str]
    response: Optional[str]
    agent_used: Optional[str]
    tool_results: Optional[Dict]
    graph_data: Optional[Dict]
    final_answer: Optional[str]
    used_web_search: Optional[bool]

from langchain_core.tools import tool
vector_search = tool(vector_search_impl)

@tool
def web_search(query: str) -> str:
    """Search the web for current information using Tavily search."""
    try:
        if not os.getenv("TAVILY_API_KEY"):
            return "Tavily API key not configured. Please set TAVILY_API_KEY in your environment variables."

        search = TavilySearchResults(max_results=5)
        results = search.invoke(query)

        formatted_results = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            content = result.get('content', 'No content')
            url = result.get('url', 'No URL')
            formatted_results.append(f"[{i}] {title}\n{content}\nURL: {url}\n")

        return "\n".join(formatted_results)
    except Exception as e:
        return f"Error in web search: {str(e)}"

tool_node = ToolNode([vector_search, web_search])

def create_agent_with_tools(llm, prompt):
    return prompt | llm

SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a Supervisor Agent. Analyze the user query and prepare it for classification. Extract key information and context."),
    ("human", "{query}")
])

DECIDER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a Decider Agent. Classify the query as 'IT', 'Finance', or 'CHAT'. Respond with ONLY the classification. CHAT is for greetings, casual conversation, and non-business queries."),
    ("human", "{query}")
])

IT_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an IT Support Agent. Help with IT-related queries using available tools. Provide clear, actionable solutions."),
    ("human", "{query}")
])

FINANCE_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a Finance Support Agent. Help with finance-related queries. Use tools to provide accurate information."),
    ("human", "{query}")
])

CALL_TOOL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a Tool Agent. Execute the requested tool and return results. Use vector_search for FAQ queries, web_search for current information."),
    ("human", "{query}")
])

CHAT_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are an AI Assistant created by Sathish for Innovation Sprint Enablement. You specialize in answering Finance and IT queries with expertise and professionalism. Be friendly, helpful, and engaging while maintaining a professional tone. You can assist with IT support, financial guidance, and general questions. For any factual questions, current events, or information that might change over time, use web search to provide accurate and up-to-date information. IMPORTANT: Do not mention 'Claude', 'Anthropic', or any other AI model names. Always identify yourself as 'AI Assistant created by Sathish for Innovation Sprint Enablement''"),
    ("human", "{query}")
])

GRAPH_GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a Graph Generator Agent. Create visualizations for finance data. Determine appropriate graph types and generate them."),
    ("human", "{query}")
])

def supervisor_agent(state: AgentState) -> AgentState:
    messages = SUPERVISOR_PROMPT.format_messages(query=state["query"])
    response = llm.invoke(messages)

    return {
        **state,
        "messages": state["messages"] + [response],
        "agent_used": "supervisor"
    }

def decider_agent(state: AgentState) -> AgentState:
    messages = DECIDER_PROMPT.format_messages(query=state["query"])
    response = llm.invoke(messages)
    classification = str(response.content).strip().upper()

    return {
        **state,
        "classification": classification,
        "agent_used": "decider"
    }

def _check_relevance(internal_result: str, query: str) -> bool:
    """Check if internal search result is relevant to the query using LLM-based evaluation."""
    no_info_indicators = [
        "no relevant information found",
        "no internal policy found",
        "error in vector search",
        "no matching",
        "not found",
        "unfortunately, there is no internal policy",
        "no information found",
        "no data available"
    ]

    if not isinstance(internal_result, str) or not internal_result.strip():
        return False

    # Basic checks first
    has_qa_content = any(marker in internal_result.lower() for marker in ["q:", "a:", "question:", "answer:"])
    not_error_message = not any(indicator in internal_result.lower() for indicator in no_info_indicators)
    has_substantial_content = len(internal_result.split()) > 10

    if not (has_qa_content and not_error_message and has_substantial_content):
        return False

    # Use LLM to evaluate semantic relevance
    relevance_prompt = f"""You are a relevance evaluator. Determine if the internal policy excerpt is relevant to the user's query.

User Query: "{query}"

Internal Policy Excerpt:
{internal_result[:1000]}  # Limit length for efficiency

Evaluate if the internal policy excerpt contains information that directly answers or is highly relevant to the user's query. Consider:
1. Does it address the specific topic/issue mentioned in the query?
2. Is it about the same subject matter?
3. Would this information be useful for answering the user's question?
4. For internal company processes (like reimbursement, payroll, IT requests), be more lenient if the content is related to company policies and procedures.

Be RELEVANT if:
- The excerpt contains information about the specific process/topic requested
- It's about company policies, procedures, or guidelines related to the query
- It provides actionable steps or information for the user's request

Be NOT_RELEVANT if:
- The excerpt is completely unrelated to the query topic
- It's about external topics (like country-specific tax laws, global news, etc.)
- It doesn't contain any useful information for the user's question

Respond with ONLY "RELEVANT" or "NOT_RELEVANT"."""

    try:
        response = llm.invoke(relevance_prompt)
        response_text = str(response.content).upper().strip()
        is_relevant = response_text == "RELEVANT"
        print(f"DEBUG: LLM relevance check result: {response.content}")
        print(f"DEBUG: LLM relevance decision: {is_relevant}")
        return is_relevant
    except Exception as e:
        print(f"DEBUG: LLM relevance check failed: {e}")
        # Fallback to basic check if LLM fails
        fallback_result = has_qa_content and not_error_message and has_substantial_content
        print(f"DEBUG: Fallback relevance decision: {fallback_result}")
        return fallback_result

def _handle_agent_query(state: AgentState, category: str, agent_prompt, web_search_format: str) -> AgentState:
    """Generic handler for IT and Finance agent queries."""
    try:
        internal_result = vector_search_impl(state["query"], category)
        print(f"DEBUG: Internal result for '{state['query']}': {internal_result[:200]}...")
        print(f"DEBUG: Full internal result length: {len(internal_result)} characters")

        has_internal_info = _check_relevance(internal_result, state["query"])
        print(f"DEBUG: Final has_internal_info decision = {has_internal_info}")

        if has_internal_info:
            print("DEBUG: Using internal information")
            enhanced_query = f"Query: {state['query']}\n\nInternal {category.lower()} policy excerpt: {internal_result}\n\nPlease answer ONLY using the internal {category.lower()} policy excerpt above. First, summarize the answer in your own words for clarity. Then, quote the most relevant internal policy excerpt as the source. Do not speculate or generalize beyond the provided excerpt."
            response = llm.invoke(agent_prompt.format_messages(query=enhanced_query))
            used_web_search = False
        else:
            print("DEBUG: Falling back to web search")
            web_result = web_search(state["query"])
            enhanced_query = f"Query: {state['query']}\n\nWeb search result: {web_result}\n\n{web_search_format}"
            response = llm.invoke(agent_prompt.format_messages(query=enhanced_query))
            used_web_search = True
    except Exception as e:
        print(f"DEBUG: Exception in {category.lower()}_agent: {e}")
        try:
            web_result = web_search(state["query"])
            enhanced_query = f"Query: {state['query']}\n\nWeb search result: {web_result}\n\n{web_search_format}"
            response = llm.invoke(agent_prompt.format_messages(query=enhanced_query))
            used_web_search = True
        except Exception as web_error:
            print(f"DEBUG: Web search also failed: {web_error}")
            response = llm.invoke(agent_prompt.format_messages(query=state["query"]))
            used_web_search = False

    return {
        **state,
        "response": str(response.content),
        "agent_used": category.lower(),
        "used_web_search": used_web_search
    }

def it_agent(state: AgentState) -> AgentState:
    """IT agent with internal search and web search fallback."""
    web_search_format = """Please provide a comprehensive answer based ONLY on the web search results provided above. Format your response clearly with:
1. Key threats in bold using Markdown (**like this**)
2. Clean, readable formatting
3. References at the end as a numbered list (1. URL, 2. URL, etc.)
4. Include specific examples and recent incidents

Do not use asterisks for emphasis or decoration. Only use Markdown bold (**text**) or italic (*text*) if needed.

IMPORTANT: Do not mention any knowledge cutoff dates. Only use information from the provided web search results."""

    return _handle_agent_query(state, "IT", IT_AGENT_PROMPT, web_search_format)

def finance_agent(state: AgentState) -> AgentState:
    """Finance agent with internal search and web search fallback."""
    web_search_format = """Please provide a comprehensive answer based ONLY on the web search results provided above. Format your response clearly with:
1. Key financial figures in bold using Markdown (**like this**)
2. Clean, readable formatting
3. References at the end as a numbered list (1. URL, 2. URL, etc.)

Do not use asterisks for emphasis or decoration. Only use Markdown bold (**text**) or italic (*text*) if needed.

IMPORTANT: Do not mention any knowledge cutoff dates. Only use information from the provided web search results."""

    return _handle_agent_query(state, "Finance", FINANCE_AGENT_PROMPT, web_search_format)

def call_tool_agent(state: AgentState) -> AgentState:
    agent_with_tools = create_agent_with_tools(llm, CALL_TOOL_PROMPT)
    response = agent_with_tools.invoke({"query": state["query"]})

    return {
        **state,
        "tool_results": {"content": str(response.content)},
        "agent_used": "call_tool"
    }

def chat_agent(state: AgentState) -> AgentState:
    try:
        agent_with_tools = create_agent_with_tools(llm, CHAT_AGENT_PROMPT)
        response = agent_with_tools.invoke({"query": state["query"]})
    except:
        response = llm.invoke(CHAT_AGENT_PROMPT.format_messages(query=state["query"]))

    return {
        **state,
        "response": str(response.content),
        "agent_used": "chat"
    }

def route_to_decider(state: AgentState) -> str:
    return "decider_agent"

def route_based_on_classification(state: AgentState) -> str:
    classification = state["classification"]
    if classification == "IT":
        return "it_agent"
    elif classification == "FINANCE":
        return "finance_agent"
    elif classification == "CHAT":
        return "chat_agent"
    else:
        return "call_tool_agent"

def route_to_tools(state: AgentState) -> str:
    return "call_tool_agent"

def route_to_graph(state: AgentState) -> str:
    return "graph_generator_agent"

def create_final_answer(state: AgentState) -> AgentState:
    response = state.get("response", "")
    tool_results = state.get("tool_results", "")
    graph_data = state.get("graph_data", "")

    agent_flow = []
    data_source = "INTERNAL SOURCE"

    if state.get("agent_used") == "supervisor":
        agent_flow.append("SUPERVISOR")
    elif state.get("agent_used") == "decider":
        agent_flow.append("SUPERVISOR → DECIDER")
    elif state.get("agent_used") == "it":
        agent_flow.append("SUPERVISOR → DECIDER → IT")
        if state.get("used_web_search"):
            data_source = "WEB SEARCH"
    elif state.get("agent_used") == "finance":
        agent_flow.append("SUPERVISOR → DECIDER → FINANCE")
        if state.get("used_web_search"):
            data_source = "WEB SEARCH"
    elif state.get("agent_used") == "chat":
        agent_flow.append("SUPERVISOR → DECIDER → CHAT")
        data_source = "CHAT"
    elif state.get("agent_used") == "call_tool":
        agent_flow.append("SUPERVISOR → DECIDER → CALL_TOOL")
        data_source = "WEB SEARCH"
    elif state.get("agent_used") == "graph_generator":
        agent_flow.append("SUPERVISOR → DECIDER → FINANCE → GRAPH_GENERATOR")
        data_source = "WEB SEARCH + GRAPH GENERATION"

    agent_flow_str = " → ".join(agent_flow) if agent_flow else "UNKNOWN"

    final_answer = f"AGENT FLOW: {agent_flow_str}\nDATA SOURCE: {data_source}\n\n"
    final_answer += str(response) if response else ""
    if tool_results:
        final_answer += f"\n\n{str(tool_results)}"
    if graph_data:
        final_answer += f"\n\n{str(graph_data)}"

    return {
        **state,
        "final_answer": final_answer
    }

def build_workflow():
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor_agent", supervisor_agent)
    workflow.add_node("decider_agent", decider_agent)
    workflow.add_node("it_agent", it_agent)
    workflow.add_node("finance_agent", finance_agent)
    workflow.add_node("chat_agent", chat_agent)
    workflow.add_node("call_tool_agent", call_tool_agent)
    workflow.add_node("final_answer", create_final_answer)

    workflow.add_edge("supervisor_agent", "decider_agent")
    workflow.add_conditional_edges("decider_agent", route_based_on_classification)
    workflow.add_edge("it_agent", "final_answer")
    workflow.add_edge("finance_agent", "final_answer")
    workflow.add_edge("chat_agent", "final_answer")
    workflow.add_edge("call_tool_agent", END)
    workflow.add_edge("final_answer", END)

    workflow.set_entry_point("supervisor_agent")

    return workflow.compile()

workflow = build_workflow()