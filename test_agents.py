import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents import supervisor_agent, decider_agent, it_agent, finance_agent, chat_agent, llm
from dotenv import load_dotenv
from tqdm import tqdm
import re

load_dotenv()

test_results = []
llm_judge_scores = []

# ANSI color codes
RESET = "\033[0m"
BOLD = "\033[1m"
UNDERLINE = "\033[4m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"


def llm_judge(query, answer, agent_name):
    if agent_name == "Decider Agent":
        prompt = f"""You are an expert evaluator. This agent's job is to classify the query as 'IT', 'Finance', or 'CHAT'.\nUser Query: {query}\nSystem Output: {answer}\n\nRate the output for:\n- Correctness: Did it classify correctly? (1-5)\n- Helpfulness: Is the classification useful for routing? (1-5)\n- Clarity: Is the classification clear? (1-5)\nRespond in this format:\nCorrectness: <1-5>\nHelpfulness: <1-5>\nClarity: <1-5>\nJustification: <short explanation>\n"""
    else:
        prompt = f"""You are an external evaluator, NOT the agent.\nYour job is to rate the following agent output for correctness, helpfulness, and clarity.\nRespond ONLY in this format and do NOT provide commentary or meta-reflection:\nCorrectness: <1-5>\nHelpfulness: <1-5>\nClarity: <1-5>\nJustification: <short explanation>\n\nUser Query: {query}\nAgent Output: {answer}\n"""
    result = llm.invoke(prompt)
    if agent_name in ["IT Agent", "Finance Agent"]:
        print(f"RAW LLM JUDGE OUTPUT for {agent_name}:\n", result.content)
    # Improved regex-based parsing
    lines = str(result.content).splitlines()
    scores = {"Correctness": None, "Helpfulness": None, "Clarity": None, "Justification": ""}
    for line in lines:
        m = re.match(r"Correctness:\s*(\d+)", line, re.I)
        if m:
            scores["Correctness"] = m.group(1)
        m = re.match(r"Helpfulness:\s*(\d+)", line, re.I)
        if m:
            scores["Helpfulness"] = m.group(1)
        m = re.match(r"Clarity:\s*(\d+)", line, re.I)
        if m:
            scores["Clarity"] = m.group(1)
        if line.lower().startswith("justification:"):
            scores["Justification"] = line.split(":", 1)[1].strip()
    llm_judge_scores.append((agent_name, scores))
    print(f"{YELLOW}{BOLD}LLM Judge Evaluation for {agent_name}:{RESET}\n", result.content)
    return result.content


def run_test(name, func, query, get_answer):
    start = time.time()
    try:
        result = func()
        duration = time.time() - start
        print(f"{BLUE}{UNDERLINE}{name}{RESET}")
        print(f"{GREEN}[PASS]{RESET} {name} ({duration:.2f}s)")
        test_results.append((name, True, duration))
        # LLM judge evaluation
        answer = get_answer(result)
        llm_judge(query, answer, name)
    except AssertionError as e:
        duration = time.time() - start
        print(f"{BLUE}{UNDERLINE}{name}{RESET}")
        print(f"{RED}[FAIL]{RESET} {name} ({duration:.2f}s): {e}")
        test_results.append((name, False, duration))
    except Exception as e:
        duration = time.time() - start
        print(f"{BLUE}{UNDERLINE}{name}{RESET}")
        print(f"{RED}[ERROR]{RESET} {name} ({duration:.2f}s): {e}")
        test_results.append((name, False, duration))


def test_supervisor_agent():
    state = {"query": "How do I reset my password?", "messages": []}
    result = supervisor_agent(state)
    print("Supervisor Agent Result:", result)
    assert "messages" in result and isinstance(result["messages"], list)
    assert result["agent_used"] == "supervisor"
    return result

def test_decider_agent():
    state = {"query": "How do I reset my password?", "messages": []}
    result = decider_agent(state)
    print("Decider Agent Result:", result)
    assert "classification" in result
    assert result["agent_used"] == "decider"
    return result

def test_it_agent():
    state = {"query": "How do I reset my password?", "messages": []}
    result = it_agent(state)
    print("IT Agent Result:", result)
    assert "response" in result
    assert result["agent_used"] == "it"
    return result

def test_finance_agent():
    state = {"query": "How do I file a reimbursement?", "messages": []}
    result = finance_agent(state)
    print("Finance Agent Result:", result)
    assert "response" in result
    assert result["agent_used"] == "finance"
    return result

def test_chat_agent():
    state = {"query": "Hello! How are you?", "messages": []}
    result = chat_agent(state)
    print("Chat Agent Result:", result)
    assert "response" in result
    assert result["agent_used"] == "chat"
    return result

if __name__ == "__main__":
    print(f"{BOLD}{YELLOW}Running unit tests for all agents...{RESET}\n")
    tests = [
        ("Supervisor Agent", test_supervisor_agent, "How do I reset my password?", lambda r: str(r.get("messages", ""))),
        ("Decider Agent", test_decider_agent, "How do I reset my password?", lambda r: str(r.get("classification", ""))),
        ("IT Agent", test_it_agent, "How do I reset my password?", lambda r: r.get("response", "")),
        ("Finance Agent", test_finance_agent, "How do I file a reimbursement?", lambda r: r.get("response", "")),
        ("Chat Agent", test_chat_agent, "Hello! How are you?", lambda r: r.get("response", "")),
    ]
    with ThreadPoolExecutor() as executor:
        futures = []
        for name, func, query, get_answer in tests:
            futures.append(executor.submit(run_test, name, func, query, get_answer))
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Running tests", ncols=80, colour='cyan'):
            pass

    print(f"\n{BOLD}{YELLOW}Test Summary:{RESET}")
    print(f"{YELLOW}{'=' * 40}{RESET}")
    total = len(test_results)
    passed = sum(1 for _, ok, _ in test_results if ok)
    # Sort results by the order in the tests list
    test_order = [name for name, _, _, _ in tests]
    sorted_results = sorted(test_results, key=lambda x: test_order.index(x[0]) if x[0] in test_order else 999)
    for name, ok, duration in sorted_results:
        status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"{BLUE}{name:25}{RESET} {status:4} ({duration:.2f}s)")
    print(f"{YELLOW}{'=' * 40}{RESET}")
    print(f"{BOLD}Total: {total}, Passed: {GREEN}{passed}{RESET}, Failed: {RED}{total - passed}{RESET}{RESET}")

    # LLM Judge Summary
    print(f"\n{BOLD}{YELLOW}LLM Judge Summary:{RESET}")
    print(f"{YELLOW}{'=' * 40}{RESET}")
    print(f"{BOLD}{'Agent':25} {'Corr.':>6} {'Help.':>6} {'Clarity':>8}  Justification{RESET}")
    for agent_name, scores in llm_judge_scores:
        print(f"{BLUE}{agent_name:25}{RESET} "
              f"{(scores['Correctness'] or 'N/A'):>6} "
              f"{(scores['Helpfulness'] or 'N/A'):>6} "
              f"{(scores['Clarity'] or 'N/A'):>8}  "
              f"{scores['Justification'] or ''}")
    print(f"{YELLOW}{'=' * 40}{RESET}")