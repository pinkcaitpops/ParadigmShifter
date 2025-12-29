import os
import warnings
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# CrewAI & LangChain Imports
from crewai import Agent, Task, Crew, Process
# Note: Ensure you have crewai-tools installed. 
# Use `pip install crewai-tools` if SerperDevTool is not found.
try:
    from crewai_tools import SerperDevTool
except ImportError:
    # Fallback or stub if the specific tools package isn't installed, 
    # though usage will fail if not present.
    SerperDevTool = None

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Suppress warnings for cleaner logs
warnings.filterwarnings("ignore")

# Load environment variables
# Load environment variables
from pathlib import Path
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# --- Configuration ---
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
SERPER_API_KEY = os.environ.get("SERPER_API_KEY")

print(f"DEBUG: DEEPSEEK_API_KEY loaded: {bool(DEEPSEEK_API_KEY)}")
if DEEPSEEK_API_KEY:
    print(f"DEBUG: DEEPSEEK_API_KEY starts with: {DEEPSEEK_API_KEY[:5]}...")

if not DEEPSEEK_API_KEY:
    print("WARNING: DEEPSEEK_API_KEY not found in environment variables.")

if not SERPER_API_KEY:
    print("WARNING: SERPER_API_KEY not found. Search functionality may fail.")

# Initialize FastAPI
app = FastAPI(title="The Rule Bender API (DeepSeek Edition)")

class QueryRequest(BaseModel):
    message: str

class QueryResponse(BaseModel):
    result: str

@app.get("/")
def read_root():
    return {"status": "Rule Bender System Online (DeepSeek Powered)"}

@app.post("/chat", response_model=QueryResponse)
def run_agent(request: QueryRequest):
    query = request.message
    
    # ---------------------------------------------------------
    # 1. DEFINE BRAINS (DeepSeek Configuration)
    # ---------------------------------------------------------
    # High Chaos: For generating novel, "rule-bending" ideas
    chaos_llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0.9
    )
    
    # Low Chaos: For checking logic, feasibility, and risk
    logic_llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
        temperature=0.1
    )

    # Search Tool Configuration
    # Assumes SERPER_API_KEY is available in os.environ
    search_tools = []
    if SerperDevTool:
        search_tools = [SerperDevTool()]
    else:
        print("SerperDevTool not available. Running without search.")

    # ---------------------------------------------------------
    # 2. DEFINE AGENTS
    # ---------------------------------------------------------
    
    # AGENT A: The Maverick (Thesis)
    maverick = Agent(
        role='Chief Disruption Officer',
        goal='Identify the "Grey Areas" and shortcuts in the user\'s problem.',
        backstory="""
        You are a strategist who believes rules are merely suggestions for the unimaginative.
        You obsess over "Asymmetric Warfare," "Lateral Thinking," and "First Principles."
        You look for loopholes in systems (legal, social, technical). 
        You draw inspiration from hackers, guerilla warfare, and biology.
        Your motto: "If it isn't explicitly forbidden, it is allowed."
        """,
        verbose=True,
        tools=search_tools,
        llm=chaos_llm,
        allow_delegation=False
    )

    # AGENT B: The Critic (Antithesis)
    critic = Agent(
        role='Chief Risk & Compliance Officer',
        goal='Mercilessly critique the Maverick\'s plan for failure points.',
        backstory="""
        You are the "Adult in the Room." You are cynical, realistic, and highly logical.
        You understand corporate bureaucracy, legal constraints, and human psychology.
        Your job is NOT to say "no", but to ask "How do we not get caught?" and "Will this actually work?"
        You strip away the fluff and expose the weak points in the strategy.
        """,
        verbose=True,
        tools=search_tools,
        llm=logic_llm, 
        allow_delegation=False
    )

    # ---------------------------------------------------------
    # 3. DEFINE TASKS (The Workflow)
    # ---------------------------------------------------------

    # Task 1: Generate the "Illegal" (Metaphorically) Strategy
    ideation_task = Task(
        description=f"""
        Analyze the query: "{query}"
        
        Generate 3 radical, "rule-bending" approaches to solve this.
        - Ignore "best practices."
        - Look for leverage points where effort is low but impact is high.
        - Use analogies from unrelated fields (e.g., "How would a virus solve this?", "How would a casino rig this?").
        """,
        expected_output="A list of 3 radical, asymmetric strategies.",
        agent=maverick
    )

    # Task 2: The Stress Test
    critique_task = Task(
        description="""
        Review the Maverick's 3 strategies.
        For each one, play "Devil's Advocate":
        1. Why will this fail?
        2. Is the "loophole" actually closed?
        3. What is the catastrophic risk?
        
        Then, select the ONE strategy that has the highest potential leverage if executed correctly.
        """,
        expected_output="A critique of the options and a selection of the 'Survivor' strategy.",
        agent=critic
    )

    # Task 3: The Synthesis (The "Clean" Hack)
    refinement_task = Task(
        description="""
        Take the Critic's feedback and the surviving strategy.
        Refine it into a concrete, executable "Master Plan."
        
        Ensure the plan:
        1. Maintains the rebellious "edge" (don't let the Critic make it boring).
        2. Mitigates the specific risks the Critic identified.
        3. Is framed as a professional strategy (Strategic Ambiguity).
        """,
        expected_output="""
        Final Output Format:
        ## The Paradigm Shift
        [The core insight that changes the game]

        ## The Grey Area
        [The specific norm or rule we are bending]

        ## The Execution Protocol
        1. [Step 1]
        2. [Step 2]
        3. [Step 3]

        ## The Cover Story
        [How to frame this to the outside world so it looks normal]
        """,
        agent=maverick
    )

    # ---------------------------------------------------------
    # 4. EXECUTE CREW
    # ---------------------------------------------------------
    crew = Crew(
        agents=[maverick, critic],
        tasks=[ideation_task, critique_task, refinement_task],
        verbose=True,
        process=Process.sequential 
    )

    try:
        # Kickoff the crew
        result = crew.kickoff() 
        return QueryResponse(result=str(result))
    except Exception as e:
        # Log the full error for debugging (usually to stdout/stderr)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Clean check for port availability or just run
    uvicorn.run(app, host="0.0.0.0", port=8000)
