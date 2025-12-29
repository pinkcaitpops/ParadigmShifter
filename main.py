import os
import sys
import json
import queue
import threading
import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI(title="ParadigmShifter - The Rule Bender API")

# Enable CORS for PERCEPTION frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# SESSION MEMORY - Multi-turn conversation support
# =============================================================================
sessions: dict = {}  # session_id -> list of {role, content}

def get_session_history(session_id: str) -> str:
    """Get formatted conversation history for a session."""
    if session_id not in sessions:
        return ""
    history = sessions[session_id]
    return "\n".join([f"{turn['role']}: {turn['content']}" for turn in history])

def save_to_session(session_id: str, user_msg: str, assistant_msg: str):
    """Save a conversation turn to session memory."""
    if session_id not in sessions:
        sessions[session_id] = []
    sessions[session_id].append({"role": "user", "content": user_msg})
    sessions[session_id].append({"role": "assistant", "content": assistant_msg})

# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================
class QueryRequest(BaseModel):
    message: str
    session_id: str = "default"

class QueryResponse(BaseModel):
    result: str
    session_id: str

# =============================================================================
# CUSTOM SEARCH TOOL
# =============================================================================
class SearchTools:
    @staticmethod
    def search_internet(query: str):
        """Useful to search the internet about a given topic and return relevant results."""
        top_result_to_return = 4
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": top_result_to_return})
        headers = {
            'X-API-KEY': os.environ.get('SERPER_API_KEY', ''),
            'content-type': 'application/json'
        }
        if not headers['X-API-KEY']:
            return "Error: SERPER_API_KEY not found in environment variables."

        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            if response.status_code != 200:
                return f"Error: Search failed with status code {response.status_code}. {response.text}"

            results = response.json().get('organic', [])
            string = []
            for result in results:
                try:
                    string.append('\n'.join([
                        f"Title: {result['title']}", 
                        f"Link: {result['link']}",
                        f"Snippet: {result['snippet']}", 
                        "---"
                    ]))
                except KeyError:
                    next
            if not string:
                return "No relevant search results found."
            return '\n'.join(string)
        except Exception as e:
            return f"Error: Failed to perform search. {str(e)}"

    @staticmethod
    def tool():
        return Tool(
            name="Search the internet",
            func=SearchTools.search_internet,
            description="Useful to search the internet about a given topic and return relevant results."
        )

# =============================================================================
# OUTPUT CAPTURE - For streaming agent thoughts
# =============================================================================
class OutputCapture:
    """Captures stdout and sends lines to a queue for streaming."""
    def __init__(self, output_queue: queue.Queue):
        self.output_queue = output_queue
        self.original_stdout = sys.stdout
        
    def write(self, text):
        self.original_stdout.write(text)
        if text.strip():
            self.output_queue.put(text)
    
    def flush(self):
        self.original_stdout.flush()

# =============================================================================
# STRATEGY CREW FACTORY - The Dialectic Engine
# =============================================================================
def create_strategy_crew(query: str, history: str = ""):
    """
    Factory for creating the Strategy Crew (Maverick + Critic).
    
    THE DIALECTIC ENGINE (3-Stage Flow):
    - Stage 1: THESIS - The Maverick generates radical ideas
    - Stage 2: ANTITHESIS - The Critic stress-tests them
    - Stage 3: SYNTHESIS - The Maverick refines the survivor
    """
    
    # ---------------------------------------------------------
    # 1. DEFINE BRAINS (Hybrid Architecture)
    # ---------------------------------------------------------
    
    # The Maverick: DeepSeek V3 (Via OpenAI Protocol)
    # "DeepSeek is the new Maverick."
    chaos_llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.9
    )
    
    # The Critic: DeepSeek V3 (Logic Mode)
    # Using DeepSeek for everything since Google API is unstable.
    logic_llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.1
    )

    # Use Custom Search Tool
    search_tool = SearchTools.tool()
    
    # Inject history if present (for multi-turn conversations)
    context_prefix = ""
    if history:
        context_prefix = f"""
--- CONVERSATION HISTORY ---
{history}
--- END HISTORY ---

The user is continuing this conversation. Consider the previous context.

"""
    
    full_prompt = f"{context_prefix}{query}"

    # ---------------------------------------------------------
    # 2. DEFINE AGENTS
    # ---------------------------------------------------------
    
    # AGENT A: The Maverick (Thesis)
    # Goal: Find the "hack", the loophole, the unfair advantage.
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
        tools=[search_tool],
        llm=chaos_llm,
        allow_delegation=False
    )

    # AGENT B: The Critic (Antithesis)
    # Goal: Shoot down the Maverick's bad ideas and harden the good ones.
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
        tools=[search_tool],
        llm=logic_llm, 
        allow_delegation=False
    )

    # ---------------------------------------------------------
    # 3. DEFINE TASKS (The Workflow)
    # ---------------------------------------------------------

    # Task 1: Generate the "Illegal" (Metaphorically) Strategy
    ideation_task = Task(
        description=f"""
        Analyze the query: "{full_prompt}"
        
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
    # The Maverick gets the final word to polish the Critic's chosen path.
    # NOW INCLUDES FULL ANALYSIS from previous tasks for transparency.
    refinement_task = Task(
        description="""
        Take the Critic's feedback and the surviving strategy.
        Synthesize it into a SPECIFIC, ACTIONABLE breakdown.
        
        CRITICAL REQUIREMENTS:
        1. Be CONCRETE and SPECIFIC to the user's actual situation - no generic advice.
        2. Maintain the rebellious "edge" (don't let the Critic make it boring).
        3. Address the specific risks the Critic identified.
        4. Each step must be something the user can DO THIS WEEK.
        
        DO NOT give vague platitudes. Give specific tactics, scripts, and actions.
        
        IMPORTANT: Your output must include a COMPLETE SUMMARY of the entire analysis process.
        """,
        expected_output="""
        YOU MUST USE THIS EXACT FORMAT:
        
        ---
        # 🔥 MAVERICK'S INITIAL STRATEGIES
        
        **Strategy 1:** [Name]
        [Brief description of the first radical approach]
        
        **Strategy 2:** [Name]
        [Brief description of the second radical approach]
        
        **Strategy 3:** [Name]
        [Brief description of the third radical approach]
        
        ---
        # 🎯 CRITIC'S ANALYSIS
        
        **Strategy 1 Verdict:** [REJECTED/RISKY/VIABLE]
        - Why it fails: [One sentence]
        - Catastrophic risk: [One sentence]
        
        **Strategy 2 Verdict:** [REJECTED/RISKY/VIABLE]
        - Why it fails: [One sentence]
        - Catastrophic risk: [One sentence]
        
        **Strategy 3 Verdict:** [REJECTED/RISKY/VIABLE]
        - Why it fails: [One sentence]
        - Catastrophic risk: [One sentence]
        
        **🏆 SURVIVOR:** [Which strategy was selected and why in one sentence]
        
        ---
        # ✅ FINAL SYNTHESIS
        
        ## The Paradigm Shift
        [One sentence: The core insight that changes the game]

        ## The Grey Area  
        [One sentence: The specific norm, rule, or assumption we are exploiting]

        ## The Execution Protocol
        1. [SPECIFIC action with exact script/template if applicable]
        2. [SPECIFIC action with timeline]
        3. [SPECIFIC action with measurable outcome]

        ## Risk Mitigation
        [How to handle the risks the Critic identified]

        ## The Cover Story
        [Exact language to use when explaining this to others]
        """,
        agent=maverick
    )

    # ---------------------------------------------------------
    # 4. RETURN CREW
    # ---------------------------------------------------------
    return Crew(
        agents=[maverick, critic],
        tasks=[ideation_task, critique_task, refinement_task],
        verbose=True,
        process=Process.sequential  # Linear flow: Idea -> Critique -> Polish
    )

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
def read_root():
    return {"status": "ParadigmShifter Online", "version": "2.0"}

# -----------------------------------------------------------------------------
# Non-Streaming Chat Endpoint
# -----------------------------------------------------------------------------
@app.post("/chat", response_model=QueryResponse)
def run_agent(request: QueryRequest):
    """Standard chat endpoint with session memory."""
    query = request.message
    session_id = request.session_id
    
    # Get conversation history
    history = get_session_history(session_id)
    
    try:
        crew = create_strategy_crew(query, history)
        result = crew.kickoff()
        result_str = str(result)
        
        # Save to session memory
        save_to_session(session_id, query, result_str)
        
        return QueryResponse(result=result_str, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------------------------------------------------
# Streaming Chat Endpoint - Shows agent thoughts in real-time
# -----------------------------------------------------------------------------
def run_crew_with_capture(query: str, history: str, output_queue: queue.Queue, result_holder: list):
    """Run the crew in a thread, capturing stdout for streaming."""
    capture = OutputCapture(output_queue)
    sys.stdout = capture
    
    try:
        crew = create_strategy_crew(query, history)
        result = crew.kickoff()
        result_holder.append(str(result))
    except Exception as e:
        result_holder.append(f"Error: {str(e)}")
    finally:
        sys.stdout = capture.original_stdout
        output_queue.put(None)  # Signal completion

@app.post("/chat/stream")
async def run_agent_stream(request: QueryRequest):
    """Streaming endpoint that sends agent thoughts in real-time."""
    query = request.message
    session_id = request.session_id
    output_queue = queue.Queue()
    result_holder = []
    
    # Get session history for memory support
    history = get_session_history(session_id)

    # Start crew in background thread with history
    thread = threading.Thread(
        target=run_crew_with_capture, 
        args=(query, history, output_queue, result_holder)
    )
    thread.start()

    def generate():
        while True:
            try:
                line = output_queue.get(timeout=120)
                if line is None:  # Completion signal
                    break
                # Send as SSE event
                yield f"data: {json.dumps({'type': 'thinking', 'content': line})}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'error', 'content': 'Timeout'})}\n\n"
                break
        
        thread.join()
        final_result = result_holder[0] if result_holder else "No result"
        
        # Save to session memory
        save_to_session(session_id, query, final_result)
        
        yield f"data: {json.dumps({'type': 'result', 'content': final_result})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

# -----------------------------------------------------------------------------
# Session Management Endpoints
# -----------------------------------------------------------------------------
@app.get("/sessions")
def list_sessions():
    """List all active sessions."""
    return {"sessions": list(sessions.keys())}

@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Get conversation history for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": sessions[session_id]}

@app.delete("/sessions/{session_id}")
def clear_session(session_id: str):
    """Clear a session's conversation history."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared", "session_id": session_id}

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
