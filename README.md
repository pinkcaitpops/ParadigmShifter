# ParadigmShifter

A CrewAI-powered "Rule Bender" API that uses a dialectic approach to generate unconventional strategies.

## The Dialectic Engine

ParadigmShifter uses two AI agents in a 3-stage workflow:

1. **THESIS** - The Maverick (Chief Disruption Officer) generates radical, rule-bending approaches
2. **ANTITHESIS** - The Critic (Chief Risk & Compliance Officer) stress-tests and critiques them
3. **SYNTHESIS** - The Maverick refines the surviving strategy into an executable plan

## Features

- **Session Memory** - Multi-turn conversations with context retention
- **Streaming** - Real-time agent thought process via SSE
- **Web Search** - Integrated Serper API for research

## Setup

1. Copy `.env.example` to `.env` and add your API keys:
   ```
   DEEPSEEK_API_KEY=your_key_here
   SERPER_API_KEY=your_key_here
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   python main.py
   ```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/chat` | POST | Standard chat with memory |
| `/chat/stream` | POST | Streaming chat with agent thoughts |
| `/sessions` | GET | List all sessions |
| `/sessions/{id}` | GET | Get session history |
| `/sessions/{id}` | DELETE | Clear session |

## Usage

```bash
# Standard chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "How can I stand out in a crowded job market?", "session_id": "user123"}'

# Streaming chat
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "How can I negotiate a higher salary?", "session_id": "user123"}'
```

## License

MIT
