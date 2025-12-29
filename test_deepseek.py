import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path

# Force load env
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

key = os.environ.get("DEEPSEEK_API_KEY")
print(f"Key loaded: {bool(key)}")
if key:
    print(f"Key prefix: {key[:5]}...")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=key,
    base_url="https://api.deepseek.com", # Try without /v1 first, then with if fails
    temperature=0.9
)

print("\n--- Testing Invoke ---")
try:
    response = llm.invoke("Hello, are you DeepSeek?")
    print(f"Response: {response.content}")
except Exception as e:
    print(f"ERROR: {e}")
