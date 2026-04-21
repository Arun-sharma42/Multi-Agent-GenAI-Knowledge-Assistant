import os

path = r'c:\Users\aruns\OneDrive\Documents\multi_agent_assistant\agents\router_agent.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Replace the block from raw = self.llm.invoke to just before return AgentResponse
import re

pattern = r'raw = self\.llm\.invoke\(messages\).*?self\.log\.info\(f"Routed .*?" \-> \{route\.upper\(\)\}\)'
replacement = """raw = self.llm.invoke(messages)
        content = raw.content
        if isinstance(content, list):
            content = "".join([c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in content])
        
        # Robust parsing
        res = str(content).lower()
        if "sql" in res:
            route = "sql"
        elif "rag" in res:
            route = "rag"
        else:
            route = "general"

        self.log.info(f"Routed '{query[:60]}' -> {route.upper()}")"""

new_text = re.sub(pattern, replacement, text, flags=re.DOTALL)

with open(path, 'w', encoding='utf-8') as f:
    f.write(new_text)
