import os
import re
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

# Load .env and set API key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = "None"

# Define response model
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Set up Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-04-17")

# Output parser
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use necessary tools. 
            Wrap the output in this format and provide no other text:\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

# Define tools
tools = [search_tool, wiki_tool, save_tool]

# Create agent executor
agent = create_tool_calling_agent(llm=llm, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Get query input
query = input("What can I help you research? ")

# Invoke agent
raw_response = agent_executor.invoke({"query": query})

# Parse response
try:
    output_text = raw_response.get("output", "")
    
    # Extract JSON from markdown-wrapped output
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", output_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = output_text  # fallback if not markdown-wrapped

    structured_response = parser.parse(json_str)
    print("\n✅ Parsed Response:")
    print(structured_response)

except Exception as e:
    print("\n❌ Error parsing response:", e)
    print("🔎 Raw Response:", raw_response)
