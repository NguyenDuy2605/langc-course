from dotenv import load_dotenv
from importlib.metadata import version
core_version = version("langchain_core")
graph_version = version("langgraph")

from langchain_anthropic import ChatAnthropic

#print(f"Langchain core version: {core_version}")

load_dotenv()

def main():
    print("Hello from langc-course!")

    llm = ChatAnthropic(model_name="claude-sonnet-4-6", temperature=0)
    response = llm.invoke("Setup complete")
    print(f"Response from Claude: {response}")


if __name__ == "__main__":
    main()
