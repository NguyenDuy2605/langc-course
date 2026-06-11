from dotenv import load_dotenv
from langchain_core import __version__ as core_version
from langgraph import version as graph_version
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
