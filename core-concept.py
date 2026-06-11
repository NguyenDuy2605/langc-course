from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


def basic_chain():
    """Demonstrates a basic chain using LCEL and Runnables."""

    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
    )
    model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.7)
    parse = StrOutputParser()

    # Compose with pipe operator
    chain = prompt | model | parse

    # Execute the chain with an input
    result = chain.invoke({"question": "What is the Langchain?"})

    print(f"Response: {result}")

    return chain


basic_chain()