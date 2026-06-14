from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

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

def batch_execution():
    """Demonstrate batch execution for multiple inputs."""
    prompt = ChatPromptTemplate.from_template("Translate to French: {text}")

    model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.7)
    parse = StrOutputParser()
    chain = prompt | model | parse

    # Batch - run with multiple inputs
    inputs = [
        {"text": "Hello, how are you?"},
        {"text": "What is your name?"},
        {"text": "Where is the nearest restaurant?"},
    ]

    results = chain.batch(inputs)

    print(results)

    for text in zip(inputs, results):
        # print(text)
        # print("\n")
        print(f"Input: {text[0]['text']} => Output {text[1]}")
        print("\n")

    # results1 = chain.batch_as_completed(inputs)

    # print(results1.)

    # for text in zip(inputs, results1):
    #     print(text)
    #     print("\n")
    #     # print(f"Input: {}")


def streaming():
    """Demonstrate streaming for real-time output."""
    prompt = ChatPromptTemplate.from_template("Write a haiku about: {topic}")

    model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.7)
    parse = StrOutputParser()
    chain = prompt | model | parse

    # Streaming - run with streaming enabled
    print("Streaming Output: ")
    for chunk in chain.stream({"topic": "nature"}):
        print(chunk, end="", flush=True)
    print()  # for newline after streaming

def schema_inspection():
    """Demonstrate input/output schema inspection."""
    prompt = ChatPromptTemplate.from_template("Summarize the following text: {text}")

    model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.7)
    parse = StrOutputParser()
    chain = prompt | model | parse

    # Inspect input and output schemas
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    print(f"Input Schema: {input_schema}")
    print(f"Output Schema: {output_schema}")


def exercise():
    """
    EXERCISE: Create a chain that:
    1. Takes a product name and target audience
    2. Generates a marketing tagline
    3. Returns just the tagline as a string

    Test with: product="AI Course", audience="developers"
    """

    # YOUR CODE HERE
    prompt = ChatPromptTemplate.from_template(
        "Create a marketing tagline for a product named '{product}' targeting '{audience}'."
    )
    model = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.7)
    parse = StrOutputParser()
    chain = prompt | model | parse

    result = chain.invoke({"product": "AI Course", "audience": "developers"})

    #print(result)
    #print(chain)
    print(f"Marketing Tagline: {result}")

if __name__ == "__main__":
    #basic_chain()

    #batch_execution()

    # streaming()

    #schema_inspection()

    exercise()