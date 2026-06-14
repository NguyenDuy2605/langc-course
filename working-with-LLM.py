from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def demo_init_chat_model():
    if os.getenv("ANTHROPIC_API_KEY"):
        claude = init_chat_model(
            model="claude-sonnet-4-6",
            temperature=0.7,  
            model_provider="anthropic",
            streaming = True, 
            max_retries = 3
        )

    response = claude.invoke("What is the capital of France? Answer in one word.")
    print(f"Response: {response.content}")


def model_comparison():
    prompt = "Explain recursion in one sentence."

    models = {
        "gpt-4o-mini": init_chat_model(
            model="gpt-4o-mini",
            temperature=0.7,
            streaming=False,
        ),
        "gpt-4o": init_chat_model(
            model="gpt-4o",
            temperature=0.7,
            streaming=False,
        ),
    }

    # add anthropic model if available
    if os.getenv("ANTHROPIC_API_KEY"):
        models["claude-sonnet-4-5-20250929"] = init_chat_model(
            model="claude-sonnet-4-5-20250929",
            model_provider="anthropic",
            temperature=0.7,
            streaming=False,
        )

    print(f"Prompt: {prompt}\n")

    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"Response from {model_name}: {response.content}\n")

def demo_messages():
    model = ChatAnthropic(model_name="claude-sonnet-4-6", temperature = 0.7)
    # using message objects (more control over roles)
    messages = [
        SystemMessage(content="You are a pirate. Always answer like a pirate."),
        HumanMessage(content="What's the weather like today?"),
    ]

    print("Using message objects:")
    print(f"Messages: {messages[0]} | {messages[1]}")

    response = model.invoke(messages)
    print(f"Response from the Pirate: {response.content}")

    # Multi-turn conversation using message objects
    messages.append(response)  # add model's response to the conversation
    #print(f"Continue: {messages}")
    messages.append(HumanMessage(content="What about tomorrow?"))

    print("\nMulti-turn conversation:")
    response = model.invoke(messages)
    print(f"Follow-up response from the Pirate: {response.content}")


if __name__ == "__main__":
    #demo_init_chat_model()
    #model_comparison()
    demo_messages()