from dotenv import load_dotenv
import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import (AIMessage, ChatMessage, HumanMessage, SystemMessage, ToolMessage)

load_dotenv()

# # # chatprompttemplate
# prompt = ChatPromptTemplate.from_template("Tell me a {adjective} joke about {topic}.")

# # # format and inspect
# messages = prompt.format_messages(adjective="funny", topic = "chickens")
# print(messages)



# multi-message templates
# prompt_multi = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful assistant that translates {input_language} to {output_language}.",
#         ),
#         ("human", "Translate the following text: {text}"),
#     ]
# )

# messages_multi = prompt_multi.format_messages(
#     input_language="English", output_language="French", text="I love programming."
# )

# print(messages_multi)

# model = init_chat_model(model="claude-sonnet-4-6", temperature = 0.7)

# response = model.invoke(messages_multi)

# print(response.content)

# messages = [
#     HumanMessage(content="Hello!"),
#     AIMessage(content="Hi there! How can I assist you today?"),
#     SystemMessage(content="This is a system message."),
#     ToolMessage(content="Tool executed successfully.", tool_call_id="call_123"),
#     ChatMessage(content="This is a general chat message."),
# ]

# Fewshot example
examples = [
    {"input": "happy", "output": "sad"},
    {"input": "tall", "output": "short"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

fewshot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)


final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Give the opposite of each word."),
        fewshot_prompt,
        ("human", "{input}"),
    ]
)

model = init_chat_model(model="claude-sonnet-4-6", temperature = 0)

response = model.invoke(final_prompt.format_messages(input="happy"))

print(response.content)