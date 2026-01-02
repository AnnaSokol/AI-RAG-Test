#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

#%% model (LOCAL)
model = ChatOllama(
    model="gemma3:1b",   # oder: llama3.1:8b
    temperature=0.0
)

#%% first run
messages = [
    (
        "system",
        "You are an author and write a children's book. "
        "Respond short and concise. End your answer with a specific question "
        "that provides a new direction for the story."
    ),
    ("user", "A mouse and a cat are best friends."),
]

prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | model
output = chain.invoke({})
print("FIRST OUTPUT:\n", output.content)

# %% next run (PROMPT CHAINING)
messages.append(("ai", output.content))
messages.append(("user", "The dog is running after the cat."))

prompt = ChatPromptTemplate.from_messages(messages)
chain = prompt | model
output = chain.invoke({})
print("\nSECOND OUTPUT:\n", output.content)
