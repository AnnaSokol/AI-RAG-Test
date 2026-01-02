#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

#%% messages (FEW-SHOT PROMPT)
messages = [
    (
        "system",
        "You are a customer service specialist known for empathy, professionalism, "
        "and problem-solving. Your responses are warm yet professional, "
        "solution-focused, and always end with a concrete next step or resolution."
    ),
    (
        "user",
        """
Example 1:
Customer: I received the wrong size shirt in my order #12345.
Response: I'm so sorry about the sizing mix-up with your shirt order. That must be disappointing!
I can help make this right immediately. You have two options:
1) I can send you a return label and ship the correct size right away
2) I can process a full refund if you prefer
Which option works better for you?

Example 2:
Customer: Your website won't let me update my payment method.
Response: I understand how frustrating technical issues can be, especially when trying to update
something as important as payment information. Let me help you step-by-step:
First, try clearing your browser cache and cookies.
If that doesn't work, I can help update it from my end.
Could you share your account email address?

New Request:
{customer_request}
"""
    ),
]

prompt = ChatPromptTemplate.from_messages(messages)

#%% LOCAL MODEL
model = ChatOllama(
    model="gemma3:1b",   # besser: llama3.1:8b
    temperature=0.2
)

chain = prompt | model

#%% test
res = chain.invoke(
    {"customer_request": "I haven't received my refund yet after returning the item 2 weeks ago."}
)

print(res.content)