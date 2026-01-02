#%% packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel, Field
import re

#%% model (LOCAL)
chat_model = ChatOllama(
    model="gemma3:1b",  # oder llama3.1:8b
    temperature=0.2
)

#%% Pydantic model (nur zur Struktur, kein harter Parser)
class FeedbackResponse(BaseModel):
    rating: int = Field(..., description="Score in percent")
    feedback: str
    revised_output: str

#%% Self-feedback function
def self_feedback(user_prompt: str, max_iterations: int = 5, target_rating: int = 90):
    content = user_prompt
    feedback = ""

    for i in range(max_iterations):
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a strict reviewer. "
                "Evaluate the text about the American Civil War. "
                "Give a rating (0–100), feedback, and an improved version.\n\n"
                "Return EXACTLY in this format:\n"
                "Rating: <number>\n"
                "Feedback: <text>\n"
                "Revised: <text>"
            ),
            (
                "user",
                f"TEXT:\n{content}\n\nPREVIOUS FEEDBACK:\n{feedback}"
            )
        ])

        chain = prompt | chat_model
        response = chain.invoke({}).content

        print(f"\n--- Iteration {i} ---")
        print(response)

        #  Simple parsing (robust for local models)
        rating_match = re.search(r"Rating:\s*(\d+)", response)
        revised_match = re.search(r"Revised:\s*(.*)", response, re.S)

        if not rating_match or not revised_match:
            print("Could not parse response, stopping.")
            return content

        rating = int(rating_match.group(1))
        revised_output = revised_match.group(1).strip()

        content = revised_output
        feedback = response

        if rating >= target_rating:
            print(f"✅ Target rating reached: {rating}")
            return content

    return content

#%% Test
user_prompt = "The American Civil War was a civil war in the United States between the north and south."
res = self_feedback(user_prompt=user_prompt, max_iterations=3, target_rating=90)
print("\nFINAL RESULT:\n", res)
