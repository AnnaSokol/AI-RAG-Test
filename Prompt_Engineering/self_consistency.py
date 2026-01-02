#%% packages
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from pprint import pprint
import re

#%% function for Chain-of-Thought Prompting (but only final answer)
def chain_of_thought_prompting(prompt: str, model_name: str = "gemma3:1b") -> str:
    model = ChatOllama(model=model_name, temperature=0.7)
    prompt_tmpl = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Be precise and concise."),
        ("user", prompt + "\n\nThink step by step, but ONLY output the final equation (one line).")
    ])
    chain = prompt_tmpl | model
    return chain.invoke({}).content.strip()

def extract_equation(text: str) -> str:
    # Try to grab something that looks like an equation containing 3,4,6,8 and 24
    # This is a simple heuristic; good enough for this exercise.
    line = text.strip().splitlines()[-1]
    return line.strip()

# %% Self-Consistency CoT (local)
def self_consistency_cot(prompt: str, number_of_runs: int = 5, model_name: str = "gemma3:1b") -> str:
    res = []
    for i in range(number_of_runs):
        current_res = chain_of_thought_prompting(prompt, model_name=model_name)
        eq = extract_equation(current_res)
        print(f"Run {i+1}: {eq}")
        res.append(eq)

    res_concat = "; ".join(res)
    judge_prompt = (
        "You will get multiple candidate equations in <<>> separated by ;.\n"
        f"<<{res_concat}>>\n\n"
        "Task: Return ONLY the most common equation exactly as written. "
        "If none repeats, return the equation that is most likely correct for making 24. "
        "Output ONE equation only."
    )

    judge = ChatOllama(model=model_name, temperature=0.0)
    prompt_tmpl = ChatPromptTemplate.from_messages([
        ("system", "You are a strict judge. Output only the final equation."),
        ("user", judge_prompt)
    ])
    chain = prompt_tmpl | judge
    return chain.invoke({}).content.strip()

#%% Test
user_prompt = (
    "The goal of the Game of 24 is to use + - * / to combine four numbers and get 24. "
    "Numbers: 3, 4, 6, 8. You MUST use all four numbers exactly once. "
    "Return one correct equation that equals 24. Use parentheses if needed. "
    "Also check the equation is correct. Output only the final equation."
)

#%%
res = self_consistency_cot(prompt=user_prompt, number_of_runs=5, model_name="gemma3:1b")
pprint(res)
