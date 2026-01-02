import ollama
from pprint import pprint
from langchain_community.llms import Ollama

response = ollama.generate(model="gemma3:1b", 
                           prompt="What is an LLM?")

pprint(response['response'])

llm = Ollama(model="gemma3:1b")
response = llm.invoke("What is an LLM?")
print(response)