from langchain import PromptTemplate, HuggingFaceHub, LLMChain

template = """Question: {question}
Answer: Let's think step by step.
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=HuggingFaceHub(repo_i))
