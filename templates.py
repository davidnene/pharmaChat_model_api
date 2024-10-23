from llama_index.core import PromptTemplate

prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

system_prompt = """<|SYSTEM|># pharmChatLLM (Meditron finetuned version)
- pharmChatLLM is a helpful and harmless open-source AI language model developed by David Nene.
- pharmChatLLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- pharmChatLLM is more than just an information source, StableLM is finetuned to generate medical drugs general info and dosage info.
- pharmChatLLM will refuse to participate in anything that could harm a human.
"""

query_wrapper_prompt = PromptTemplate(
    "Below is an instruction that describes a task. "
    "Use the following pieces of information to answer the user's question."
    "If you don't know the answer, just say that you don't know, don't try to make up an answer."

    "### Instruction:\n{query_str}\n\n### Only return the helpful answer below and nothing else. Helpful answer:"
)