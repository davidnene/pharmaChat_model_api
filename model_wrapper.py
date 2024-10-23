from templates import system_prompt, prompt_template, query_wrapper_prompt
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

offload_dir ="/model_offload"

# Load model from huggingface
config = PeftConfig.from_pretrained("davidnene/meditron-pharmachat-ft")
base_model = AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-7b")
model = PeftModel.from_pretrained(base_model, "davidnene/meditron-pharmachat-ft")
tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
print('Model loaded')
# Wrap the fine-tuned model in HuggingFaceLLM
llm = HuggingFaceLLM(
    context_window=2048,
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.7, "do_sample": False,
                    #  "max_length": 600,
                      "top_p": 0.95,
                      "top_k": 50,},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="davidnene/meditron-pharmachat-ft",
    model_name="davidnene/meditron-pharmachat-ft",
    # tokenizer_name = tokenizer,
    # model_name=model,
    device_map="auto",
    stopping_ids=[50278, 50279, 50277, 1, 0],
    tokenizer_kwargs={"max_length": 4096},
    model_kwargs={
        "torch_dtype": torch.float16,
        "offload_folder": "/model_offload"}
)