from datasets import load_dataset
from transformers import AutoModelForCausalLM

dataset = load_dataset("nomic-ai/gpt4all-j-prompt-generations", revision="v1.2-jazzy")
model = AutoModelForCausalLM.from_pretrained("nomic-ai/gpt4all-j", revision="v1.2-jazzy")