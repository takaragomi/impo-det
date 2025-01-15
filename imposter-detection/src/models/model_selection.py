import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model.eval()
    return model, tokenizer

if __name__ == "__main__":
    model_name = "elyza/Llama-3-ELYZA-JP-8B"
    model, tokenizer = load_model(model_name)