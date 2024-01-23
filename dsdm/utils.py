from transformers import AutoTokenizer

def tokenizer_maker():
    name = f"EleutherAI/pythia-70m"
    revision = f"step143000"
    tokenizer = AutoTokenizer.from_pretrained(
        name,
        revision=revision,
        use_fast=True
    )

    return tokenizer