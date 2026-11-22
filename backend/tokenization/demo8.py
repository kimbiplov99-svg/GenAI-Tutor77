#BPE Tokenization using Pre-trained GPT2 Tokenizer
from transformers import AutoTokenizer

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = input("Enter a sentence: ") or "I love learning python"

tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Original Sentence:",text)
print("BPE Tokens:",tokens)
print("Token IDs:",token_ids)
print("Length of tokens:",len(tokens))
print("Length of token IDs:",len(token_ids))
