# SentencePiece Unigram Tokenization

# Trains a SentencePiece Unigram tokenizer on corpus.txt, then uses it to tokenize and decode input.

from collections import Counter
from pathlib import Path
import sys
import sentencepiece as spm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Find the corpus beside this script, regardless of where the command is run.
corpus_path = Path(__file__).with_name("corpus.txt")

# Step 1: Train SentencePiece Unigram tokenizer.
# Keep the generated model and vocabulary beside this script.
model_prefix = str(corpus_path.with_name("my_tokenizer"))
spm.SentencePieceTrainer.train(
    input=str(corpus_path),
    model_prefix=model_prefix,
    vocab_size=40,
    model_type="unigram",
    hard_vocab_limit=False
)

# Step 2: Load the trained tokenizer
sp = spm.SentencePieceProcessor(
    model_file=f"{model_prefix}.model"
)

# Step 3: Get input sentence
text = input("Enter a sentence: ")  or "I love Learning python"

# Step 4: Generate SentencePiece tokens
tokens = sp.encode(text, out_type=str)

# Step 5: Generate token IDs
token_ids = sp.encode(text, out_type=int)

print("\n========== TOKENIZATION RESULT ==========")

print("Original Text:", text)
print("SentencePiece Tokens:", tokens)
print("Token IDs:", token_ids)
print("Total Tokens:", len(tokens))

# Step 6: Calculate individual token frequencies
token_frequency = Counter(tokens)

print("\n========== TOKEN FREQUENCIES ==========")

for token, frequency in token_frequency.items():
    print(token, ":", frequency)

# Step 7: Count neighboring tokens using the list shifted by one position.
pair_frequency = Counter(zip(tokens, tokens[1:]))

print("\n========== ADJACENT TOKEN PAIR FREQUENCIES ==========")

for pair, frequency in pair_frequency.items():
    print(pair, ":", frequency)

# Step 8: Decode token IDs
decoded_text = sp.decode(token_ids)

print("\n========== DECODED TEXT ==========")

print("Decoded Text:", decoded_text)
