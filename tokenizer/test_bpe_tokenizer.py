from BPETokenizer import BPETokenizer

# tokens = ['b', 'a', 'n', 'a', 'n', 'a']

# print(get_pair_counts(tokens))

# tokens = ['b', 'a', 'n', 'a', 'n', 'a']

# print(
#     merge_pair(
#         tokens,
#         ('a', 'n')
#     )
# )

# counts = {
#     ('b', 'a'): 3,
#     ('a', 'n'): 6,
#     ('n', 'a'): 5
# }

# print(get_best_pair(counts))


# tokenizer = BPETokenizer()

# tokenizer.train(
#     "banana",
#     num_merges=10
# )

# tokenizer = BPETokenizer()

# tokens = tokenizer.train(
#     "banana",
#     num_merges=10
# )

# print(tokens)
# print(tokenizer.merges)

corpus = "banana bandana bananas"

tokenizer = BPETokenizer()

tokenizer.train(corpus, vocab_size=20)

# encoded = tokenizer.encode("banana")
# decoded = tokenizer.decode(encoded)

# print(encoded)
# print(decoded)

print(tokenizer.encode("bandana"))
print(tokenizer.encode("bananas"))
print(tokenizer.encode("apple"))

print(tokenizer.vocab)
print(tokenizer.merges)
print(tokenizer.stoi)
print(tokenizer.itos)