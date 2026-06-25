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

tokenizer = BPETokenizer()

tokens = tokenizer.train(
    "banana",
    num_merges=10
)

print(tokens)
print(tokenizer.merges)