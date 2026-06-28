from tokenizer.tokenizer import CharacterTokenizer

tokenizer = CharacterTokenizer()

tokenizer.train("banana")

print(tokenizer.stoi)
print(tokenizer.itos)

encoded = tokenizer.encode("banana")
print(encoded)

decoded = tokenizer.decode(encoded)
print(decoded)


print(tokenizer.encode("cat"))