def get_banned_ngram_tokens(token_ids, n):
    if len(token_ids) < n:
        return []
    prefix = tuple(token_ids[-(n - 1):])
    banned = []
    for i in range(len(token_ids) - n + 1):
        ngram = tuple(token_ids[i : i + n])
        if ngram[:-1] == prefix:
            banned.append(ngram[-1])
    return banned

# Test
# "de exemplu de exemplu" -> tokens [10, 20, 10, 20]
# n = 3 -> prefix is [10, 20] -> seen was [10, 20, 10] -> banned is 10!
tokens = [1, 2, 3, 10, 20, 10, 20]
banned = get_banned_ngram_tokens(tokens, 3)
print("Banned tokens:", banned)
assert 10 in banned
print("Assertion passed!")
