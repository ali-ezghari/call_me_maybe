import json

from llm_sdk import Small_LLM_Model
import numpy as np

# with open("data.json", "r") as file:
#     data = json.load(file)

# s = "Hello world!"

# chars = sorted(list(set(s)))
# vocab_size = len(chars)

# print("".join(chars))
# print(vocab_size)


# chars = sorted(list(set("abcd")))

# # stoi = {ch: i for i, ch in enumerate(chars)}
# stoi = [i for i in enumerate(chars)]

# itos = {i: ch for i, ch in enumerate(chars)}

# print(stoi)

# disc = {1:2, 23:3}
# print(type(disc))

# encode = lambda s: [stoi[c] for c in s]
# decode = lambda l: "".join([itos[i] for i in l])

# print(encode(s))
# print(decode(encode(s)))


# a = 234
# for i in enumerate(a):
#     print(f"{i} ")


# def fn_add_numbers(a: int, b: int) -> int:
#     return a + b


def main() -> None:
    text = "What is the sum of 2 and 3?"
    model = Small_LLM_Model()

    tokens = model.encode(text)

    for token in tokens[0]:
        logits = model.get_logits_from_input_ids(token)
        print(f"token: [{token.item()}] => Logits: [{logits}]")

    # print(tokens[0])
    # print(tokens)
    # print(model.decode(tokens))
    # print("\n")
    print(f"Logits: {len(logits)}")


# for token_id in ids[0]:
#     print(token_id.item(), repr(model.decode([token_id.item()])))


# def main() -> None:
#     text = "hello xixaxos!"
#     model = Small_LLM_Model()

#     ids = model.encode(text)
#     print(ids)
#     print(model.decode(ids))

#     for token_id in ids[0]:
#         print(token_id.item(), repr(model.decode([token_id.item()])))


if __name__ == "__main__":
    main()
