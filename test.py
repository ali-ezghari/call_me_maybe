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


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def main() -> None:
    text = "What is the sum of 2 and 3?"
    model = Small_LLM_Model()
    for text in [
        "What is the sum of 2 and 3?",
        "What is the sum of 2 and 3?\n",
        "What is the sum of 2 and 3? The answer is",
    ]:
        tokens = model.encode(text)
        logits = np.array(model.get_logits_from_input_ids(tokens[0].tolist()))
        token_id = np.argmax(logits)
        print(f"{text!r} -> {model.decode([token_id])}")



# def main() -> None:
#     text = "What is the sum of 2 and 3?"
#     model = Small_LLM_Model()

#     tokens = model.encode(text)
#     logits = model.get_logits_from_input_ids(tokens[0].tolist())
#     logits_array = np.array(logits)
    
    
#     next_token_id = np.argmax(logits_array)

#     top_ids = np.argsort(logits_array)[-10:][::-1]

#     for token_id in top_ids:
#         print(
#         token_id,
#         model.decode([token_id]),
#         logits_array[token_id]
#     )

    # 4. Decode the single token ID into text
    # print(f"Predicted Token ID: {next_token_id}")
    # print(f"Predicted Token Text: {model.decode([next_token_id])}")
    
    # print(type(logits))
    # print(np.array(logits).shape)
    # print(np.array(logits)[:10])
    
    
    
    # print(type(logits))
    # print(len(logits))
    # print(max(logits))

    # probs = softmax(logits)
    # probs = np.argmax(logits)

    # token_id = probs.argmax()
    # print(probs)
    # print(model.decode(probs.argmax()))
    


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
