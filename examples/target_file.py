from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer.encode("I'm happy", add_special_tokens=False)
output = tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
print(output)  # 输出: I'm happy
