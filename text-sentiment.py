import gpt4all

model = gpt4all.GPT4All("ggml-gpt4all-j-v1.3-groovy.bin", model_path='/workspaces/gpt4alltests/', allow_download=False)
text_list = ["I love this product", "This is terrible", "Ok I will call you", "no worries", "I need to think about it"]

for text in text_list:
    response = model.chat_completion([{"role":"user", "content":f"Evaluate the sentiment of the text and just answer using the words positive and negative or neutral: {text}"}], streaming=False, verbose=False)
    sentiment = response['choices'][0]["message"]['content']
    print(f"{text}: {sentiment} \n")