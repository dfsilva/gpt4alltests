from gpt4all import GPT4All
import os

model_path = "/Users/diegosilva/Library/Application Support/nomic.ai/GPT4All"

model = GPT4All("ggml-gpt4all-j-v1.3-groovy.bin", model_path=model_path, allow_download=False)
text_list = [
            "Stop sending me messages, I dont like!",
            "I love this product", 
             "This is terrible", 
             "Ok I will call you", 
             "no worries", 
             "I need to think about it", 
             "I think its ok, but could be better", 
             "Im confused",
             "I dont't know",
             "what I feel about it?"
             ]

for text in text_list:
    response = model.chat_completion([{"role":"user", 
                                       "content":f"describe the sentiment of the text in terms of having an agreement or satisfaction strictly as 'positive','neutral' or 'negative': {text}"}], streaming=False, verbose=False)
    sentiment = response['choices'][0]["message"]['content']
    # print("-----------------------------------------------------------------")
    print(f"{text}: {sentiment} \n")
    print("-----------------------------------------------------------------")