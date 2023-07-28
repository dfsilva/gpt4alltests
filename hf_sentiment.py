from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")

text_list = ["Stop sending me messages, I dont like!",
            "I love this product", 
             "This is terrible", 
             "Ok I will call you", 
             "no worries", 
             "I need to think about it", 
             "I think its ok, but could be better", 
             "Im confused",
             "I dont't know",
             "what I feel about it?"]
result = sentiment_pipeline(text_list)

print('------------\n\n')
for i in range(len(text_list)):
    print(f"{text_list[i]}: {result[i]}")