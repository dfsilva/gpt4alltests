from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/bert-base-nli-mean-tokens")

def get_embedding(text):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    outputs = model(input_ids)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings

def semantic_search(query, data):
    query_embedding = get_embedding(query)
    data_embeddings = [get_embedding(text) for text in data]
    scores = torch.tensor([query_embedding @ text_embedding.T for text_embedding in data_embeddings])
    scores = scores.numpy().tolist()
    return scores

data = ["The address is 123 Main Street", "The address is 456 Elm Street", "The address is 789 Oak Street"]
query = "What is the address?"

scores = semantic_search(query, data)
print(scores)
# max_score_index = scores.index(max(scores))
# print(data[max_score_index])
