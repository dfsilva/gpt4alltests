from flask import Flask, request
import gpt4all

app = Flask(__name__)
model = gpt4all.GPT4All("ggml-gpt4all-j-v1.3-groovy.bin", model_path='/Users/diegosilva/projects/gpt4alltests', allow_download=False)

@app.route('/answer', methods=['POST'])
def process_string():
    data = request.get_json()  # Get the JSON data from the request
    input_string = data['message']  # Extract the input string from the JSON data

    # Process the string (you can replace this with your own logic)
    response = model.chat_completion([{"role":"user", "content":input_string}], streaming=False, verbose=False)
    print(response)
    answer = response['choices'][0]["message"]['content']
    # Create a response JSON object
    response = {'answer': answer}
    return response
if __name__ == '__main__':
    app.run()