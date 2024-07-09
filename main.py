from flask import Flask, request, jsonify
from model import ask_question

app = Flask(__name__)

@app.route('/answer', methods=['POST'])
def answer():
    data = request.json
    question = data['question']
    answer = ask_question(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)