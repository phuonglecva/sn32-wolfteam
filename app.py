import time
from argparse import ArgumentParser

from flask import Flask, request, jsonify

from validation import predict_texts


def parse():
    parser = ArgumentParser()
    parser.add_argument('--port', type=int, default=8080, help='Port to run the server on.')
    return parser.parse_args()


args = parse()

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Hello! I am model service"


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time_ns()
    if request.is_json:
        data = request.get_json()
        input_data = data['list_text']
        validator_hotkey = data['validator_hotkey']
        print(f'start predicting {len(input_data)} texts, validator_hotkey = {validator_hotkey}')
        distance_response = predict_texts(input_data, validator_hotkey)
        print(f"time loading {int(time.time_ns() - start_time):,} nanosecond")
        return jsonify({"message": "predict successfully", "result": distance_response}), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=args.port)
