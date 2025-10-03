from flask import Flask, request, jsonify
from main import predict_image  # import your prediction function

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    file.save("temp.jpg")  # save uploaded image
    result = predict_image("temp.jpg")
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)
