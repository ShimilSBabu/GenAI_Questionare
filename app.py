from flask import Flask, request, jsonify, render_template
from questionare import questionare_maker
from werkzeug.utils import secure_filename
import os
# from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
    # return render_template('index.html', content=)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()
    topic = num_questions = api_key = ""
    print(f'data : {data}')
    topic = data['topic']
    num_questions = data['num_questions']
    api_key = data['api_key']

    questions = questionare_maker(api_key = api_key, topic = topic, num_questions = num_questions)
    # print(questions)
    return jsonify({'prediction': questions})
    # return render_template('index.html', prediction=questions)


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if not exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=['POST'])
def upload_file():
    # print("upload start")
    if 'pdf' not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files['pdf']
    # print("file request")
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        # print("filename")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # print("filepath")
        file.save(filepath)
        # print("filesave")
        return jsonify({"message": f"File uploaded successfully: {filename}"}), 200
    else:
        return jsonify({"message": "Invalid file type, only PDFs allowed"}), 400


if __name__ == '__main__':
    app.run(debug=True)
