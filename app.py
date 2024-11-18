from flask import Flask, request, jsonify, render_template, redirect, url_for
from predict import predict_image, predict_video
import os

app = Flask(__name__)

# Temporary directory for uploaded files
TEMP_DIR = os.path.join(os.getcwd(), "temp")
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

# Route for the homepage
@app.route("/")
def home():
    return '''
    <!doctype html>
    <title>DeepFake Detection</title>
    <h1>Upload an Image or Video</h1>
    <form action="/predict" method="post" enctype="multipart/form-data">
        <p>Type: 
            <select name="type">
                <option value="image">Image</option>
                <option value="video">Video</option>
            </select>
        </p>
        <p><input type="file" name="file"></p>
        <p><input type="submit" value="Upload"></p>
    </form>
    '''

# Route for file upload and prediction
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    file_type = request.form.get("type")  # 'image' or 'video'

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    file_path = os.path.join(TEMP_DIR, file.filename)
    file.save(file_path)

    if file_type == "image":
        result = predict_image(file_path)
    elif file_type == "video":
        result = predict_video(file_path)
    else:
        return jsonify({"error": "Invalid file type"}), 400

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
