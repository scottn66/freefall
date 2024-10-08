from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import os
import socket
from custom.utils import process_image, create_today_folder
from werkzeug.utils import secure_filename
import torch

# Initialize the Flask application
app = Flask(__name__)

app.secret_key = os.urandom(24)

# Define constants
api_input = './api_input'
folder = create_today_folder(api_input)
UPLOAD_FOLDER = folder
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
model = None

# Configure the app's upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Get the host IP address
hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)

# Define a function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize the YOLO model
def init_model():
    global model
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path="./model_checkpoints/model_final.pt")
        print("Model loaded successfully")
    except Exception as e:
        print(f'Error loading model: {e}')
        model = None


# Define the root route that redirects to /upload_file
@app.route("/", methods=["GET"])
def root_route():
    return redirect(url_for('upload_file'))

# Define the file upload route
@app.route("/upload_file", methods=["GET", "POST"])
def upload_file():
    if model is None:
        init_model()

    if request.method == 'POST':
        # Check if the POST request contains the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the image using the function from utils.py
            result = process_image(file_path, model)

            # Handle errors in processing
            if not result:
                flash('An error occurred while processing the image.')
                return redirect(request.url)

            # Render the result template with the processed data
            return render_template('dle_result.html', result=result)

    return render_template('dle_home.html')

# Main entry point for the application
if __name__ == '__main__':
    app.run(debug=True)
