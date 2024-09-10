# Importing essential libraries
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from werkzeug.utils import secure_filename
import socket
import pandas as pd
from custom.old_python_files.original_utils import *
# from custom.create_today_folder import create_today_folder
# from custom.storing import store_excel
# from custom.global_data import class_names
# import custom.initialize_run as initrun


# Define a function to check if the uploaded file has an allowed extension
def allowed_file(filename):
    # The file is allowed if it has an extension and the extension is in the allowed list
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Initialize the Flask application
app = Flask(__name__)

# Define constants for upload configuration
api_input = './api_input'  # Directory for input files
folder = create_today_folder(api_input)  # Folder creation based on the current date and time
UPLOAD_FOLDER = folder  # Directory to store uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}  # Allowed file extensions

# Configure the app's upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Get the host IP address
hostname = socket.gethostname()
IPAddr = socket.gethostbyname(hostname)

# Initialize any necessary components for the application (the Model)
# initrun.init()
init()

# Dictionary to store the processed image predictions
dictimages = {}


# Define the root route
@app.route("/", methods=["GET", "POST"])
def root_route():
    # This route simply instructs users to navigate to the /upload_file endpoint
    print("hello, please go to /upload_file")
    return redirect(url_for('upload_file'))


# Define the file upload route
@app.route("/upload_file", methods=["GET", "POST"])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request contains the file part
        if 'files' not in request.files:
            # If no file part is present, print an error message and redirect to the same URL
            print('Error occurred: no file is present')
            return redirect(request.url)

        files = request.files.getlist('files')  # Get the list of files from the request
        print(f"files received: {files}")

        for file in files:
            # If no file is selected, flash a message and redirect to the same URL
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)

            # If the file is allowed, process it
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)  # Secure the filename

                # Re-create the upload folder to ensure the correct folder structure is used
                folder = create_today_folder(api_input)
                UPLOAD_FOLDER = folder
                app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

                try:
                    # Save the file to the configured upload folder
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    print(f'file saved to: {file_path}')
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

                    # Run the custom processing on the saved file
                    print(f"running run() on file: {file}")
                    dictpred = run(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                    print(f"dictpred = {dictpred} \n\n\n ----------")

                    # Store the predictions in the dictimages dictionary
                    dictimages[filename] = dictpred.copy()
                except Exception as e:
                    print(e)

        # Convert the dictionary of predictions to a JSON response
        resp = jsonify(dictimages)
        print(resp)

        # Convert the dictionary of predictions to a DataFrame and store it as an Excel file
        final_dict_df = pd.DataFrame.from_dict(dictimages, orient='index', columns=class_names)
        store_excel(final_dict_df)

        # Set the response status code
        resp.status_code = 201

        # Clear the dictimages dictionary for the next upload
        dictimages.clear()

        # Render the result template with the DataFrame values
        return render_template('dle_result.html', column_names=final_dict_df.columns.values,
                               row_data=list(final_dict_df.values.tolist()),
                               link_column="face", zip=zip)

    # Render the home template if the request method is GET
    return render_template('dle_home.html', result=dictimages)


# Main entry point for the application
if __name__ == '__main__':
    # Enable debug mode for the app, useful during development
    # app.debug = True
    app.run(debug=True)
