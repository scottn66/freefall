#! usr/bin/python
from datetime import datetime
import os
import cv2
import json
# object detection libraries
import torch
from PIL import Image
import imagehash
import numpy as np

# from custom.runOCR import runOCR
# from custom.global_data import checkpoint_file, class_names
# from custom.storing import store_image, store_face

import pytesseract

# Adding custom options
# pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# tessdata_dir_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'
#
# custom_config=r'--psm 11 --oem 3 --tessdata-dir "/Users/scottnelson/Desktop/freefall_code/github_repo/freefall/src"'
# custom_config= r'--oem 3 --psm 11'

# custom_config =r'--psm 6 --oem 3'


# GLOBALS
checkpoint_file = r"../../model_checkpoints/model_final.pt"

# new class names
class_names = ['dob', 'exp_date', 'name', 'address', 'sex', 'issue_date', 'face', 'license_number']

# Storage
input_dir = './api_input'
output_dir = './api_output'

ocrdict = {}

# Debugging
debugger_enable = False


def runOCR(img, x1, y1, x2, y2, cname):
    try:
        roi = img[y1:y2, x1:x2]
        output = pytesseract.image_to_string(roi)
        if not output:
            print(f"OCR output is empty for {cname}.")
            return ''  # Return empty string if OCR doesn't detect anything
        print(f"OCR output before cleaning for {cname}: {output}")  # Log raw OCR output
        return output
    except Exception as e:
        print(e)
        return ''  # Return empty string if an exception occurs


predicts = []

def find_model_file(start_dir, filename):
    for root, dirs, files in os.walk(start_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def init():
    print("in init() trying to load model...")

    start_directory = "/Users/scottnelson/Desktop/wikiOntology/wiki/Flask_tutorial/"
    model_file = find_model_file(start_directory, "model_final.pt")

    if model_file:
        print(f"Model file found at: {model_file}")
        checkpoint_file = model_file  # Update the checkpoint_file to the correct path
    else:
        print("Model file not found in the specified directory tree.")
        return  # Exit the function if the model file is not found

    try:
        global model
        absolute_path = os.path.abspath(checkpoint_file)
        print(f"Looking for model file at: {absolute_path}")

        if not os.path.isfile(absolute_path):
            raise FileNotFoundError(f"Model file not found at {absolute_path}")

        model = torch.hub.load('ultralytics/yolov5', 'custom', path=absolute_path)
        print("Model loaded successfully")
    except Exception as e:
        print('Exception during model loading:')
        print(e)


def class_to_label(x):
    try:
        class_name = class_names[int(x)]
        return class_name
    except Exception as e:
        print(e)


def process_entity(image, x1, y1, x2, y2, acc, class_name):
    # Ensure that the coordinates are within the image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.shape[1], x2)
    y2 = min(image.shape[0], y2)

    roi = image[y1:y2, x1:x2]
    cv2.imwrite(f'./debug_{class_name}.jpg', roi)  # Save ROI for inspection

    output = runOCR(roi, x1, y1, x2, y2, class_name)
    if output is not None:
        output = output.replace("\n\f", '').replace("\n\n", '').replace("\n", '').replace("~", '').replace("|", '')
    else:
        output = ''  # Ensure output is a string even if OCR fails
    return output

def process_face(image, image_name, x1, y1, x2, y2, acc, class_name):
    stored_path = store_face(image, image_name, x1, y1, x2, y2, acc, class_name)
    return stored_path


def run(img):
    print(f'/initialize_run/run(img) Processing file: {img}')
    try:
        ocrdict = {}
        dict_cat = {}
        hashvalue = imagehash.average_hash(Image.open(img))
        image_name = os.path.basename(img)
        original = cv2.imread(img)
        image = cv2.imread(img)
        predictions = model(img)
        print(f"model predictions output directly from the model: {predictions}\n\n")
        labels, cord = predictions.xyxyn[0][:, -1].numpy(), predictions.xyxyn[0][:, :-1].numpy()
        print(f"the labels extracted from the predictor output of model(img): {labels}")
        x_shape, y_shape = image.shape[1], image.shape[0]
        classes_check = []
        predicts.clear()

        for i in range(len(labels)):
            row = cord[i]
            row = np.append(row, labels[i])
            if row[4] >= 0.1:
                x1, y1, x2, y2, acc, class_name = int(row[0] * x_shape), int(row[1] * y_shape), int(
                    row[2] * x_shape), int(row[3] * y_shape), row[4], class_to_label(row[5])
                print(f"Processing {class_name} with confidence {acc}")
                if class_name not in classes_check:
                    classes_check.append(class_name)
                    predicts.append([x1, y1, x2, y2, acc, class_name])
                    if class_name == "face":
                        facepath = process_face(image, image_name, x1, y1, x2, y2, acc, class_name)
                        ocrdict[class_name] = facepath
                    else:
                        output = process_entity(image, x1, y1, x2, y2, acc, class_name)
                        ocrdict[class_name] = output if output else 'Not detected'

        print(f"Final Predictions: {ocrdict}")
        store_image(img, original, image, image_name, hashvalue, predicts, ocrdict)

        return ocrdict

    except Exception as e:
        print(f"Error during processing: {e}")
        return None


def create_today_folder(folder):
    today = datetime.now()
    if today.hour < 12:
        h = "00"
    else:
        h = "12"

    path = folder + "/" + today.strftime('%Y%m%d') + h
    # for test purposes, using the following path: test!
    path = folder + "/test"
    if not os.path.exists(path):
        os.makedirs(path)

    return path


def get_predicts(result, class_names):
    predicts = []
    for res, cname in zip(result, class_names):
        r = list(res[0])
        r.append(cname)
        predicts.append(r)
    print(predicts)
    return predicts


def store_image(imgpath, original, image, image_name, hashvalue, predicts, ocrdict):
    try:
        output_dir = "./api_output"
        jsondata = json.dumps(ocrdict)
        folder = create_today_folder(output_dir)
        predictionpath = os.path.join(folder, os.path.splitext(image_name)[0] + '_output.jpg')
        cv2.imwrite(predictionpath, image)
        generate_xml(original, image, image_name, predicts, folder)
        # insert_table(hashvalue,imgpath,predictionpath,jsondata)
    except Exception as e:
        print(e)


def generate_xml(original, image, image_name, predicts, folder):
    try:
        desfile = os.path.splitext(image_name)[0] + '.xml'
        desdir = os.path.join(folder, desfile)
        h, w, c = image.shape
        with open(desdir, 'w') as f:
            content = '''<annotation>
					<folder>images</folder>
					<filename>{}</filename>
					<path>{}</path>
					<source>
						<database>Unknown</database>
					</source>
					<size>
						<width>{}</width>
						<height>{}</height>
						<depth>3</depth>
					</size>
					<segmented>0</segmented>
				'''

            f.write(content.format(image_name, os.path.join(folder, image_name), w, h))
            for pred in predicts:
                try:
                    x1, y1, x2, y2, class_name = int(pred[0]), int(pred[1]), int(pred[2]), int(pred[3]), pred[5]
                    content = '''
						<object>
							<name>{}</name>
							<pose>Unspecified</pose>
							<truncated>0</truncated>
							<difficult>0</difficult>
							<bndbox>
								<xmin>{}</xmin>
								<ymin>{}</ymin>
								<xmax>{}</xmax>
								<ymax>{}</ymax>
							</bndbox>
						</object>
						'''
                    f.write(content.format(class_name, x1, y1, x2, y2))
                except Exception as e:
                    print(e)

            f.write('\n\n</annotation>')
    except Exception as e:
        print(e)


def store_face(image, image_name, x1, y1, x2, y2, acc, class_name):
    try:
        output_dir = "./api_output"
        folder = create_today_folder(output_dir)
        roi = image[y1:y2, x1:x2]
        predictionpath = os.path.join(r'../../static', os.path.splitext(image_name)[0] + '_face_output.jpg')
        image_name = os.path.basename(predictionpath)
        print(predictionpath)
        cv2.imwrite(predictionpath, roi)
        return image_name
    except Exception as e:
        print(e)


def store_excel(df):
    try:
        output_dir = "./api_output"
        folder = create_today_folder(output_dir)
        outputpath = os.path.join(folder, 'final_output.xlsx')
        # df.to_excel(outputpath,index = False)
        df.to_excel(outputpath)
    except Exception as e:
        print(e)
