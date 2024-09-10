import os
import re
import cv2
import imagehash
import pytesseract
import numpy as np
from PIL import Image
from datetime import datetime
# Class names corresponding to the YOLO model's output
class_names = ['dob', 'exp_date', 'name', 'address', 'sex', 'issue_date', 'face', 'license_number']
# class_names = ['dob', 'first_name', 'last_name']
# Process the uploaded image
def process_image(image_path, model):
    try:
        if not model:
            print("Model not initialized.")
            return None

        ocr_results = {}
        image = cv2.imread(image_path)
        predictions = model(image_path)

        """ Extract predictions (1) last col of pred_matrix containing class label for each detected object
        : selects all rows (each row corresponds to a detected object
        -1 selects the last column, which contains the predicted class labels
        (2) [:, :-1] extracts everything but the last column of the matrix, containing the bounding box coordinates
        and the confidence scores. [r, c] ~ : all rows, :-1 all except the last c
        converted to numpy arrays
        labels: This will contain a 1D NumPy array with the class labels for each detected object 
                (e.g., [0, 2, 1, ...], where each number represents a different class such as "person", "car", etc.).

        cords: This will contain a 2D NumPy array where each row corresponds to a detected object's bounding box 
                and confidence score (e.g., [[x1, y1, x2, y2, confidence], [x1, y1, x2, y2, confidence], ...]).
        """
        labels, cords = predictions.xyxyn[0][:, -1].numpy(), predictions.xyxyn[0][:, :-1].numpy()
        x_shape, y_shape = image.shape[1], image.shape[0]

        # iterates over all detected objects (face, name, license_number, etc.) in the image
        for i in range(len(labels)):
            # i represents the index of the detected object in the list of cords and labels
            # cords[i] extracts bounding boxes for the i-th detected object [x1, y1, x2, y2, confidence]
            row = cords[i]
            # adds the class_label to the end of the row array
            row = np.append(row, labels[i])

            # Extract coordinates and the label
            # multiply the normalized coordinates [0,1] by the image pixel resolution dimensions for pixel coordinates
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            class_name = class_names[int(row[5])]

            # Process the entity or face
            if class_name == "face":
                face_path = save_face(image, x1, y1, x2, y2, image_path)
                ocr_results[class_name] = {
                    "text": face_path,
                    "image_path": face_path
                }
            else:
                # Save the cropped image box before OCR
                cropped_image_path = save_cropped_box(image, x1, y1, x2, y2, image_path, class_name)

                # Extract and post-process the text for each field
                cleaned_text = extract_and_clean_text(image, x1, y1, x2, y2, class_name)

                # Store both the cropped image and the extracted text in ocr_results
                ocr_results[class_name] = {
                    "text": cleaned_text,
                    "image_path": cropped_image_path
                }


        return ocr_results

    except Exception as e:
        print(f'Error processing image: {e}')
        return None



def preprocess_image(image):
    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    # image = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply thresholding to binarize the image
    # _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return image


# Extract text from the cropped image region
def extract_text(image, x1, y1, x2, y2, field_name):
    try:
        roi = image[y1:y2, x1:x2]
        preprocessed_roi = preprocess_image(roi)

        # Save the preprocessed image to check what Tesseract is processing
        preprocessed_debug_path = f'./static/debug_preprocessed_{x1}_{y1}.jpg'
        cv2.imwrite(preprocessed_debug_path, preprocessed_roi)
        print(f"Saved preprocessed region for debugging: {preprocessed_debug_path}")

        # Save the cropped original ROI for comparison
        cropped_debug_path = f'./static/debug_cropped_{x1}_{y1}.jpg'
        cv2.imwrite(cropped_debug_path, roi)
        print(f"Saved cropped region for debugging: {cropped_debug_path}")

        # Fine-tuning config: {PSM: "Page Segmentation Mode", OEM: "OCR Engine Mode"}
        # custom_config = r'--oem 1 --psm 6'
        # config = custom_config
        # Fine-tuning config based on the field
        if field_name == 'license_number':
            custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789'
        elif field_name == 'name':
            custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz \'."'
        elif field_name in ['dob', 'exp_date']:
            custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=0123456789/'
        elif field_name == 'sex':
            custom_config = r'--oem 1 --psm 6 -c tessedit_char_whitelist=MF'
        elif field_name == 'address':
            custom_config = r'--oem 1 --psm 4'
        else:
            custom_config = r'--oem 1 --psm 6'
        data = pytesseract.image_to_data(preprocessed_roi, config=custom_config, output_type=pytesseract.Output.DICT)

        # Extract text and confidence levels
        text = ' '.join(data['text'])
        confidence = data['conf']
        print(f"Extracted Text: {text}, Confidence Levels: {confidence}")

        # Post-process the text for additional cleanup
        return post_process_text(field_name, text.strip())
    except Exception as e:
        print(f'Error extracting text: {e}')
        return ''

def post_process_text(field_name, text):
    if field_name == 'dob' or field_name == 'exp_date':
        # Regex for dates (e.g., MM/DD/YYYY)
        match = re.search(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', text)
        if match:
            return match.group(0)
    elif field_name == 'license_number':
        # Filter out anything that isn't a digit
        return ''.join(filter(str.isdigit, text))
    elif field_name == 'name':
        # Example: capitalize each word in the name
        return re.sub(r'[^a-zA-Z\s]', '', text).title().strip()
    elif field_name == 'address':
        # Basic cleaning of common OCR errors
        return text.replace('|', '').replace('~', '').replace('\n', ' ')
    elif field_name == 'sex':
        # Match against known values (e.g., "M" or "F")
        match = re.search(r'\b(M|F)\b', text, re.IGNORECASE)
        if match:
            return match.group(0).upper()

    # Return the cleaned text or the original if no cleaning applied
    return text

# Use this in the processing pipeline
def extract_and_clean_text(image, x1, y1, x2, y2, field_name):
    raw_text = extract_text(image, x1, y1, x2, y2, field_name)
    print(f"--\n\n-----field name detected: {field_name}, \nraw text extracted: {raw_text}-----\npost-processed text: {post_process_text(field_name, raw_text)}-----\n\n")
    return post_process_text(field_name, raw_text)

# Save the detected face region
def save_face(image, x1, y1, x2, y2, image_path):
    try:
        roi = image[y1:y2, x1:x2]
        face_filename = os.path.splitext(os.path.basename(image_path))[0] + '_face.jpg'
        face_path = os.path.join('./static', face_filename)
        cv2.imwrite(face_path, roi)
        return face_filename
    except Exception as e:
        print(f'Error saving face image: {e}')
        return None


def save_cropped_box(image, x1, y1, x2, y2, image_path, field_name):
    try:
        # Extract the region of interest (ROI) from the image
        roi = image[y1:y2, x1:x2]

        # Create a filename for the cropped image box
        cropped_filename = os.path.splitext(os.path.basename(image_path))[0] + f'_{field_name}_box.jpg'

        # Path to save the cropped image
        cropped_image_path = os.path.join('./static/cropped_boxes', cropped_filename)

        # Save the cropped image
        cv2.imwrite(cropped_image_path, roi)

        return cropped_image_path

    except Exception as e:
        print(f'Error saving cropped image: {e}')
        return None


# Create a folder based on today's date
def create_today_folder(folder):
    today = datetime.now()
    h = "00" if today.hour < 12 else "12"
    path = os.path.join(folder, today.strftime('%Y%m%d') + h)

    if not os.path.exists(path):
        os.makedirs(path)

    return path

# Store the processed result as an Excel file (if needed)
def store_excel(df):
    try:
        output_dir = "./api_output"
        folder = create_today_folder(output_dir)
        output_path = os.path.join(folder, 'final_output.xlsx')
        df.to_excel(output_path)
    except Exception as e:
        print(f'Error storing Excel file: {e}')
