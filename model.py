import os
import shutil
import time
import glob
import torch
from PIL import Image
import cv2
import random
import string
import numpy as np
import random
import traceback
from ultralytics import YOLO
import pandas as pd

def get_random_string(length):
    """
    Generate a random string of fixed length 

    Inputs
    ------
    length: int - length of the string to be generated

    Returns
    -------
    str - random string

    """
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
    return result_str

def load_model():
    """
    Load the model from the local directory
    """
    # model = YOLO("./pytorch-models/YOLOv8_Week9.pt")
    model = torch.hub.load('./yolov5', 'custom', path="./pytorch-models/Week9_ver1.pt", source='local')
    return model

def draw_own_bbox(img,x1,y1,x2,y2,label,color=(36,255,12),text_color=(0,0,0)):
    """
    Draw bounding box on the image with text label and save both the raw and annotated image in the 'own_results' folder

    Inputs
    ------
    img: numpy.ndarray - image on which the bounding box is to be drawn

    x1: int - x coordinate of the top left corner of the bounding box

    y1: int - y coordinate of the top left corner of the bounding box

    x2: int - x coordinate of the bottom right corner of the bounding box

    y2: int - y coordinate of the bottom right corner of the bounding box

    label: str - label to be written on the bounding box

    color: tuple - color of the bounding box

    text_color: tuple - color of the text label

    Returns
    -------
    None

    """
    name_to_id = {
        "NA": 'NA',
        "Bullseye": 10,
        "One": 11,
        "Two": 12,
        "Three": 13,
        "Four": 14,
        "Five": 15,
        "Six": 16,
        "Seven": 17,
        "Eight": 18,
        "Nine": 19,
        "A": 20,
        "B": 21,
        "C": 22,
        "D": 23,
        "E": 24,
        "F": 25,
        "G": 26,
        "H": 27,
        "S": 28,
        "T": 29,
        "U": 30,
        "V": 31,
        "W": 32,
        "X": 33,
        "Y": 34,
        "Z": 35,
        "Up": 36,
        "Down": 37,
        "Right": 38,
        "Left": 39,
        "Up Arrow": 36,
        "Down Arrow": 37,
        "Right Arrow": 38,
        "Left Arrow": 39,
        "Stop": 40
    }
    # Reformat the label to {label name}-{label id}
    label = label + "-" + str(name_to_id[label])
    # Convert the coordinates to int
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    # Create a random string to be used as the suffix for the image name, just in case the same name is accidentally used
    rand = str(int(time.time()))

    # Save the raw image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(f"own_results/raw_image_{label}_{rand}.jpg", img)

    # Draw the bounding box
    img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
    FONT_SCALE = 2
    FONT_THICKNESS = 5
    # For the text background, find space required by the text so that we can put a background with that amount of width.
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
    # Print the text  
    img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 - h - h // 4), color, -1)
    img = cv2.putText(img, label, (x1, y1 - h // 4), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE,\
                      text_color, FONT_THICKNESS)
    # Save the annotated image
    cv2.imwrite(f"own_results/annotated_image_{label}_{rand}.jpg", img)


def predict_image(image, model: YOLO, signal):
    """
    Predict the image using the model and save the results in the 'runs' folder
    
    Inputs
    ------
    image: str - name of the image file

    model: torch.hub.load - model to be used for prediction

    signal: str - signal to be used for filtering the predictions

    Returns
    -------
    str - predicted label
    """
    try:
        # Load the image
        img = Image.open(os.path.join('uploads', image))

        # Predict the image using the model
        result = model.predict(img, conf=0.5, save=True, project='./runs/detect')[0]
        names = model.names
        boxes_array = result.boxes.numpy()
        df_results = pd.DataFrame({'cls' : boxes_array.cls, 'confidence' : boxes_array.conf,\
                                'xmin' : boxes_array.xyxy[:, 0], 'ymin' : boxes_array.xyxy[:, 1],\
                                'xmax' : boxes_array.xyxy[:, 2], 'ymax' : boxes_array.xyxy[:, 3],\
                                'bboxWt': boxes_array.xywh[:, 2], 'bboxHt': boxes_array.xywh[:, 3]})
        df_results['name'] = df_results['cls'].map(names)

        # Convert the results to a pandas dataframe and calculate the height and width of the bounding box and the area of the bounding box
        df_results['bboxArea'] = df_results['bboxHt'] * df_results['bboxWt']

        # Label with largest bbox height will be last
        df_results = df_results.sort_values('bboxArea', ascending=False)

        # Filter out Bullseye
        pred_list = df_results 
        pred_list = pred_list[pred_list['name'] != 'Bullseye']
        
        # Initialize prediction to NA
        pred = None

        if len(pred_list) == 1:
            pred = pred_list.iloc[0]

        # If more than 1 label is detected
        elif len(pred_list) > 1:
            # Sort the predictions by xmin
            pred_list.sort_values(by='xmin', inplace=True)

            # If signal is 'L', choose the first prediction in the list, i.e. leftmost in the image
            if signal == 'L':
                pred = pred_list.iloc[0]
            
            # If signal is 'R', choose the last prediction in the list, i.e. rightmost in the image
            elif signal == 'R':
                pred = pred_list.iloc[-1]
            
            # If signal is 'C', choose the prediction that is central in the image
            else:
                # Choosing one with largest area if none are central
                pred_list.sort_values(by='bboxArea', inplace=True) 
                pred = pred_list.iloc[-1]
        
        name_to_id = {
            "NA": 'NA',
            "Bullseye": 10,
            "One": 11,
            "Two": 12,
            "Three": 13,
            "Four": 14,
            "Five": 15,
            "Six": 16,
            "Seven": 17,
            "Eight": 18,
            "Nine": 19,
            "A": 20,
            "B": 21,
            "C": 22,
            "D": 23,
            "E": 24,
            "F": 25,
            "G": 26,
            "H": 27,
            "S": 28,
            "T": 29,
            "U": 30,
            "V": 31,
            "W": 32,
            "X": 33,
            "Y": 34,
            "Z": 35,
            "Up": 36,
            "Down": 37,
            "Right": 38,
            "Left": 39,
            "Up Arrow": 36,
            "Down Arrow": 37,
            "Right Arrow": 38,
            "Left Arrow": 39,
            "Stop": 40
        }

        if pred is not None:
            draw_own_bbox(np.array(img), pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax'], pred['name'])
            image_id = name_to_id.get(pred['name'], 'NA')
        else:
            image_id = 'NA'
        
        print(f"Final result: {image_id}")
        return image_id
    # If some error happened, we just return 'NA' so that the inference loop is closed
    except Exception as e:
        traceback.print_exc()
        return 'NA'

def predict_image_week_9(image, model):
    # Load the image
    img = Image.open(os.path.join('uploads', image))

    # result = model.predict(img, conf=0.5, save=True, project='./runs/detect')[0]
    # boxes_array = result.boxes.numpy()
    # Run inference
    result = model(img)
    df_results = result.pandas().xyxy[0]

    if len(df_results) == 0 or df_results.iloc[0]['confidence'] < 0.5:
    # if len(boxes_array) == 0 or boxes_array.conf[0] < 0.6:
        for i in [3, 6]:
            width, height = img.size   # Get dimensions
            new_width = width // i
            new_height = height // i
            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2
            # Crop the center of the image
            img_cropped = img.crop((left, top, right, bottom))

            result = model(img_cropped)
            df_results = result.pandas().xyxy[0]
            if len(df_results) > 0 and df_results.iloc[0]['confidence'] >= 0.5:
                img = img_cropped
                break
            # result = model.predict(img_cropped, conf=0.4, save=True, project='./runs/detect')[0]
            # boxes_array = result.boxes.numpy()
            # if len(boxes_array) > 0:
            #     break

        # img = np.asarray(img)
        # w, h = img.shape[:2]
        # M = w // 4
        # N = h // 4
        # tiles = [img[x:x+M,y:y+N] for x in range(0,w,M) for y in range(0,h,N)]

        # tiles = tiles[4:-4] + tiles[:4] + tiles[-4:]

        # for tile in tiles:
        #     img = Image.fromarray(tile)
        #     result = model(img)
        #     df_results = result.pandas().xyxy[0]
        #     if len(df_results) > 0 and df_results.iloc[0]['confidence'] >= 0.5:
        #         break
    result.save('runs')

    # Calculate the height and width of the bounding box and the area of the bounding box
    df_results['bboxHt'] = df_results['ymax'] - df_results['ymin']
    df_results['bboxWt'] = df_results['xmax'] - df_results['xmin']

    # names = model.names
    # boxes_array = result.boxes.numpy()
    # df_results = pd.DataFrame({'cls' : boxes_array.cls, 'confidence' : boxes_array.conf,\
    #                         'xmin' : boxes_array.xyxy[:, 0], 'ymin' : boxes_array.xyxy[:, 1],\
    #                         'xmax' : boxes_array.xyxy[:, 2], 'ymax' : boxes_array.xyxy[:, 3],\
    #                         'bboxWt': boxes_array.xywh[:, 2], 'bboxHt': boxes_array.xywh[:, 3]})
    # df_results['name'] = df_results['cls'].map(names)

    # Label with largest bbox height will be last
    df_results['bboxArea'] = df_results['bboxHt'] * df_results['bboxWt']
    df_results.sort_values(by='bboxArea', ascending=False, inplace=True)
    pred_list = df_results 
    pred = 'NA'
    # If prediction list is not empty
    if pred_list.size != 0:
        # Go through the predictions, and choose the first one with confidence > 0.5
        for _, row in pred_list.iterrows():
            if row['name'] != 'Bullseye':
                pred = row    
                break

        # Draw the bounding box on the image 
        if not isinstance(pred,str):
            draw_own_bbox(np.array(img), pred['xmin'], pred['ymin'], pred['xmax'], pred['ymax'], pred['name'])
        
    # Dictionary is shorter as only two symbols, left and right are needed
    name_to_id = {
        "NA": 'NA',
        "Bullseye": 10,
        "Right": 38,
        "Left": 39,
        "Right Arrow": 38,
        "Left Arrow": 39,
    }
    # Return the image id
    if not isinstance(pred,str):
        image_id = str(name_to_id[pred['name']])
    else:
        image_id = 'NA'
    return image_id


def stitch_image():
    """
    Stitches the images in the folder together and saves it into runs/stitched folder
    """
    # Initialize path to save stitched image
    imgFolder = 'runs'
    stitchedPath = os.path.join(imgFolder, f'stitched-{int(time.time())}.jpeg')

    # Find all files that ends with ".jpg" (this won't match the stitched images as we name them ".jpeg")
    imgPaths = glob.glob(os.path.join(imgFolder+"/detect/*/", "*.jpg"))
    # Open all images
    images = [Image.open(x) for x in imgPaths]
    # Get the width and height of each image
    width, height = zip(*(i.size for i in images))
    # Calculate the total width and max height of the stitched image, as we are stitching horizontally
    total_width = sum(width)
    max_height = max(height)
    stitchedImg = Image.new('RGB', (total_width, max_height))
    x_offset = 0

    # Stitch the images together
    for im in images:
        stitchedImg.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    # Save the stitched image to the path
    stitchedImg.save(stitchedPath)

    # Move original images to "originals" subdirectory
    for img in imgPaths:
        shutil.move(img, os.path.join(
            "runs", "originals", os.path.basename(img)))

    return stitchedImg

def stitch_image_own():
    """
    Stitches the images in the folder together and saves it into own_results folder

    Basically similar to stitch_image() but with different folder names and slightly different drawing of bounding boxes and text
    """
    imgFolder = 'own_results'
    stitchedPath = os.path.join(imgFolder, f'stitched-{int(time.time())}.jpeg')

    imgPaths = glob.glob(os.path.join(imgFolder+"/annotated_image_*.jpg"))
    imgTimestamps = [imgPath.split("_")[-1][:-4] for imgPath in imgPaths]
    
    sortedByTimeStampImages = sorted(zip(imgPaths, imgTimestamps), key=lambda x: x[1])

    images = [Image.open(x[0]) for x in sortedByTimeStampImages]
    width, height = zip(*(i.size for i in images))
    total_width = sum(width)
    max_height = max(height)
    stitchedImg = Image.new('RGB', (total_width, max_height))
    x_offset = 0

    for im in images:
        stitchedImg.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    stitchedImg.save(stitchedPath)

    return stitchedImg

