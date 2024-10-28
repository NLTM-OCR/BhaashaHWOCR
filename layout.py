import glob
import os
from os.path import basename, join

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

# from .models import BoundingBox, LayoutImageResponse, Region


def transform_predict_to_df(results: list, labeles_dict: dict) -> pd.DataFrame:
    """
    Transform predict from yolov8 (torch.Tensor) to pandas DataFrame.

    Args:
        results (list): A list containing the predict output from yolov8 in the form of a torch.Tensor.
        labeles_dict (dict): A dictionary containing the labels names, where the keys are the class ids and the values are the label names.
        
    Returns:
        predict_bbox (pd.DataFrame): A DataFrame containing the bounding box coordinates, confidence scores and class labels.
    """
    # Transform the Tensor to numpy array
    predict_bbox = pd.DataFrame(results[0].to("cpu").numpy().boxes.xyxy, columns=['xmin', 'ymin', 'xmax','ymax'])
    # Add the confidence of the prediction to the DataFrame
    predict_bbox['confidence'] = results[0].to("cpu").numpy().boxes.conf
    # Add the class of the prediction to the DataFrame
    predict_bbox['class'] = (results[0].to("cpu").numpy().boxes.cls).astype(int)
    # Replace the class number with the class name from the labeles_dict
    predict_bbox['name'] = predict_bbox["class"].replace(labeles_dict)
    return predict_bbox

def get_model_predict(model: YOLO, input_image: Image, save: bool = False, image_size: int = 1248, conf: float = 0.5, augment: bool = False) -> pd.DataFrame:
    """
    Get the predictions of a model on an input image.
    
    Args:
        model (YOLO): The trained YOLO model.
        input_image (Image): The image on which the model will make predictions.
        save (bool, optional): Whether to save the image with the predictions. Defaults to False.
        image_size (int, optional): The size of the image the model will receive. Defaults to 1248.
        conf (float, optional): The confidence threshold for the predictions. Defaults to 0.5.
        augment (bool, optional): Whether to apply data augmentation on the input image. Defaults to False.
    
    Returns:
        pd.DataFrame: A DataFrame containing the predictions.
    """
    # Make predictions
    predictions = model.predict(
        imgsz=image_size, 
        source=input_image, 
        conf=conf,
        save=save, 
        augment=augment,
        flipud= 0.0,
        fliplr= 0.0,
        mosaic = 0.0,
        device = [0 if torch.cuda.is_available() else "cpu"]
    )
    
    # Transform predictions to pandas dataframe
    predictions = transform_predict_to_df(predictions, model.model.names)
    return predictions


def crop_words(image_path, layout_file, word_folder):
    with open(layout_file, 'r') as f:
        a = f.read().strip().split('\n')
    a = [list(map(int, i.strip(' ,').split(','))) for i in a]
    a = [i for i in a if len(i) == 5]
    img = Image.open(image_path).convert('RGB')
    for idx,i in enumerate(a):
        img.crop((
            i[0], i[1],
            i[0]+i[2], i[1]+i[3]
        )).save(join(word_folder, f'{idx}.jpg'))


def sort_words(boxes):
    """Sort boxes - (x, y, x+w, y+h) from left to right, top to bottom."""
    boxes.sort(key=lambda box: box[1])
    mean_height = sum([y2 - y1 for _, y1, _, y2 in boxes]) / len(boxes)

    # boxes.view('i8,i8,i8,i8').sort(order=['f1'], axis=0)
    current_line = boxes[0][1]
    lines = []
    tmp_line = []
    for box in boxes:
        if box[1] > current_line + mean_height:
            lines.append(tmp_line)
            tmp_line = [box]
            current_line = box[1]            
            continue
        tmp_line.append(box)
    lines.append(tmp_line)

    for line in lines:
        line.sort(key=lambda box: box[0])

    return lines


def main(image_path, pretrained, output):
    model = YOLO(join(pretrained, 'layout.pt'))
    print(image_path)
    result = get_model_predict(
        model=model,
        input_image=image_path,
        conf=0.15,
        augment=False,
        image_size=1280,
        save=False
    )
    a = []
    for i in range(len(result)):
        x = int(result['xmin'][i])
        y = int(result['ymin'][i])
        w = int(result['xmax'][i]) - x
        h = int(result['ymax'][i]) - y
        a.append((x, y, x+w, y+h))
    a = sort_words(a)
    regions = []
    for lineno, line in enumerate(a):
        for word in line:
            regions.append(','.join(list(map(str, [
                word[0], word[1], word[2]-word[0], word[3]-word[1], lineno+1,
            ]))))
    outfile = join(output, 'layout.txt')
    with open(outfile, 'w', encoding='utf-8') as f:
        f.write('\n'.join(regions))
    word_folder = join(output, 'words')
    if not os.path.exists(word_folder):
        os.makedirs(word_folder)
    crop_words(image_path, outfile, word_folder)
    return word_folder
