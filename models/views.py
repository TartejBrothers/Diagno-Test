from django.shortcuts import render, redirect
from django.http import HttpResponseServerError

import os
from django.conf import settings
from django.template.response import TemplateResponse
from django.utils.datastructures import MultiValueDictKeyError
from django.core.files.storage import FileSystemStorage
import numpy as np
import cv2
import pandas as pd
from keras.models import load_model  # Importing load_model from Keras
import pickle
import tensorflow as tf
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from .train import load_data, create_dir, tf_dataset
from .metrics import dice_loss, dice_coef, iou

with open("svm_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Importing required functions from your nlp module
from nlpmodel.nlp import (
    train_model,
    get_symptoms_dict,
    get_description_list,
    get_precaution_dict,
    tree_to_code,
)

# Load your trained model outside of the view function using Keras
# loaded_model = load_model("svm_model.h5")


# Custom FileSystemStorage to ensure unique file names
class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name


def index(request):
    message = ""
    prediction = ""
    fss = CustomFileSystemStorage()

    try:
        image = request.FILES["image"]
        _image = fss.save(image.name, image)

        path = os.path.join(settings.MEDIA_ROOT, _image)
        custom_img = cv2.imread(path, 0)  # Read the image in grayscale
        resized_custom_img = cv2.resize(custom_img, (200, 200))
        preprocessed_custom_img = (
            resized_custom_img.reshape(1, -1) / 255.0
        )  # Flatten and normalize the image

        # Make a prediction using the loaded model
        prediction = loaded_model.predict(preprocessed_custom_img)

        # Display the prediction
        print("The custom image is predicted as:", prediction[0])
        if prediction[0] == 0:
            condition = "No Tumour"
        else:
            condition = "Tumour"

        filename = _image
        return TemplateResponse(
            request,
            "index.html",
            {
                "message": message,
                "filename": filename,
                "image_url": fss.url(_image),
                "prediction": condition,
            },
        )

    except MultiValueDictKeyError:
        return TemplateResponse(
            request,
            "index.html",
            {"message": "No Image Selected"},
        )
    except Exception as e:
        return TemplateResponse(
            request,
            "index.html",
            {"message": str(e)},
        )


def lung_index(request):
    message = ""
    prediction = ""
    fss = CustomFileSystemStorage()

    H = 512
    W = 512
    try:
        image = request.FILES["image"]
        _image = fss.save(image.name, image)

        path = os.path.join(settings.MEDIA_ROOT, _image)
        np.random.seed(42)
        tf.random.set_seed(42)

        create_dir("results")

        with CustomObjectScope(
            {"iou": iou, "dice_coef": dice_coef, "dice_loss": dice_loss}
        ):
            model = tf.keras.models.load_model("lung_model.h5")

        dataset_path = "../lung-segmentation/MontgomerySet"
        (
            (train_x, train_y1, train_y2),
            (valid_x, valid_y1, valid_y2),
            (test_x, test_y1, test_y2),
        ) = load_data(dataset_path)

        for x, y1, y2 in tqdm(zip(test_x, test_y1, test_y2), total=len(test_x)):
            """Extracing the image name."""
            image_name = x.split("/")[-1]

            """ Reading the image """
            ori_x = cv2.imread(x, cv2.IMREAD_COLOR)
            ori_x = cv2.resize(ori_x, (W, H))
            x = ori_x / 255.0
            x = x.astype(np.float32)
            x = np.expand_dims(x, axis=0)

            """ Reading the mask """
            ori_y1 = cv2.imread(y1, cv2.IMREAD_GRAYSCALE)
            ori_y2 = cv2.imread(y2, cv2.IMREAD_GRAYSCALE)
            ori_y = ori_y1 + ori_y2
            ori_y = cv2.resize(ori_y, (W, H))
            ori_y = np.expand_dims(ori_y, axis=-1)  ## (512, 512, 1)
            ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)  ## (512, 512, 3)

            """ Predicting the mask. """
            y_pred = model.predict(x)[0] > 0.5
            y_pred = y_pred.astype(np.int32)

            """ Saving the predicted mask along with the image and GT """
            save_image_path = f"results/{image_name}"
            y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)

            sep_line = np.ones((H, 10, 3)) * 255

            cat_image = np.concatenate(
                [ori_x, sep_line, ori_y, sep_line, y_pred * 255], axis=1
            )
            cv2.imwrite(save_image_path, cat_image)

        filename = _image
        return TemplateResponse(
            request,
            "index.html",
            {
                "message": message,
                "filename": filename,
                "image_url": fss.url(_image),
                # "result_image_url": result_image_url,
                # "description_html": description_html,
            },
        )

    except MultiValueDictKeyError:
        return TemplateResponse(
            request,
            "index.html",
            {"message": "No Image Selected"},
        )
    except Exception as e:
        return TemplateResponse(
            request,
            "index.html",
            {"message": str(e)},
        )
