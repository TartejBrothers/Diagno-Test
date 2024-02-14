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
import pickle as pkl
import sklearn
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from .train import load_data, create_dir, tf_dataset
from .metrics import dice_loss, dice_coef, iou

with open("svm_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

from django.shortcuts import render
from django.http import JsonResponse
import nltk

print("Downloading nltk packages")
print(nltk.__version__)
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download("punkt")
nltk.download("wordnet")

import random
import numpy as np
import json
import pickle
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

with open("intents.json") as json_file:
    intents = json.load(json_file)

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("chatbotmodel.h5")


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    result = "TMKCM Drift"
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


def chatbot(request):
    if request.method == "POST":
        message = request.POST.get("message")
        ints = predict_class(message)
        res = get_response(ints, intents)
        return JsonResponse({"response": res})
    else:
        return render(request, "chatbot.html")


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
    filename = ""  # Define filename outside the try-except block

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

        filename = _image  # Assign the value of _image to filename if the try block executes successfully

    except MultiValueDictKeyError:
        message = "No Image Selected"
    except Exception as e:
        message = str(e)

    return render(
        request,
        "index.html",
        {
            "message": message,
            "filename": filename,
            "image_url": (fss.url(filename) if filename else ""),
        },
    )


def pneumino_index(request):
    message = ""
    prediction = ""
    fss = CustomFileSystemStorage()
    loaded_model = load_model("pneumonia.h5")
    try:
        image = request.FILES["image"]
        _image = fss.save(image.name, image)
        path = os.path.join(settings.MEDIA_ROOT, _image)
        custom_img = cv2.imread(path, cv2.IMREAD_COLOR)
        resized_custom_img = cv2.resize(custom_img, (224, 224))
        preprocessed_custom_img = resized_custom_img.reshape(1, 224, 224, 3) / 255.0

        prediction = loaded_model.predict(preprocessed_custom_img)
        score = tf.nn.softmax(prediction[0])
        normal = f"{score[0]:.2%}"

        pneumonia = f"{score[1]:.2%}"
        if normal > pneumonia:
            condition = "Normal"
        else:
            condition = "Pneumonia"
        filename = _image
        return TemplateResponse(
            request,
            "pneumonia.html",
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
            "pneumonia.html",
            {"message": "No Image Selected"},
        )
    except Exception as e:
        return TemplateResponse(
            request,
            "pneumonia.html",
            {"message": str(e)},
        )


from django.shortcuts import render
from django.template.response import TemplateResponse
import pickle as pkl
from .models import HeartDiseasePrediction


def heart_index(request):
    if request.method == "POST":
        try:
            age = int(request.POST.get("age"))
            sex = bool(int(request.POST.get("sex")))
            cp = request.POST.get("cp")
            trestbps = int(request.POST.get("trestbps"))
            chol = int(request.POST.get("chol"))
            fbs = bool(int(request.POST.get("fbs")))
            restecg = request.POST.get("restecg")
            thalach = int(request.POST.get("thalach"))
            exang = bool(int(request.POST.get("exang")))
            oldpeak = float(request.POST.get("oldpeak"))
            slope = request.POST.get("slope")
            ca = int(request.POST.get("ca"))
            thal = request.POST.get("thal")

            # Create a HeartDiseasePrediction object
            heart_data = HeartDiseasePrediction.objects.create(
                age=age,
                sex=sex,
                cp=cp,
                trestbps=trestbps,
                chol=chol,
                fbs=fbs,
                restecg=restecg,
                thalach=thalach,
                exang=exang,
                oldpeak=oldpeak,
                slope=slope,
                ca=ca,
                thal=thal,
            )

            # Load the machine learning model
            log_model = pkl.load(open("log_model.pkl", "rb"))

            # Make prediction
            prediction = log_model.predict(
                [
                    [
                        heart_data.age,
                        heart_data.sex,
                        heart_data.cp,
                        heart_data.trestbps,
                        heart_data.chol,
                        heart_data.fbs,
                        heart_data.restecg,
                        heart_data.thalach,
                        heart_data.exang,
                        heart_data.oldpeak,
                        heart_data.slope,
                        heart_data.ca,
                        heart_data.thal,
                    ]
                ]
            )[0]

            if prediction == 1:
                condition = "Chances of Heart Disease"
            else:
                condition = "No Heart Disease"

            return render(
                request,
                "heart.html",
                {
                    "message": "",
                    "prediction": condition,
                },
            )

        except Exception as e:
            return render(
                request,
                "heart.html",
                {"message": str(e)},
            )
    else:
        return render(request, "heart.html")
