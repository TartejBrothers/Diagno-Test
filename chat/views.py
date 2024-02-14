from django.shortcuts import render


from django.shortcuts import render
from django.http import JsonResponse
import tensorflow as tf


def load_model():
    model = tf.keras.models.load_model("nlp_model.h5")
    return model


def predict_response(model, input_text):
    prediction = model.predict([input_text])

    return prediction


def chatbot_view(request):
    if request.method == "POST":
        input_text = request.POST.get("input_text", "")
        model = load_model()

        response = predict_response(model, input_text)

        return JsonResponse({"response": response})
    else:
        return render(request, "chat.html")


def index(request):
    return render(request, "index.html")
