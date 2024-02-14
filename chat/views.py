from django.shortcuts import render, redirect
from django.http import HttpResponseServerError
from .forms import SymptomForm
from nlpmodel.nlp import (
    train_model,
    get_symptoms_dict,
    get_description_list,
    get_precaution_dict,
    tree_to_code,
)
import pandas as pd


def predict_disease(request):
    if request.method == "POST":
        form = SymptomForm(request.POST)
        if form.is_valid():
            symptom = form.cleaned_data["symptom"]
            days = form.cleaned_data["days"]
            training_data = "Data/Training.csv"
            mean_score, svm_score, clf, le, cols = train_model(training_data)
            symptoms_dict = get_symptoms_dict(training_data)
            description_list = get_description_list()
            precautionDictionary = get_precaution_dict()
            reduced_data = pd.DataFrame()
            print("Symptoms dict:", symptoms_dict)
            try:
                result = tree_to_code(
                    clf,
                    cols,
                    description_list,
                    precautionDictionary,
                    symptom,
                    days,
                    le,
                    reduced_data,
                )
                print(result)
                return redirect("result_page", result=result)
            except Exception as e:
                print("An error occurred:", e)
                return HttpResponseServerError(
                    "An error occurred. Please try again later."
                )
    else:
        form = SymptomForm()
    return render(request, "form.html", {"form": form})


def result_page(request):
    result = request.GET.get("result")
    return render(request, "result_page.html", {"result": result})


# def predict(symptom_input, num_days):
#     # Assuming you have getInfo() implemented to get user info
#     getInfo()

#     # Assuming you have check_pattern() implemented to check pattern of input
#     conf, cnf_dis = check_pattern(symptoms_dict.keys(), symptom_input)

#     # Assuming you have sec_predict() implemented to make a secondary prediction
#     second_prediction = sec_predict(symptoms_exp)

#     # Assuming you have calc_condition() implemented to calculate condition based on symptoms and number of days
#     condition = calc_condition(symptoms_exp, num_days)

#     # Assuming you have tree_to_code() implemented to get prediction from decision tree
#     result = tree_to_code(clf, cols)

#     return result


# def chatbot_view(request):
#     if request.method == "POST":
#         input_text = request.POST.get("input_text", "")
#         response = getInfo(input_text)
#         return JsonResponse({"response": response})
#     else:
#         return render(request, "chatbot.html")


# def health_chat(request):
#     if request.method == "POST":
#         symptom_input = request.POST.get("symptom")
#         num_days = request.POST.get("duration")

#         # Call functions directly from nlp.py
#         disease_prediction = load_data_and_predict(
#             symptom_input, num_days
#         )  # Your custom function
#         description = getDescription(disease_prediction)  # Your custom function
#         precautions = getPrecautions(disease_prediction)  # Your custom function

#         context = {
#             "predicted_disease": disease_prediction[0],
#             "description": description,
#             "precautions": precautions,
#         }
#         return render(request, "prediction_results.html", context)
#     else:
#         return render(request, "prediction_form.html")


# def chatbot_view(request):
#     if request.method == "POST":
#         input_text = request.POST.get("input_text", "")
#         response = process_user_input(input_text)
#         return JsonResponse({"response": response})
#     else:
#         # If GET request, render the HTML template
#         return render(request, "chatbot.html")


# Import other necessary libraries from the NLP script


# def health_chat(request):
#     if request.method == "POST":
#         symptom_input = request.POST.get("symptom")
#         num_days = request.POST.get("duration")

#         # Load model, dictionaries, and functions using nlpnew
#         disease_prediction, description, precautions = nlpnew.process_data(
#             symptom_input, num_days
#         )

#         context = {
#             "predicted_disease": disease_prediction,
#             "description": description,
#             "precautions": precautions,
#         }
#         return render(request, "prediction_results.html", context)
#     else:
#         return render(request, "prediction_form.html")


# def load_model():
#     model = tf.keras.models.load_model("nlp_model.h5")
#     return model


# def predict_response(model, input_text):
#     prediction = model.predict([input_text])
#     return prediction


# def chatbot_view(request):
#     if request.method == "POST":
#         input_text = request.POST.get("input_text", "")
#         model = load_model()

#         response = predict_response(model, input_text)
#         print(response)
#         return JsonResponse({"response": response})
#     else:
#         return render(request, "chat.html")


def index(request):
    return render(request, "index.html")
