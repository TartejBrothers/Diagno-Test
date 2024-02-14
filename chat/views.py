from django.shortcuts import render


from django.shortcuts import render
from django.http import JsonResponse
import tensorflow as tf

def load_model():
    # Load the pre-trained TensorFlow model
    model = tf.keras.models.load_model('path/to/your/model.h5')
    return model

def predict_response(model, input_text):
    # Preprocess the input text if needed
    # For example, tokenize and convert to numerical format
    
    # Perform prediction using the loaded model
    prediction = model.predict([input_text])
    
    # Postprocess the prediction if needed
    # For example, convert numerical output to text
    
    return prediction

def chatbot_view(request):
    if request.method == 'POST':
        input_text = request.POST.get('input_text', '')
        
        # Load the model
        model = load_model()
        
        # Get the response from the model
        response = predict_response(model, input_text)
        
        # You may need to format the response as per your requirement
        
        return JsonResponse({'response': response})
    else:
        return render(request, 'chatbot/index.html')

# Create your views here.
def index(request):
    return render(request, "index.html")


def nlp(request):
    return render(request, "chat.html")
