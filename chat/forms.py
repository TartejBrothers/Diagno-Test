from django import forms


class SymptomForm(forms.Form):
    symptom = forms.CharField(label="Symptom", max_length=100)
    days = forms.IntegerField(label="Days")
