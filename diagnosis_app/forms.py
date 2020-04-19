from django import forms

GENDER_CHOICE = [
    ('MALE', 'Male'),
    ('FEMALE', 'Female'),
]
COUGH_CHOICE = [('1', 'No'), ('2', 'New continuous')]

class DiagnosisForm(forms.Form):
    sex = forms.ChoiceField(choices = GENDER_CHOICE,widget=forms.Select(attrs={'class': 'form-control'}))
    age = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
    xray = forms.FileField(widget=forms.FileInput(attrs={'class': 'form-control'}))
