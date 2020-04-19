from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.contrib.auth.decorators import login_required
from PIL import Image
from .forms import DiagnosisForm
from .ml_model_k import run_diagnosis

# Create your views here.
def index(request):
    context = {}
    return render(request, "diagnosis/index.html", context)

@login_required
def diagnosis(request):
    context = {}
    if request.method == 'POST':
        form = DiagnosisForm(request.POST, request.FILES)
        if form.is_valid():
            myfile = request.FILES['xray']
            fs = FileSystemStorage()
            filename = fs.save(myfile.name, myfile)
            rfname, cfname, hfname = handle_uploaded_file(filename)
            context['result'] = '/{}'.format(rfname)
            context['chart'] = '/{}'.format(cfname)
            context['heatmap'] = '/{}'.format(hfname)
            return render(request, "diagnosis/results.html", context)
    else:
        form = DiagnosisForm()

    context['form'] = form
    return render(request, "diagnosis/diagnose.html", context)

def handle_uploaded_file(filename):
    rfname, cfname, hfname= run_diagnosis(filename,settings.MEDIA_ROOT)
    return rfname, cfname, hfname

