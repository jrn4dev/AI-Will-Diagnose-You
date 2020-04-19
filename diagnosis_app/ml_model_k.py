from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from django.conf import settings

MODEL_DIR = settings.BASE_DIR
SAVE_DIR= 'media'
MODEL_FILE = 'covid19_3class.h5.model'
CLASSES = ['normal','covid','pneumonia']
RESULT_CHART = 'chart.png' 
RESULT_IMG = 'result.png'
RESULT_HT = 'cam.png'

def print_chart(prediction):
    chart_fname = unique_filename(RESULT_CHART)
    pclasses = tuple(CLASSES) 
    y_pos = np.arange(len(pclasses))
    performance = prediction
    
    plt.figure()
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, pclasses)
    plt.ylabel('Percentage')
    plt.title('Probabilities to have a pulmonar disease')
    plt.savefig(chart_fname)
    return chart_fname

def print_heatmap(model,img_path,index):
    LAYER_NAME = 'block5_conv3'
    covid_CLASS_INDEX = index
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])

    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img])/255.0)
        loss = predictions[:, covid_CLASS_INDEX]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    cam = np.ones(output.shape[0: 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)

    output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)
    u_hfname = unique_filename(RESULT_HT)
    cv2.imwrite(u_hfname, output_image)

    return u_hfname
    
def print_results(outcome):
    r_fname = unique_filename(RESULT_IMG)
    #img = Image.open(outcome["img_path"])
    img = plt.imread(outcome["img_path"])
    fig = plt.figure()
    fig.suptitle(outcome["message"])
    plt.imshow(np.asarray(img))
    plt.savefig(r_fname)
    return r_fname

def unique_filename(filename):
    pref_fix = random.randint(10000, 80000)
    u_fname ='{}_{}'.format(pref_fix,filename)
    return os.path.join(SAVE_DIR,u_fname)

def load_model():
    mod_path = os.path.join(MODEL_DIR,MODEL_FILE)
    model = tf.keras.models.load_model(mod_path)
    return model

def get_img(imagePath):
    data = []
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    data.append(image)
    data = np.array(data) / 255.0
    return data

def predict_disease(img_path):
    xray_img = get_img(img_path)
    model = load_model()
    prediction = model.predict(xray_img)

    return prediction, model

def diagnosis_msg(ind):
    class_str = 'Covid 19, Pneumonia'
    switcher = {
        'normal': "It is unlikely the patient has any of the following {}".format(class_str)
    }
    default = "It is likely the patient has  '{}'. We advice to see a doctor.".format(CLASSES[ind])
    return switcher.get(CLASSES[ind],default)


def run_diagnosis(img_fname,media_path):
    SAVE_DIR = media_path
    message = ""
    class_str = ','.join(CLASSES)
    image_path = os.path.join(SAVE_DIR,img_fname)

    prediction, model = predict_disease(image_path)
    index = prediction.argmax()
    message = diagnosis_msg(index)

    results = {"img_path":image_path, "message":message, "probability": prediction[0]}

    rfname = print_results(results)
    cfname = print_chart(results["probability"])
    hfname = print_heatmap(model,image_path,index)

    return rfname, cfname, hfname


