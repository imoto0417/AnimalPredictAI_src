from flask import send_from_directory
import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename

# from keras.models import Sequential, load_model
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop

import keras
import sys
import numpy as np
from PIL import Image


classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ================
classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50


def build_model(model_file):
    # モデルのロード
    # model = load_model('./animal_cnn_inc.h5')
    model = load_model(model_file)

    return model


def predict_animal(animal_file, model_file):

    image = Image.open(animal_file)
    model_file = model_file

    image = image.convert('RGB')
    image = image.resize((image_size, image_size))
    data = np.asarray(image) / 255
    X = []
    X.append(data)
    X = np.array(X)
    model = build_model(model_file)
    # model.summary()

    result = model.predict([X])[0]
    print("__________________________________________")
    print("Result model = ", model_file)
    print("__________________________________________")
    for no in range(len(result)):
        percentage = int(result[no] * 100)

        if no == result.argmax():
            print("**{0} ({1} %)".format(classes[no], percentage))
        else:
            print("  {0} ({1} %)".format(classes[no], percentage))
    print("_______________")

    animal_class = result.argmax()
    return classes[animal_class], int(result[animal_class] * 100)
# ================


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # model = load_model('./animal_cnn_inc.h5')
            #
            # image = Image.open(filepath)
            # image = image.convert('RGB')
            # image = image.resize((image_size, image_size))
            # data = np.asarray(image)
            # X = []
            # X.append(data)
            # X = np.array(X)
            #
            # result = model.predict([X])[0]
            # predicted = result.argmax()
            # percentage = int(result[predicted] * 100)

            predicted, percentage = predict_animal(
                filepath, "./animal_cnn_inc.h5")

            # return "ラベル： " + classes[predicted] + ", 確率：" + str(percentage) + " %"
            return "ラベル： " + predicted + ", 確率：" + str(percentage) + " %"
            # return redirect(url_for('uploaded_file', filename=filename))
    return '''
    <!doctype html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>ファイルをアップロードして判定しよう</title></head>
    <body>
    <h1>ファイルをアップロードして判定しよう！</h1>
    <form method = post enctype = multipart/form-data>
    <p><input type=file name=file>
    <input type=submit value=Upload>
    </form>
    </body>
    </html>
    '''


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
