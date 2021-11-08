# from keras.models import Sequential, load_model
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation, Dropout, Flatten, Dense
# # from keras.utils import np_utils
# import keras
# import sys
# import numpy as np
# from PIL import Image

#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop

# from keras.utils import np_utils

import keras
import sys
import numpy as np
from PIL import Image

classes = ["monkey", "boar", "crow"]
num_classes = len(classes)
image_size = 50


def predict_animal(animal_file, model_file):

    image = Image.open(animal_file)
    model_file = model_file

    image = image.convert('RGB')
    image = image.resize((image_size, image_size))
    data = np.asarray(image) / 255
    X = []
    X.append(data)
    X = np.array(X)

    model = load_model(model_file)

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


def main():
    args = sys.argv
    if len(args) < 3:
        print(
            'Error short args => Usage:python predict.py image_file[*.jpg] cnn_file[*.h5]', len(args))
        return

    animal_file = sys.argv[1]
    model_file = sys.argv[2]

    # image = Image.open(animal_file)
    #
    # image = image.convert('RGB')
    # image = image.resize((image_size, image_size))
    # data = np.asarray(image) / 255
    # X = []
    # X.append(data)
    # X = np.array(X)
    # model = build_model(model_file)
    # # model.summary()
    #
    # result = model.predict([X])[0]
    # predicted = result.argmax()
    # percentage = int(result[predicted] * 100)
    # print("{0} ({1} %)".format(classes[predicted], percentage))

    predicted, percentage = predict_animal(animal_file, model_file)


if __name__ == "__main__":
    main()
