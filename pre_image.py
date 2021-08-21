from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from os import listdir
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from pickle import dump

def extract_features(directory):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    
    features = dict()
    for name in listdir(directory):
        filename = directory + '/' + name
        # VGG16 expects 224x224 images
        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)
        # Reshape the image array to fit the model input
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        # Retrieve image features, verbose displays information in terminal if you'd like.
        feature = model.predict(image, verbose=0)
        image_id = name.split('.')[0]

        # Store the images features in the dictionary at the image's ID.
        features[image_id] = feature

    return features


# THIS CODE CAN TAKE A VERY LONG TIME TO RUN!
# Standard CPU's can expect upwards of an hour of runtime.
# Using GPU (or TPU of course) will significantly speed up the process.
# Otherwise just be patient and consider the magic that your computer is
# reading every pixel and processing them in thousand of images in 
# a relatively low timeframe. 
if __name__=='__main__':
    directory = 'Flicker8k_Dataset'
    features = extract_features(directory)
    print(f'Extracted Features: {len(features)}')
    dump(features, open('features.pkl', 'wb'))