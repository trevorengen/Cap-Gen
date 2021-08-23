from pickle import load
from keras.models import load_model, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import img_to_array, load_img
from numpy import reshape
from testing import generate_desc

def extract_features(filename):
    model = VGG16()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # 224x224 image size for VGG16 || 299x299 for InceptionV3
    image = load_img(filename, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    feature = model.predict(image, verbose=0)
    return feature

if __name__=='__main__':
    tokenizer = load(open('tokenizer.pkl', 'rb'))
    maxlen = 34

    model = load_model('models\model-ep003-loss3.654-val_loss3.874.h5')

    photo = extract_features('Flicker8k_Dataset/162152393_52ecd33fc5.jpg')
    description = generate_desc(model, tokenizer, photo, maxlen)
    print(description)