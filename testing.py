from tokenizer import create_tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model
from pickle import load
from load_data import load_clean_descriptions, load_photo_features, load_set
from tokenizer import max_length

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    for key, desc_list in descriptions.items():
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        references = [d.split() for d in desc_list]
        actual.append(references)
        predicted.append(yhat.split())

    print(f'BLEU-1: {corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))}')
    print(f'BLEU-2: {corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))}')
    print(f'BLEU-3: {corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))}')
    print(f'BLEU-4: {corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))}')

if __name__=='__main__':
    filename = 'text/Flickr_8k.trainImages.txt'
    train = load_set(filename)
    print(f'Dataset: {len(train)}')
    train_descriptions = load_clean_descriptions('descriptions.txt', train)
    print(f'Descriptions: train={len(train_descriptions)}')

    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index)
    print(f'Vocabulary Size: {vocab_size}')

    maxlen = max_length(train_descriptions)
    print(f'Description Length: {maxlen}')

    filename = 'text/Flickr_8k.testImages.txt'
    test = load_set(filename)
    print(f'Dataset: {len(test)}')

    test_descriptions = load_clean_descriptions('descriptions.txt', test)
    print(f'Descriptions: test={len(test_descriptions)}')

    test_features = load_photo_features('features.pkl', test)
    print(f'Photos: test={len(test_features)}')

    filename = 'model-ep003-loss3.654-val_loss3.874.h5'
    model = load_model(filename)
    evaluate_model(model, test_descriptions, test_features, tokenizer, maxlen)