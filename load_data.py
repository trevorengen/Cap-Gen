from keras.callbacks import ModelCheckpoint
from pre_text import load_doc
from pickle import load
from tokenizer import to_lines, create_tokenizer, max_length, create_sequences
from model import define_model

# WARNING!!! Running this file can take a LONG time and may not work on some computers.
# Preparing the data in timesteps alone can take close to an hour even on modern CPUs.
# Do not attempt to run with 8GB of RAM or less (it will just crash with a memory error).
# If you can use CUDA and cuDNN with an NVIDIA GPU to drastically reduce the training time.
# Training on a modern CPU for ~20 epochs will take close to a day, much less on GPU.
# I'd suggest running it on a google jupyter notebook if you can (unless they're all
# taken like when I wanted to use it and now am waiting 20 hours for my model to train
# because my sad old 980ti doesn't have the vRAM to cut it). Anyways much love to 
# Jason Brownlee for much of this code!

def load_set(filename):
    doc = load_doc(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

def load_clean_descriptions(filename, dataset):
    doc = load_doc(filename)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_id, image_desc = tokens[0], tokens[1:]
        if image_id in dataset:
            if image_id not in descriptions:
                descriptions[image_id] = list()
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            descriptions[image_id].append(desc)
    return descriptions

# This is slow, refactor if I feel like it at some point.
def load_photo_features(filename, dataset):
    all_features = load(open(filename, 'rb'))
    features = {k: all_features[k] for k in dataset}
    return features

if __name__=='__main__':
    # Train data
    filename = 'text/Flickr_8k.trainImages.txt'
    train = load_set(filename)
    print(f'Dataset: {len(train)}')

    train_descriptions = load_clean_descriptions('descriptions.txt', train)
    print(f'Descriptions: {len(train_descriptions)}')

    train_features = load_photo_features('vgg16features.pkl', train)
    print(f'Photos: {len(train_features)}')

    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)

    max_length = max_length(train_descriptions)
    print(f'Max Description Length: {max_length}')

    X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)

    # Test data
    filename = 'text/Flickr_8k.devImages.txt'
    test = load_set(filename)
    print(f'Dataset: {len(test)}')

    test_descriptions = load_clean_descriptions('descriptions.txt', test)
    print(f'Descriptions: test={len(test_descriptions)}')

    test_features = load_photo_features('vgg16features.pkl', test)
    print(f'Photos: test={len(test_features)}')

    X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features, vocab_size)

    # Fit model
    model = define_model(vocab_size, max_length)
    filepath = 'model-ep6{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # IF RUNNING VGG16 LOWER EPOCHS IT IS MUCH SLOWER THAN INCEPTIONV3
    model.fit([X1train, X2train], ytrain, epochs=10, verbose=1, callbacks=[checkpoint], validation_data=([X1test, X2test], ytest))
