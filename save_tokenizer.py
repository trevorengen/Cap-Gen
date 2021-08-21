from load_data import load_set, load_clean_descriptions
from pre_text import load_doc
from tokenizer import to_lines, create_tokenizer
from pickle import dump

if __name__=='__main__':
    filename = 'text/Flickr_8k.trainImages.txt'
    train = load_set(filename)
    print(f'Dataset: {len(train)}')

    train_descriptions = load_clean_descriptions('descriptions.txt', train)
    print(f'Descriptions: train={len(train_descriptions)}')

    tokenizer = create_tokenizer(train_descriptions)

    dump(tokenizer, open('tokenizer.pkl', 'wb'))