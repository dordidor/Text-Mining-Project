from tqdm.notebook import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# deep learning library
from keras.models import *
from keras.layers import *
from keras.callbacks import *

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def run_word_embedding(df, name):
    tokenized_corpus = []

    tokenized_corpus = [x for i, y in df['reference_token'].apply(list).iteritems() for x in y]
    tokenized_corpus = [word for sent in tokenized_corpus for word in sent]
    tokenized_corpus = [word for sent in df['translation_token'] for word in sent]
    # [tokenized_corpus.append(word) for doc in df['reference_token'] for word in doc]
    # [tokenized_corpus.append(word) for word in df['translation_token']]
    # vocabulary = {word for doc in tokenized_corpus for word in doc}

    word2idx = {w: idx for (idx, w) in enumerate(set(tokenized_corpus))}

    # load word embeddings 
    W1 = np.load('../corpus/' + str(name) + '/laser.reference_embeds.npy')
    W2 = np.load('../corpus/' + str(name) + '/laser.translation_embeds.npy')

    # training_pairs = build_word_embedding_training(tokenized_corpus, word2idx)

    # W1, W2, losses = Skip_Gram(training_pairs, word2idx, epochs=2)

    W = torch.from_numpy(W1) + torch.from_numpy(W2)
    W = (torch.t(W) / 2).clone().detach()

    df['wordEmbDistance'] = get_word_embedding_distance(W, word2idx, df['reference_token'], df['translation_token'])
    print(df)
    return df


def get_word_embedding_distance(W, word2idx, reference, translation):
    distances = []
    for sent_idx in range(len(reference)):
        distances.append(
            apply_word_embedding_distance(W, word2idx, reference.iloc[sent_idx], translation.iloc[sent_idx]))

    return distances


def apply_word_embedding_distance(W, word2idx, sentence1, sentence2):
    distance = 0
    for word1 in sentence1[0]:
        for word2 in sentence2:
            distance += euclidean_distances([W[word2idx[word1]].numpy()], [W[word2idx[word2]].numpy()])
    return distance


def build_word_embedding_training(tokenized_corpus, word2idx, window_size=2):
    window_size = 2
    idx_pairs = []

    # for each sentence
    for sentence in tokenized_corpus:
        indices = [word2idx[word] for word in sentence]
        # for each word, threated as center word
        for center_word_pos in range(len(indices)):
            # for each window position
            for w in range(-window_size, window_size + 1):
                context_word_pos = center_word_pos + w
                # make sure not jump out sentence
                if context_word_pos < 0 or \
                        context_word_pos >= len(indices) or \
                        center_word_pos == context_word_pos:
                    continue
                context_word_idx = indices[context_word_pos]
                idx_pairs.append((indices[center_word_pos], context_word_idx))
    return np.array(idx_pairs)


def get_onehot_vector(word_idx, vocabulary):
    x = torch.zeros(len(vocabulary)).float()
    x[word_idx] = 1.0
    return x


def Skip_Gram(training_pairs, vocabulary, embedding_dims=5, learning_rate=0.001, epochs=10):
    torch.manual_seed(3)
    W1 = Variable(torch.randn(embedding_dims, len(vocabulary)).float(), requires_grad=True)
    W2 = Variable(torch.randn(len(vocabulary), embedding_dims).float(), requires_grad=True)
    losses = []
    for epo in tqdm(range(epochs)):
        loss_val = 0
        for input_word, target in training_pairs:
            x = Variable(get_onehot_vector(input_word, vocabulary)).float()
            y_true = Variable(torch.from_numpy(np.array([target])).long())

            # Matrix multiplication to obtain the input word embedding
            z1 = torch.matmul(W1, x)

            # Matrix multiplication to obtain the z score for each word
            z2 = torch.matmul(W2, z1)

            # Apply Log and softmax functions
            log_softmax = F.log_softmax(z2, dim=0)
            # Compute the negative-log-likelihood loss
            loss = F.nll_loss(log_softmax.view(1, -1), y_true)
            loss_val += loss.item()

            # compute the gradient in function of the error
            loss.backward()

            # Update your embeddings
            W1.data -= learning_rate * W1.grad.data
            W2.data -= learning_rate * W2.grad.data

            W1.grad.data.zero_()
            W2.grad.data.zero_()

        losses.append(loss_val / len(training_pairs))

    return W1, W2, losses


def run_NN(df):

    model = Sequential()

    # embedding layer
    model.add(Embedding(size_of_vocabulary, 300, input_length=100, trainable=True))

    # lstm layer
    model.add(LSTM(128, return_sequences=True, dropout=0.2))

    # Global Maxpooling
    model.add(GlobalMaxPooling1D())

    # Dense Layer
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Add loss function, metrics, optimizer
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["acc"])

    # Adding callbacks
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', save_best_only=True, verbose=1)

    # Print summary of model
    print(model.summary())

    history = model.fit(np.array(x_tr_seq), np.array(y_tr), batch_size=128, epochs=10,
                        validation_data=(np.array(x_val_seq), np.array(y_val)), verbose=1, callbacks=[es, mc])

    model = load_model('best_model.h5')

    # evaluation
    _, val_acc = model.evaluate(x_val_seq, y_val, batch_size=128)
    print(val_acc)

def learn_embedd(df):
    # Tokenize the sentences
    tokenizer = Tokenizer()

    # preparing vocabulary
    tokenizer.fit_on_texts(list(x_tr))

    # converting text into integer sequences
    x_tr_seq = tokenizer.texts_to_sequences(x_tr)
    x_val_seq = tokenizer.texts_to_sequences(x_val)

    # padding to prepare sequences of same length
    x_tr_seq = pad_sequences(x_tr_seq, maxlen=100)
    x_val_seq = pad_sequences(x_val_seq, maxlen=100)


def pretrain_embedding(df):
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('../input/glove6b/glove.6B.300d.txt')

    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    f.close()

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((size_of_vocabulary, 300))

    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


"""from data_manager import load_dataset
from definitions import *"""
if __name__ == '__main__':
    preprocess_config = {
        'lemmatize': False,
        'stemmer': False,
        'punctuation': False,
        'stop_words': False,
        'stop': stop_en
        # lowercase
        # remove punctuation
    }

    df = pd.read_csv("../corpus/cs-en/scores.csv")

    updates = clean(df["reference"],
                    lower=True,
                    lemmatize=preprocess_config['lemmatize'],
                    stemmer=preprocess_config['stemmer'],
                    stop_words=preprocess_config['stop_words'],
                    stop=preprocess_config['stop'])
    update_df(df, updates, "reference")

    updates = clean(df["translation"], lower=True,
                    lemmatize=preprocess_config['lemmatize'],
                    stemmer=preprocess_config['stemmer'],
                    stop_words=preprocess_config['stop_words'],
                    stop=preprocess_config['stop'])
    update_df(df, updates, "translation")

    number_token(df)
    df = tokenize(df)

    df = remove_empty(df)

    run_word_embedding(df, "cs-en")
