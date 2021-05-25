from tqdm.notebook import tqdm
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def run_word_embedding(df, name):

    tokenized_corpus = []
    [tokenized_corpus.append(word) for doc in df['reference_token'] for word in doc]
    [tokenized_corpus.append(word) for doc in df['translation_token'] for word in doc]
    vocabulary = {word for doc in tokenized_corpus for word in doc}

    word2idx = {w: idx for (idx, w) in enumerate(vocabulary)}

    # load word embeddings 
    W1 = np.load('corpus/'+ str(name) +'/laser.reference_embeds.npy')
    W2 = np.load('corpus/'+ str(name) +'/laser.translation_embeds.npy')

    #training_pairs = build_word_embedding_training(tokenized_corpus, word2idx)

    #W1, W2, losses = Skip_Gram(training_pairs, word2idx, epochs=2)

    W = torch.from_numpy(W1) + torch.from_numpy(W2)
    W = (torch.t(W)/2).clone().detach()


    df['wordEmbDistance'] = get_word_embedding_distance(W, word2idx, df['reference'], df['translation'])
    print(df)
    return df


def get_word_embedding_distance(W, word2idx, reference, translation):
    distances = []
    for sent_idx in range(len(reference)):
        distances.append(apply_word_embedding_distance(W, word2idx, reference.iloc[sent_idx], translation.iloc[sent_idx]))

    return distances


def apply_word_embedding_distance(W, word2idx, sentence1, sentence2):
    distance = 0
    for word1 in sentence1:
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


if __name__ == '__main__':
    a = np.load('../corpus/' + str("de-en") + '/laser.translation_embeds.npy')
    1+1
