from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk.translate.gleu_score as sentence_gleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.nist_score import sentence_nist
#from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from rouge import Rouge
#from nltk import RegexpTokenizer
from transformers import BertModel, BertTokenizerFast
import torch.nn.functional as F
import kiwi


#https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

def cosine_sim(text):
    vectors = [t for t in get_vectors(text)]
    return cosine_similarity(vectors)

def get_vectors(text):
    vectorizer = CountVectorizer()
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()

def get_tfidf(text, doc):
    cv = CountVectorizer()
    cv.fit(text)

    tfidf_vectorizer = TfidfTransformer()
    tfidf_vectorizer.fit(text)

    #generate tf-idf for the given document
    tf_idf_vector = tfidf_vectorizer.transform(cv.transform([doc]))

def extract_feature_scores(feature_names, document_vector):
    """
    Function that creates a dictionary with the TF-IDF score for each feature.
    :param feature_names: list with all the feature words.
    :param document_vector: vector containing the extracted features for a specific document
    
    :return: returns a sorted dictionary "feature":"score".
    """
    feature2score = {}
    for i in range(len(feature_names)):
        feature2score[feature_names[i]] = document_vector[0][i]    
    return sorted(feature2score.items(), key=lambda kv: kv[1], reverse=True)

# levenshtein distance: minimum number of edits (insertion, deletions or substitutions) required to change the hypotheses sentence into the reference.
def wer(translation, reference, print_matrix=False):
  N = len(translation)
  M = len(reference)
  L = np.zeros((N,M))
  for i in range(0, N):
    for j in range(0, M):
      if min(i,j) == 0:
        L[i,j] = max(i,j)
      else:
        deletion = L[i-1,j] + 1
        insertion = L[i,j-1] + 1
        sub = 1 if translation[i] != reference[j] else 0
        substitution = L[i-1,j-1] + sub
        L[i,j] = min(deletion, min(insertion, substitution))
        # print("{} - {}: del {} ins {} sub {} s {}".format(hyp[i], ref[j], deletion, insertion, substitution, sub))
  if print_matrix:
    print("WER matrix ({}x{}): ".format(N, M))
    print(L)
  return int(L[N-1, M-1])

# measures precision
def baseline_bleu(df):
    smoothie = SmoothingFunction().method1
    df['bleu'] = df.apply(lambda x: sentence_bleu(x['reference1'], x['translation1'], weights=(1,0,0,0), smoothing_function=smoothie), axis=1)
    df.drop(columns = ['reference1','translation1'],inplace=True)
    return df 

def gleu_model(df):
    df['reference1'] = [[x.split()] for x in df['reference']]
    df['translation1'] = [x.split() for x in df['translation']]
    df['gleu'] = df.apply(lambda x: sentence_gleu(x['reference1'], x['translation1'], weights=(1,0,0,0)), axis=1)
    df.drop(columns = ['reference1','translation1'],inplace=True)
    return df 

def nist(df):
    #df['reference1'] = [[x.split()] for x in df['reference']]
    #df['translation1'] = [x.split() for x in df['translation']]
    df['nist'] = df.apply(lambda x: sentence_nist(x['reference'], x['translation']),axis=1)
    #df.drop(columns = ['reference','translation'],inplace=True)
    return df 

# measures recall
def rouge(df):
    df['rouge'] = df.apply(lambda x: Rouge.get_scores(x['translation'], x['reference']),axis=1)
    return df

def bleu_rouge(Bleu, Rouge):
    F1 = 2 * (Bleu * Rouge) / (Bleu + Rouge)
    return F1

def meteor(df):
    df['meteor'] = df.apply(lambda x: meteor_score([x['reference']], x['translation']),axis=1)
    #If no words match during the method returns the score of 0
    return df

def comet(df):
    pass

def charf(df):
    pass

def labse(df, language1, language2):
    tokenizer = BertTokenizerFast.from_pretrained("setu4993/LaBSE")
    model = BertModel.from_pretrained("setu4993/LaBSE")
    model = model.eval()

    source_inputs = tokenizer(df['Source'], return_tensors="pt", padding=True)
    translation_inputs = tokenizer(df['Translation'], return_tensors="pt", padding=True)

    with torch.no_grad():
        source_outputs = model(**source_inputs)
        translation_outputs = model(**translation_inputs)

    source_embeddings = source_outputs.pooler_output
    translation_embeddings = translation_outputs.pooler_output
    return source_embeddings, translation_embeddings

    def similarity(embeddings_1, embeddings_2):
        normalized_embeddings_1 = F.normalize(embeddings_1, p=2)
        normalized_embeddings_2 = F.normalize(embeddings_2, p=2)
        return torch.matmul(
            normalized_embeddings_1, normalized_embeddings_2.transpose(0, 1)
        )

    print(similarity(source_embeddings, translation_embeddings))


    
