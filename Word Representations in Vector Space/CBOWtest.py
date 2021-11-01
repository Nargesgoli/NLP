import numpy as np
import pandas as pd
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from collections import Counter
from collections import defaultdict
from scipy import linalg
from matplotlib import pyplot
# from utils2 import sigmoid, get_batches, compute_pca, get_dict
nltk.data.path.append('.')
df = pd.read_excel (r'C:\Users\nrgsg\OneDrive\Desktop\cities and states.xlsx')
city = df['city'].tolist()
city=[ ch.lower() for ch in city]
state = df['state'].tolist()
state=[ ch.lower() for ch in state]
z=list(zip(city , state))
import re
f=open(r"C:\Users\nrgsg\OneDrive\Desktop\shakespeare2.txt")
data = f.read()
print('len(data)=',len(data))
data = re.sub(r'[,!?;-]', '.',data)
print(len(data))
data = nltk.word_tokenize(data)
print(len(data))
data = [ ch.lower() for ch in data if ch.isalpha() or ch == '.']
print("Number of tokens:", len(data),'\n', data[:15])
print('data[0:10]=',data[0:10])
# Compute the frequency distribution of the words in the dataset (vocabulary)
fdist=FreqDist (word for word in data)
print("Size of vocabulary: ",len(fdist) )
print('len(data)=',len(data))
print("Most frequent tokens: ",fdist.most_common(20) )
# data = ['Best way to success is through hardwork and persistence']
def get_dict(data):
    words = sorted(list(set(data)))
    n = len(words)
    idx = 0
    word2Ind = {}
    Ind2word = {}
    for k in words:
        word2Ind[k] = idx
        Ind2word[idx] = k
        idx += 1
    return word2Ind, Ind2word

word2Ind, Ind2word = get_dict(data)

V = len(word2Ind)
print("Size of vocabulary: ", V)
print('len(data)=', len(data))
def initialize_model(N,V,random_seed):
    np.random.seed(random_seed)
    W1=np.random.rand(N,V)
    W2=np.random.rand(V,N)
    b1=np.random.rand(N,1)
    b2=np.random.rand(V,1)
    return(W1,W2,b1,b2)
# print("W2=",W2)
def softmax(z):
    yhat = ((np.exp(z))/(np.sum(np.exp(z),axis=0)))
    return yhat

def get_windows(wordss, C):
    i = C
    while i < len(wordss) - C:
        center_word = wordss[i]
        context_words = wordss[(i - C):i] + wordss[(i+1):(i+C+1)]
        # centerword.append(center_word)
        # contextwords.append(context_words)
        yield context_words,center_word
        i += 1
def get_idx(words, word2Ind):
    idx = []
    for word in words:
        idx = idx + [word2Ind[word]]
    return idx


def pack_idx_with_frequency(context_words, word2Ind):
    freq_dict = defaultdict(int)
    for word in context_words:
        freq_dict[word] += 1
    idxs = get_idx(context_words, word2Ind)
    packed = []
    for i in range(len(idxs)):
        idx = idxs[i]
        freq = freq_dict[context_words[i]]
        packed.append((idx, freq))
    return packed

def get_vectors(data, word2Ind, V, C):
    i = C
    while True:
        y = np.zeros(V)
        x = np.zeros(V)
        center_word = data[i]
        y[word2Ind[center_word]] = 1
        context_words = data[(i - C):i] + data[(i+1):(i+C+1)]
        num_ctx_words = len(context_words)
        for idx, freq in pack_idx_with_frequency(context_words, word2Ind):
            x[idx] = freq/num_ctx_words
        yield x, y
        i += 1
        if i >= len(data):
            print('i is being set to 0')
            i = 0

def word_to_one_hot_vector(word, word2Ind, V):
    one_hot_vector = np.zeros(V)
    one_hot_vector[word2Ind[word]] = 1
    return one_hot_vector
# print('word=',word_to_one_hot_vector(center_word,word2Ind, V))
# # Define the 'context_words_to_vector' function as seen in a previous notebook
def context_words_to_vector(context_words, word2Ind, V):
    context_words_vectors = [word_to_one_hot_vector(w, word2Ind, V) for w in context_words]
    context_words_vectors = np.mean(context_words_vectors, axis=0)
    return context_words_vectors
one_hot_vector=word_to_one_hot_vector('king', word2Ind, V)

print('one_hot_vector=',np.argmax(one_hot_vector))
# print('context_words_vectors=',np.sum(context_words_vectors))
def get_batches(data, word2Ind, V, C, batch_size):
    batch_x = []
    batch_y = []
    for x, y in get_vectors(data, word2Ind, V, C):
        while len(batch_x) < batch_size:
            batch_x.append(x)
            batch_y.append(y)
        else:
            yield np.array(batch_x).T, np.array(batch_y).T
            batch = []

batch=get_batches(data, word2Ind, V, 2, 10)
j=0
for x,y in  get_batches(data, word2Ind, V, 2, 10):
    print(j,x,y,x.shape)
    j +=1


    # print('y=',y)
def forward_prop(x, W1, W2, b1, b2):
    h=(np.dot(W1,x))+b1
    h = h.copy()
    h[h<0]=0
    z=(np.dot(W2,h))+b2
    return(z,h)

def compute_cost(y,yhat,batch_size):
    logprobs = np.multiply(np.log(yhat),y)
    cost = - 1/batch_size * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost
# cost=compute_cost(y,yhat,batch_size)
#
def back_prop (x, yhat, y, h, W1, W2, b1, b2, batch_size):
    l1 = np.dot(W2.T,(yhat-y))
    l1 = l1.copy()
    l1[l1<0]=0
    grad_W1 = (1/batch_size)*(np.dot(l1,x.T))
    grad_W2 = (1/batch_size)*(np.dot((yhat-y),h.T))
    grad_b1 = (1/batch_size)*(np.sum(l1,axis=1,keepdims=True))
    grad_b2 = (1/batch_size)*(np.sum((yhat-y),axis=1,keepdims=True))
    return grad_W1, grad_W2, grad_b1, grad_b2
# print('data[:128]=',type(data[:128]))

def gradient_descent(data, word2Ind, N, V, num_iters, alpha=0.03):
    batch_size = 128
    W1, W2, b1, b2 = initialize_model(N,V,random_seed=282)
    iters = 0
    C = 2
    for x,y in get_batches(data, word2Ind, V, C, batch_size=batch_size):
        z, h = forward_prop(x, W1, W2, b1, b2)
        yhat = softmax(z)
        cost = compute_cost(y, yhat, batch_size=batch_size)
        if ( (iters+1) % 10 == 0):
            print(f"iters: {iters + 1} cost: {cost:.6f}")
        grad_W1, grad_W2, grad_b1, grad_b2 = back_prop(x, yhat, y, h, W1, W2, b1, b2, batch_size=batch_size)

        W1 = W1-(alpha*grad_W1)
        W2 = W2-(alpha*grad_W2)
        b1 = b1-(alpha*grad_b1)
        b2 = b2-(alpha*grad_b2)

        iters += 1
        if iters == num_iters:
            break
        if iters % 100 == 0:
            alpha *= 0.66
    return W1, W2, b1, b2

def compute_pca(data, n_components=2):
    m, n = data.shape
    data -= data.mean(axis=0)
    R = np.cov(data, rowvar=False)
    evals, evecs = linalg.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    evals = evals[idx]
    evecs = evecs[:, :n_components]
    return np.dot(evecs.T, data.T).T

W1, W2, b1, b2 = gradient_descent(data, word2Ind, 50, V, 150)
# print('W1=',W1[0])
# print('W2=',W2[0])
# print('b1=',b1[0])
# print('b2=',b2[0])
# embs = (W1.T + W2)/2.0
embs = (W1.T+W2)/2
words=['king', 'queen','man','lord', 'woman','dog','wolf',
         'rich','happy','sad']
idx1 = [word2Ind[word] for word in words]
print('idx=',idx1)
X=embs[idx1,:]
print('X=',X)
Y=X[2]-X[0]+X[1]
# print('Y=',Y)
def embedings(words,word2Indv,embs):
    idx = word2Ind[words]
    X=embs[idx,:]
    return X
e=embedings('king',word2Ind,embs)
# print('e=',e)
def cosine_similarity(A, B):
    dot = np.dot(A,B)
    norma = np.linalg.norm(A)
    normb = np.linalg.norm(B)
    cos = (dot/(norma*normb))
    return cos
def nearest_neighbor(v, candidates, k):
    similarity_l = []
    for row in candidates:
        cos_similarity = cosine_similarity(v,row)
        similarity_l.append(cos_similarity)
    sorted_ids = np.argsort( similarity_l)
    k_idx = sorted_ids[-k:]
    return k_idx
k1=nearest_neighbor(Y, embs, k=3)
print('k1=',k1)
# p=Ind2word[k1[0],k1[1]]
print('p=',Ind2word[k1[0]],Ind2word[k1[1]],Ind2word[k1[2]])
# def accuracy(y, yhat):
#     assert(y.shape == yhat.shape)
#     return np.sum(y == yhat) * 100.0 / y.size
# def accuracy(z,data):
#     i=0
#     yhat=[]
#     y=[]
#     while i<2:
#         pair=random.choices(z,k=2)
#         p1=pair[0][0]
#         p2=pair[0][1]
#         p3=pair[1][0]
#         p4=pair[1][1]
#         if (p1 in data) and (p2 in data) and (p3 in data) and (p4 in data):
#             # q0=embedings(p1,word2Ind,embs)
#             # q1=embedings(p2,word2Ind,embs)
#             # q2=embedings(p3,word2Ind,embs)
#             # q3=embedings(p4,word2Ind,embs)
#             # A=q2-q0+q1
#             # pred=nearest_neighbor(A, embs, k=1)
#             y.append(p4)
#             # yhat.append(pred)
#             i+=1
#     # assert(y.shape == yhat.shape)
#     # return np.sum(y == yhat) * 100.0 / len(y)
#     return y
# acc=accuracy(z,data)
# print('accuracy=',acc)
