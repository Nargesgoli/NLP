import re
import numpy as np
f=open(r"C:\Users\nrgsg\OneDrive\Desktop\shakespeare2.txt")
data = f.read()
# print(data)
text = 'Best way to success is through hardwork and persistence'
def initialize_model(N,vocab_size,random_seed):
    np.random.seed(random_seed)
    W1=np.random.rand(N,vocab_size)
    W2=np.random.rand(vocab_size,N)
    # b1=np.random.rand(N,1)
    # b2=np.random.rand(V,1)
    return(W1,W2)

def generate_dictinoary_data(text):
    word_to_index= dict()
    index_to_word = dict()
    corpus = []
    count = 0
    vocab_size = 0

    for word in text.split():
        # for word in row.split():
            word = word.lower()
            corpus.append(word)
            if word_to_index.get(word) == None:
                word_to_index.update ( {word : count})
                index_to_word.update ( {count : word })
                count  += 1
    vocab_size = len(word_to_index)
    length_of_corpus = len(corpus)
    return word_to_index,index_to_word,corpus,vocab_size,length_of_corpus
word_to_index,index_to_word,corpus,vocab_size,length_of_corpus=generate_dictinoary_data(data)
# print('corpus=',corpus)

def get_one_hot_vectors(target_word,context_words,vocab_size,word_to_index):
    trgt_word_vector = np.zeros(vocab_size)
    index_of_word_dictionary = word_to_index.get(target_word)
    trgt_word_vector[index_of_word_dictionary] = 1
    ctxt_word_vector = np.zeros(vocab_size)
    for word in context_words:
        index_of_word_dictionary = word_to_index.get(word)
        ctxt_word_vector[index_of_word_dictionary] = 1
    return trgt_word_vector,ctxt_word_vector

def generate_training_data(corpus,window_size,vocab_size,word_to_index,length_of_corpus,sample=None):
    training_data =  []
    training_sample_words =  []
    for i,word in enumerate(corpus):
        index_target_word = i
        target_word = word
        context_words = []
        if i == 0:
            context_words = [corpus[x] for x in range(i + 1 , window_size + 1)]
        elif i == len(corpus)-1:
            context_words = [corpus[x] for x in range(length_of_corpus - 2 ,length_of_corpus -2 - window_size  , -1 )]
        else:
            before_target_word_index = index_target_word - 1
            for x in range(before_target_word_index, before_target_word_index - window_size , -1):
                if x >=0:
                    context_words.extend([corpus[x]])
            after_target_word_index = index_target_word + 1
            for x in range(after_target_word_index, after_target_word_index + window_size):
                if x < len(corpus):
                    context_words.extend([corpus[x]])
        trgt_word_vector,ctxt_word_vector = get_one_hot_vectors(target_word,context_words,vocab_size,word_to_index)
        training_data.append([trgt_word_vector,ctxt_word_vector])
        if sample is not None:
            training_sample_words.append([target_word,context_words])
    return training_data,training_sample_words
training_data,training_sample_words=generate_training_data(corpus,2,vocab_size,word_to_index,length_of_corpus,sample=True)



def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def forward_prop(W1,W2,x):#target_word_vector = x , W1 =  weights for input layer to hidden layer
    h = np.dot(W1,x) #W2 = weights for hidden layer to output layer
    h = h.copy()
    h[h<0]=0
    u = np.dot(W2,h)
    y_predicted = softmax(u)
    return y_predicted, h, u

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def calculate_error(y_pred,context_words):
    total_error = [None] * len(y_pred)
    index_of_1_in_context_words = {}
    for index in np.where(context_words == 1)[0]:
        index_of_1_in_context_words.update ( {index : 'yes'} )
    number_of_1_in_context_vector = len(index_of_1_in_context_words)
    for i,value in enumerate(y_pred):
        if index_of_1_in_context_words.get(i) != None:
            total_error[i]= (value-1) + ( (number_of_1_in_context_vector -1) * value)
        else:
            total_error[i]= (number_of_1_in_context_vector * value)
    return  np.array(total_error)
def get_batches(data, word_to_index, vocab_size, C, batch_size):
    batch_x = []
    batch_y = []
    for x, y in training_data:
        while len(batch_x) < batch_size:
            batch_x.append(x)
            batch_y.append(y)
        else:
            yield np.array(batch_x).T, np.array(batch_y).T
            batch = []
def backward_prop(W1,W2,total_error, h, target_word_vector,learning_rate):
    dl_W1 = (np.outer(target_word_vector, np.dot(W2.T, total_error)))
    dl_W2 = (np.outer(h,total_error))
    W1 = W1 - (learning_rate * dl_W1)
    W2 = W2 - (learning_rate * dl_W2)
    return dl_W1,dl_W2
def back_prop (x, yhat, y, h, W1, W2, batch_size):
    l1 = np.dot(W2.T,(yhat-y))
    l1 = l1.copy()
    l1[l1<0]=0
    grad_W1 = (1/batch_size)*(np.dot(l1,x.T))
    grad_W2 = (1/batch_size)*(np.dot((yhat-y),h.T))
    # grad_b1 = (1/batch_size)*(np.sum(l1,axis=1,keepdims=True))
    # grad_b2 = (1/batch_size)*(np.sum((yhat-y),axis=1,keepdims=True))
    return grad_W1, grad_W2
def compute_cost(y,yhat,batch_size):
    logprobs = np.multiply(np.log(yhat),y)
    cost = - 1/batch_size * np.sum(logprobs)
    cost = np.squeeze(cost)
    return cost

def calculate_loss(u,ctx):
    sum_1 = 0
    for index in np.where(ctx==1)[0]:
        sum_1 = sum_1 + u[index]
    sum_1 = -sum_1
    sum_2 = len(np.where(ctx==1)[0]) * np.log(np.sum(np.exp(u)))
    total_loss = sum_1 + sum_2
    return total_loss

def gradient_descent(corpus, word_to_index, N ,vocab_size ,window_size, alpha , num_iters,batch_size=128):
    # W1 = np.random.uniform(-1, 1, (N,vocab_size))
    # W2 = np.random.uniform(-1, 1, (vocab_size,N))
    W1,W2=initialize_model(N,vocab_size,282)
    training_data,training_sample_words=generate_training_data(corpus, window_size ,vocab_size,word_to_index,length_of_corpus,sample=False)
    iters=0
    # epoch_loss = []
    # weights_1 = []
    # weights_2 = []
    for x,y in get_batches(corpus, word_to_index,vocab_size, N ,128):
        # for x,y in training_data:
        y_predicted, h, u =forward_prop(W1,W2,x)
        total_error=calculate_error(y_predicted,y)
        cost=compute_cost(y,y_predicted, batch_size=128)
            # W1,W2 = backward_prop(W1,W2,total_error, h, x, alpha )
        # loss_temp = calculate_loss(u,y)
            # print('loss_temp=',loss_temp)
        if ( (iters+1) % 10 == 0):
            print(f"iters: {iters + 1} cost: {cost:.6f}")
        grad_W1, grad_W2= back_prop(x, y_predicted, y, h, W1, W2, batch_size)
        W1 = W1-(alpha*grad_W1)
        W2 = W2-(alpha*grad_W2)
        iters += 1
        if iters == num_iters:
            break
        if iters % 100 == 0:
            alpha *= 0.66

    return np.array(W1),np.array(W2)
W1, W2 = gradient_descent(corpus, word_to_index, 50, vocab_size, 3, 0.03 , 150 ,128)
W=W1.T
def embedings(words,word_to_index,W):
    idx = word_to_index[words]
    X=W[idx]
    return X
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
x0=embedings('king',word_to_index,W)
x1=embedings('queen',word_to_index,W)
x2=embedings('man',word_to_index,W)
x3=embedings('woman',word_to_index,W)
y0=x0-x1+x2
y1=nearest_neighbor(y0, W, k=1)
print('y1=',y1)
print('y1=',index_to_word[y1[0]])
