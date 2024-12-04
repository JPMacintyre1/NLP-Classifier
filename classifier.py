import os
import re
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import word_tokenize
import string
import math
import numpy as np
from sklearn.naive_bayes import MultinomialNB


# path for input text files
data_path = "data\\data\\"
manual_stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now", "/br", "/the"]

# separate into positive and negative
neg_list = os.listdir(data_path + "neg")
pos_list = os.listdir(data_path + "pos")


neg_set_size = len(neg_list)
pos_set_size = len(pos_list)
total_data_set_size = neg_set_size + pos_set_size

training_set = {}  # 80% 3200
dev_set = {}  # 10% 400
test_set = {}  # 10% 400

# populate data_sets with ratios specified above
for i in range(neg_set_size):
    if i < neg_set_size*0.8:
        with open(data_path + "neg\\" + neg_list[i], "r", encoding="utf8") as f:
            training_set[neg_list[i]] = f.read()
        with open(data_path + "pos\\" + pos_list[i], "r", encoding="utf8") as f:
            training_set[pos_list[i]] = f.read()
    elif i < neg_set_size*0.9:
        with open(data_path + "neg\\" + neg_list[i], "r", encoding="utf8") as f:
            dev_set[neg_list[i]] = f.read()
        with open(data_path + "pos\\" + pos_list[i], "r", encoding="utf8") as f:
            dev_set[pos_list[i]] = f.read()
    else:
        with open(data_path + "neg\\" + neg_list[i], "r", encoding="utf8") as f:
            test_set[neg_list[i]] = f.read()
        with open(data_path + "pos\\" + pos_list[i], "r", encoding="utf8") as f:
            test_set[pos_list[i]] = f.read()

def manual_tokenize(training_set):
    # tokenize all data in training set (whitespace)
    for i in range (len(training_set)):
        # tokenize by whitespace and set to lowercase
        training_set[i][1] = (training_set[i][1]).lower().split()
        # remove irrelevant characters
        for j in range(len(training_set[i][1])):
            training_set[i][1][j] = re.sub('[<>@#$?.-//,]', '', training_set[i][1][j])

        training_set[i][1] = [word for word in training_set[i][1] if word not in manual_stopwords]

def nltk_tokenize(training_set, stopwords):
    stopwords = set(stopwords.words('english'))
    for key in training_set.keys():
        tokens = word_tokenize(training_set[key])
        output = []
        for word in tokens:
            # Remove irrelevant chars
            if (not word in ["``", "''", "br", "I", "n't", "'s"]) and (not word in stopwords) and (not word in string.punctuation):
                output.append(word.lower()) # Make word lowercase
        training_set[key] = output
    return training_set

def collect_vocabulary(training_set):
    all_terms = []
    for key in training_set.keys():
        text = training_set[key]
        for word in text:
            all_terms.append(word)
    return list(set(all_terms))

def count_terms(training_set):
    doc_term_freqs = {}
    for doc_id in training_set.keys():
        term_freqs = {}
        for word in training_set[doc_id]:
            if word in term_freqs:
                term_freqs[word] += 1
            else:
                term_freqs[word] = 1
        doc_term_freqs[doc_id] = term_freqs
    return doc_term_freqs

def vectorize_tf(vocabulary, doc_term_freq):
    output = {}
    for doc_id in doc_term_freq.keys():
        terms = doc_term_freq.get(doc_id)
        output_vector = []
        for word in vocabulary:
            if word in terms.keys():
                output_vector.append(int(terms.get(word)))
            else:
                output_vector.append(0)
        output[doc_id] = output_vector
    return output

def calculate_idfs(vocabulary, doc_term_freq):
    doc_idfs = {}
    for term in vocabulary:
        doc_count = 0  # no. of documents containing the term
        for doc_id in doc_term_freq.keys():
            terms = doc_term_freq.get(doc_id)
            if term in terms.keys():
                doc_count += 1
        doc_idfs[term] = math.log(float(len(doc_term_freq))/float(1 + doc_count), 10)
    return doc_idfs

def vectorize_idf(doc_term_freq, doc_idfs, all_terms):
    output = {}
    for doc_id in doc_term_freq.keys():
        terms = doc_term_freq.get(doc_id)
        output_vector = []
        for term in all_terms:
            if term in terms.keys():
                output_vector.append(doc_idfs.get(term)*float(terms.get(term)))
            else:
                output_vector.append(float(0))
        output[doc_id] = output_vector
    return output

def get_target_values(training_set):
    output = []
    for doc_id in training_set.keys():
        rating = doc_id.split('_')[1].split('.')[0]
        output.append(rating)
    return output

training_set = nltk_tokenize(training_set, stopwords)
all_terms = collect_vocabulary(training_set)
doc_term_freqs = count_terms(training_set)
doc_idfs = calculate_idfs(all_terms, doc_term_freqs)
v_doc_tf = vectorize_tf(all_terms, doc_term_freqs)
v_idf = vectorize_idf(doc_term_freqs, doc_idfs, all_terms)

target_values = get_target_values(training_set)

dev_set = nltk_tokenize(dev_set, stopwords)
dev_dtf = count_terms(dev_set)
dev_v_idf = vectorize_idf(dev_dtf, doc_idfs, all_terms)
dev_target_values = get_target_values(dev_set)

X = np.array([v_idf[i] for i in v_idf.keys()])
Y = np.array([dev_v_idf[i] for i in dev_v_idf.keys()])
print(X.shape)
print(Y.shape)

clf = MultinomialNB()
clf.fit(X, target_values)
print(clf.score(X, target_values))
print(clf.score(Y, dev_target_values))
    
