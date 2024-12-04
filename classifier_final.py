import os
import re
import nltk
import random
#nltk.download('punkt_tab')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger_eng')
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import ngrams
from nltk import pos_tag
from nltk import FreqDist
from nltk.chunk import RegexpParser
from nltk.stem import *
from nltk.corpus import wordnet as wn
import string
import math
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from collections import Counter
from collections import defaultdict

def load_data():
    # path for input text files
    data_path = "data\\data\\"

    # separate into positive and negative
    neg_list = os.listdir(data_path + "neg")
    pos_list = os.listdir(data_path + "pos")

    all_reviews = neg_list + pos_list

    training_set = {}  # 80% 3200
    dev_set = {}  # 10% 400
    test_set = {}  # 10% 400

    all_doc_id, all_doc_rating = get_target_values(all_reviews)  # Retrieve all document ID's and associated ratings in 2 lists

    train_id, X_temp, train_ratings, y_temp = train_test_split(all_doc_id, all_doc_rating, test_size=0.2, random_state=42, stratify=all_doc_rating) # Split to training and temp

    dev_id, test_id, dev_ratings, test_ratings = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp) # Split temp into dev and test

    # Populate data sets
    training_set = populate_set(train_id, train_ratings, data_path)
    dev_set = populate_set(dev_id, dev_ratings, data_path)
    test_set = populate_set(test_id, test_ratings, data_path)

    train_rating_dict = {}
    dev_rating_dict = {}
    test_rating_dict = {}

    for rating in train_ratings:
        if rating in train_rating_dict.keys():
            train_rating_dict[rating] += 1
        else:
            train_rating_dict[rating] = 1
    for rating in dev_ratings:
        if rating in dev_rating_dict.keys():
            dev_rating_dict[rating] += 1
        else:
            dev_rating_dict[rating] = 1
    for rating in test_ratings:
        if rating in test_rating_dict.keys():
            test_rating_dict[rating] += 1
        else:
            test_rating_dict[rating] = 1
    print("Training set size: " + str(len(training_set)))
    print("Development set size: " + str(len(dev_set)))
    print("Test set size: " + str(len(test_set)))
    #print(train_rating_dict)
    #print(dev_rating_dict)
    #print(test_rating_dict)
    return training_set, dev_set, test_set


def get_target_values(data_set):
    output_id = []
    output_rating = []
    for doc_id in data_set:
        rating = doc_id.split('_')[1].split('.')[0]
        output_id.append(doc_id)
        output_rating.append(rating)
    return output_id, output_rating


def populate_set(data_set, data_set_rating, data_path):
    output_set = {}
    for rating_id, id in enumerate(data_set):
        if int(data_set_rating[rating_id]) < 5:
            with open(data_path + "neg\\" + id, "r", encoding="utf8") as f:
                output_set[id] = f.read()
        else:
            with open(data_path + "pos\\" + id, "r", encoding="utf8") as f:
                output_set[id] = f.read()
    return output_set


def nltk_tokenize(data_set, stemming, lemmatization, ngram_cond):
    output = {}
    stopword_list = set(stopwords.words('english'))
    st = PorterStemmer()
    lem = WordNetLemmatizer()
    grammar_NP = "NP: {<DT>?<JJ>*<NN>}"
    chunker = RegexpParser(grammar_NP)

    for doc_id, text in data_set.items():

        tokens = [word.lower() for word in word_tokenize(text) if word not in ["br", '``', "n't", "''", "...", "'s", "'re", "<", ">"] and word not in string.punctuation]  # Tokenize text and set to lowercase

        pos_tags = pos_tag(tokens)  # Generate POS tags
        
        tree = chunker.parse(pos_tags)  # Use chunking to gather noun phrases
        NP = []  # Noun Phrases
        for subtree in tree.subtrees():
            if subtree.label() == "NP":
                NP.append(" ".join(word for word, tag in subtree.leaves()))

        if stemming == True:
            tokens_edited = [st.stem(word) for word in tokens]
            NP_edited = [st.stem(phrase) for phrase in NP]
        elif lemmatization == True:
            tokens_edited = [lem.lemmatize(word) for word in tokens]
            NP_edited = [lem.lemmatize(phrase) for phrase in NP]
        else:
            tokens_edited = tokens
            NP_edited = NP

        ngram = []
        
        if ngram_cond == 2:
            ngram = [' '.join(gram) for gram in ngrams(tokens, 2)]  # generate bigrams using ngrams function
        if ngram_cond == 3:
            ngram = [' '.join(gram) for gram in ngrams(tokens, 3)]  # generate trigrams using ngrams function
        if ngram_cond == 4:
            ngram = [' '.join(gram) for gram in ngrams(tokens, 4)]  # generate quadgrams using ngrams function

        # Remove stopword, punctuation and misc.
        tokens_edited = [word for word in tokens_edited if word not in stopword_list and word not in string.punctuation and word not in ["br", '``', "n't", "''", "...", "'s", "'re"]]
        NP_edited = [word for word in NP_edited if word not in stopword_list and word not in string.punctuation and word not in ["br", '``', "n't", "''", "...", "'s", "'re"]]

        if ngram_cond != 0:
            combined_tokens = list(set(tokens_edited + NP_edited + ngram))  # Combine all to get an output
        else:
            combined_tokens = list(set(tokens_edited + NP_edited))
        
        output[doc_id] = combined_tokens
    return output


def boost_features(doc_term_freq_matrix):
    final_output = defaultdict(float)
    for doc_id, all_terms in doc_term_freq_matrix.items():
        output = defaultdict(float)
        for term, count in all_terms.items():  # Iterate through all terms in a document
            synsets = wn.synsets(term)  # Retrieve synonym sets of each term
            if synsets:
                synset = synsets[0]  # Extract the most common set
                hypernyms = synset.hypernyms()  # Find the hypernyms for that sysnset
                for hypernym in hypernyms:  # Iterate through the hypernyms
                    lemma = hypernym.lemma_names()[0].lower()
                    output[lemma] += 0.00001  # Extract the actual word/lemma and add a small count
        for hypernym, count in output.items():  # Iterate through all hypernyms that have been extracted
            if all_terms.get(hypernym):  # If the hypernym is present in the document then boost it's term frequency
                all_terms[hypernym] += count
        final_output[doc_id] = all_terms  # Append updated term frequency vector to output
    return final_output


def calc_udtf(data_set):
    udtf = defaultdict(int)
    for doc_id, tokens in data_set.items():
        unique_doc_terms = list(set(tokens))
        for term in unique_doc_terms:
            udtf[term] += 1
    return udtf


def plot_udtf(udtf):
    import matplotlib.pyplot as plt
    term_freqs = [count for word, count in sorted(udtf.items(), key=lambda item: item[1])]
    x_axis = np.arange(0, len(udtf), 1)
    plt.plot(x_axis, term_freqs)
    plt.xlabel("Terms")
    plt.ylabel("Unique Document Frequency")
    plt.show()


def calc_term_freqs(training_set, all_terms):
    output = {}
    for doc_id, tokens in training_set.items():  # Iterate through each document
        term_freqs = defaultdict(int)
        doc_term_freqs = dict(Counter(tokens))  # Generate dictionary containing term: count for all terms in document
        for word in all_terms:  # Filter terms by all_terms (which has had the UDF cutoff)
            if word in doc_term_freqs.keys():
                term_freqs[word] = doc_term_freqs[word] / len(tokens)  # Update document frequency row and normalise by document length
            else:
                term_freqs[word] = 0
        output[doc_id] = term_freqs  # Update output dictionary
    return output


def calc_idf(training_set, all_terms, unique_doc_freq):
    total_num_documents = len(training_set)
    output = defaultdict(int)
    for token in all_terms:
        output[token] = math.log(total_num_documents/(1+unique_doc_freq[token]), 10)  # IDF of a term = log(total documents/total documents term appears in)
    return output


def vec_idf(all_terms_idf):
    return np.array(list(all_terms_idf.values()))


def vectorize_tfm(doc_term_freq_matrix):
    vec_dtf = []
    for doc_id, tokens in doc_term_freq_matrix.items():  # Iterate through each document
        vec_dtf.append(list(tokens.values()))  # Add all values to a new row in the output vector
    return vec_dtf
    

def calc_tfidf(all_terms_idf, data_dtf):
    output = []
    for doc in data_dtf:  # Iterate through each row (document) in the document term frequency matrix
        output.append(all_terms_idf*doc)  # Multiply each document term frequency values by the IDF vector
    return np.array(output)


def calc_l2_norm(data_dtf):
    output = []
    for document in data_dtf:  # Iterate through each document

        l2_norm = np.sqrt(np.sum(np.array(document)**2))  # Calculate L2 normalisation factor
        
        output.append(document/l2_norm)  # Update the output vector with the document term frequency normalised by L2 factor

    return np.array(output)
        

def get_target_sentiment(data_set):
    output = []
    for doc_id in data_set.keys():
        rating = doc_id.split('_')[1].split('.')[0]
        if int(rating) >= 7:
            output.append("P")
        else:
            output.append("N")
    return output


class NBClassifier:
    def __init__(self, alpha, feature_count):
        self.alpha = alpha
        self.log_class_conditional_likelihoods = 0
        self.log_class_priors = 0
        self.feature_count = feature_count


    def estimate_log_class_priors(self):
        #  Log class priors should be equal as training data is equally distributed between positive and negative reviews
        log_class_priors = np.array((np.log(0.5), np.log(0.5)))
        return log_class_priors
    

    def get_class_values(self, tfidf_vec, target_values):
        # Initialise return variables
        total_tfidfs_pos = 0  # Stores total TFIDF score of all terms in positive class
        total_tfidfs_neg = 0  # Stores total TFIDF score of all terms in negative class
        tfidfs_pos = np.zeros_like(tfidf_vec[0])  # Stores the total TFIDF score for each term across positive documents
        tfidfs_neg = np.zeros_like(tfidf_vec[0])  # Stores the total TFIDF score for each term across negative documents
        # Iterate through each row (document) in the TFIDF vector
        for i in range(len(tfidf_vec)):
            sentiment = target_values[i]  # Extract sentiment
            if sentiment == "P":
                total_tfidfs_pos += np.sum(tfidf_vec[i])  # Add the summed TFIDF scores for all terms in document
                tfidfs_pos += tfidf_vec[i]  # Update TFIDF vector by adding each term TFIDF rating (vectorized operation)
            else:  # Do the same for negative documents
                total_tfidfs_neg += np.sum(tfidf_vec[i])
                tfidfs_neg += tfidf_vec[i]
        return total_tfidfs_pos, total_tfidfs_neg, tfidfs_pos, tfidfs_neg
        

    def estimate_log_class_conditional_likelihoods(self, total_tfidfs_pos, total_tfidfs_neg, tfidfs_pos, tfidfs_neg):
        alpha = self.alpha  # Alpha value used for laplace smoothing
        PPOS = np.log((tfidfs_pos + alpha)/(total_tfidfs_pos + alpha * len(tfidfs_pos)))  # Vector corresponds to each unique term and it's associated class conditional likelihood
        PNEG = np.log((tfidfs_neg + alpha)/(total_tfidfs_neg + alpha * len(tfidfs_neg)))
        theta = np.array((PPOS, PNEG))
        return theta

    def predict(self, new_data):
        class_predictions = np.empty(len(new_data), dtype=str)
        for document, terms in enumerate(new_data):  # Iterate through each document (row)
            PosRating = self.log_class_priors[0] + np.sum(terms * self.log_class_conditional_likelihoods[0])  # Calculate positive posterior probability
            NegRating = self.log_class_priors[1] + np.sum(terms * self.log_class_conditional_likelihoods[1])  # Calculate negative posterior probability
            if PosRating > NegRating:
                class_predictions[document] = "P"
            else:
                class_predictions[document] = "N"
        return class_predictions
    
    def train(self, tfidf_vec, target_values):
        self.log_class_priors = self.estimate_log_class_priors()
        total_tfidfs_pos, total_tfidfs_neg, tfidfs_pos, tfidfs_neg = self.get_class_values(tfidf_vec, target_values)
        self.log_class_conditional_likelihoods = self.estimate_log_class_conditional_likelihoods(total_tfidfs_pos, total_tfidfs_neg, tfidfs_pos, tfidfs_neg)


def create_classifier(features, tfidf_vec, target_values):
    classifier = NBClassifier(alpha=1.0, feature_count=features)
    classifier.train(tfidf_vec, target_values)
    return classifier


def main():
    # Load text data into variables
    training_set, dev_set, test_set = load_data()

    # Apply tokenization and other various methods controlled by True/False => Dict[Doc_ID] : Tokens
    training_set = nltk_tokenize(training_set, stemming=False, lemmatization=True, ngram_cond = 2)
    dev_set = nltk_tokenize(dev_set, stemming=False, lemmatization=True, ngram_cond = 2)
    test_set = nltk_tokenize(test_set, stemming=False, lemmatization=True, ngram_cond = 2)

    # Calculates unique doc term frequency to perform token cutoff
    unique_doc_term_frequency = calc_udtf(training_set)

    plot_udtf(unique_doc_term_frequency)  # Used for plotting udtf

    # Update all_terms to include only tokens after cutoff
    #all_terms = [word for word, count in sorted(unique_doc_term_frequency.items(), key=lambda item: item[1]) if 200 >= count >= 7]  # Cutoff features here!
    all_terms = {word: count for word, count in sorted(unique_doc_term_frequency.items(), key=lambda item: item[1]) if 200 >= count >= 7}

    # Calculate IDF for all filtered vocabulary and vectorize
    all_terms_idf = calc_idf(training_set=training_set, all_terms=all_terms, unique_doc_freq=unique_doc_term_frequency)  # Dictionary => Dict[Token] : IDF
    vec_all_terms_idf = vec_idf(all_terms_idf)  # List of idf values

    # Term Frequency Matrix generation
    doc_term_freq_matrix = calc_term_freqs(training_set, all_terms)  # Retrieve term frequency values for each term in each document => Dict[Doc_ID] : Dict[Token] : TF (normalised)
    dev_doc_tfm = calc_term_freqs(dev_set, all_terms)  # Retrieve term frequency values for each term in each dev document => Dict[Doc_ID] : Dict[Token] : TF (normalised)
    test_doc_tfm = calc_term_freqs(test_set, all_terms)

    # Feature boosting variable
    feature_boosting = False

    if feature_boosting == True:  # Feature boosting
        dtf_train_boosted = boost_features(doc_term_freq_matrix)
        dtf_dev_boosted = boost_features(dev_doc_tfm)
        dtf_test_boosted = boost_features(test_doc_tfm)
        vec_term_freq_matrix_train = vectorize_tfm(dtf_train_boosted)
        vec_tfm_dev = vectorize_tfm(dtf_dev_boosted)
        vec_tfm_test = vectorize_tfm(dtf_test_boosted)
    else:  # Vectorise term frequency matrix to help calculate TFIDF efficiently
        vec_term_freq_matrix_train = vectorize_tfm(doc_term_freq_matrix)
        vec_tfm_dev = vectorize_tfm(dev_doc_tfm)
        vec_tfm_test = vectorize_tfm(test_doc_tfm)


    # L2_normalisation variable
    l2_normalisation = False
    if l2_normalisation == True:  # L2 Normalisation
        vec_l2_norm_train = calc_l2_norm(vec_term_freq_matrix_train)
        vec_l2_norm_dev = calc_l2_norm(vec_tfm_dev)
        vec_l2_norm_test = calc_l2_norm(vec_tfm_test)


    # TFIDF variable
    TFIDF = True
    if TFIDF == True:  # TFIDF Normalisation
        vec_tfidf_train = calc_tfidf(all_terms_idf=vec_all_terms_idf, data_dtf=vec_term_freq_matrix_train)
        vec_tfidf_dev = calc_tfidf(all_terms_idf=vec_all_terms_idf, data_dtf=vec_tfm_dev)
        vec_tfidf_test = calc_tfidf(all_terms_idf=vec_all_terms_idf, data_dtf=vec_tfm_test)


    # Retrieve target values for each document
    training_target_values = get_target_sentiment(training_set)
    dev_target_values = get_target_sentiment(dev_set)
    test_target_values = get_target_sentiment(test_set)

    # Should be same dimension as all_terms, but to double check normalisation + vectorisation has been performed correctly
    feature_count = vec_tfidf_train.shape[1]


    # Classifier functions
    print("Number of features: " + str(feature_count))

    if TFIDF == True:  # TFIDF Classifier
        clf = MultinomialNB(alpha=1.0)
        clf.fit(vec_tfidf_train, training_target_values)
        print("SKLearn Training Accuracy: " + str(clf.score(vec_tfidf_train, training_target_values)))
        print("SKLearn Development Accuracy: " + str(clf.score(vec_tfidf_dev, dev_target_values)))
        print("SKLearn Test Accuracy: " + str(clf.score(vec_tfidf_test, test_target_values)))
        max_alpha = 0.0
        max_accuracy = 0.0
        alpha_testing = False  # Update this to True to test for optimal alpha values
        if alpha_testing == True:
            # Function below tests optimal alpha values for laplace smoothing
            for i in np.arange(0.01, 1.01, 0.01):
                clf = MultinomialNB(alpha=i)
                clf.fit(vec_tfidf_train, training_target_values)
                accuracy = clf.score(vec_tfidf_dev, dev_target_values)
                if accuracy >= max_accuracy:
                    max_accuracy = accuracy
                    max_alpha = i
            print(max_alpha, max_accuracy)

    if l2_normalisation == True:  # L2 Classifier
        clf_l2 = MultinomialNB(alpha=1.0)
        clf_l2.fit(vec_l2_norm_train, training_target_values)
        print("Training Accuracy (L2): " + str(clf_l2.score(vec_l2_norm_train, training_target_values)))
        print("Development Accuracy (L2): " + str(clf_l2.score(vec_l2_norm_dev, dev_target_values)))
        print("Test Accuracy (L2): " + str(clf_l2.score(vec_l2_norm_test, test_target_values)))


    # Manual NB Classifier
    classifier = create_classifier(feature_count, vec_tfidf_train, training_target_values)
    dev_predictions = classifier.predict(vec_tfidf_dev)
    test_predictions = classifier.predict(vec_tfidf_test)
    correct_count = 0
    for index, pred in enumerate(dev_predictions):
        if pred == dev_target_values[index]:
            correct_count += 1
    print("Manual NB Classifier Development Accuracy: " + str(correct_count/400))
    correct_count = 0
    for index, pred in enumerate(test_predictions):
        if pred == test_target_values[index]:
            correct_count += 1
    print("Manual NB Classifier Test Accuracy: " + str(correct_count/400))
    




if __name__ == "__main__":
    main()

