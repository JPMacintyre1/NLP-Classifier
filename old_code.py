"""
def get_word_counts(training_set):
    neg_dictionary = {}
    pos_dictionary = {}
    for i in range (len(training_set)):
        if training_set[i][0] == "N":
            for word in training_set[i][1]:
                if word in neg_dictionary:
                    neg_dictionary[word] += 1
                else:
                    neg_dictionary[word] = 1
        else:
            for word in training_set[i][1]:
                if word in pos_dictionary:
                    pos_dictionary[word] += 1
                else:
                    pos_dictionary[word] = 1

    neg_sorted_words = sorted(neg_dictionary.items(), key=lambda item: item[1])
    pos_sorted_words = sorted(pos_dictionary.items(), key=lambda item: item[1])

    combined_counts = {}
    for word, count in neg_sorted_words + pos_sorted_words:
        if word in combined_counts:
            combined_counts[word] += count
        else:
            combined_counts[word] = count
    all_sorted_words = sorted(list(combined_counts.items()), key=lambda item: item[1])
    return neg_sorted_words, pos_sorted_words, all_sorted_words

def get_term_freq(training_set, all_tf):
    for doc_no, (det, document) in enumerate(training_set):
        counts = {}
        tf = {}
        for term in document:
            if term in counts:
                counts[term] += 1
            else:
                counts[term] = 1
        document_length = len(document)
        for word in counts.keys():
            tf[word] = counts[word]/document_length
        all_tf.append(tf)
    return all_tf

all_input_doc_ratings = {}
for id in neg_list:
    rating = id.split('_')[1].split('.')[0]
    if rating in all_input_doc_ratings.keys():
        all_input_doc_ratings[rating] = all_input_doc_ratings[rating] + 1
    else:
        all_input_doc_ratings[rating] = 1
for id in pos_list:
    rating = id.split('_')[1].split('.')[0]
    if rating in all_input_doc_ratings.keys():
        all_input_doc_ratings[rating] = all_input_doc_ratings[rating] + 1
    else:
        all_input_doc_ratings[rating] = 1
print(all_input_doc_ratings)

neg_set_size = len(neg_list)
pos_set_size = len(pos_list)
total_data_set_size = neg_set_size + pos_set_size

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


        for doc_id in doc_term_freq.keys():
            terms = doc_term_freq.get(doc_id)
            if term in terms.keys():
                doc_count += 1
        doc_idfs[term] = math.log(float(len(doc_term_freq))/float(1 + doc_count), 10)
    return term_idfs

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
"""
from nltk.corpus import wordnet as wn
from collections import defaultdict
from collections import Counter
text = "This is one of the dumbest films, I've ever seen. It rips off nearly ever type of thriller and manages to make a mess of them all. There's not a single good line or character in the whole mess. If there was a plot, it was an afterthought and as far as acting goes, there's nothing good to say so Ill say nothing. I honestly cant understand how this type of nonsense gets produced and actually released, does somebody somewhere not at some stage think, 'Oh my god this really is a load of shite' and call it a day. Its crap like this that has people downloading illegally, the trailer looks like a completely different film, at least if you have download it, you haven't wasted your time or money Don't waste your time, this is painful."
text = text.split()
all_terms = Counter(text)
final_output = defaultdict(float)
output = defaultdict(float)
for term, count in all_terms.items():
    synsets = wn.synsets(term)
    if synsets:
        synset = synsets[0]
        hypernyms = synset.hypernyms()
        for hypernym in hypernyms:
            lemma = hypernym.lemma_names()[0].lower()
            output[lemma] += 0.05
for hypernym, count in output.items():
    if all_terms.get(hypernym):
        all_terms[hypernym] += count
print(all_terms)