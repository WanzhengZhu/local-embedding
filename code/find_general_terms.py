import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.linear_model import Lasso, SGDClassifier
import sys

def read_file(filename):
    lines = []
    with open(filename) as file:
        for line in file:
            line = " ".join(line.split())  # Get rid of the sentence ID
            lines.append(line)
    return lines

def read_file1(filename, num):
    training_data = []
    training_label = []
    i=0
    with open(filename) as file:
        for line in file:
            training_data.append(int(line.split()[0]))
            training_label.append(int(line.split()[1]))
            i = i+1
            if i > num:
                break
    return training_data, training_label

def read_wordEmb(filename):
    words = []
    wordEmb = []
    with open(filename) as file:
        for line in file:
            words.append(line.split()[0])  # Vocabulary
            wordEmb.append(map(float, line.split()[1:])) # word vectors
    return (words,wordEmb)

def Normalize(senvec):
    norm = 0
    for x in senvec:
        norm += x**2
    senvec = [x/(norm**0.5) for x in senvec]
    return senvec

corpus = 'dblp'
print 'Reading files...'
filename = '../data/' + corpus + '/input/papers.txt'
lines = read_file(filename)
filename1 = '../data/' + corpus + '/non-para/paper_cluster.txt'
training_data_index, training_label = read_file1(filename1, num=0.01 * len(lines))
lines_used = []
for i in training_data_index:
    lines_used.append(lines[i])
del lines

print 'Initializing tf-idf...'
vectorizer = TfidfVectorizer(min_df=10)
A = vectorizer.fit_transform(lines_used)
print A._shape
tfidf = A.toarray()

print 'Running LASSO...'
clf = SGDClassifier(loss="log", penalty="l1", alpha=0.00001, n_jobs=20)
clf.fit(tfidf, training_label)

print 'Writing to file...'
filename_to_write = '../data/' + corpus + '/non-para/repre_keyword.txt'
general_keyword_to_write = '../data/' + corpus + '/non-para/general_keyword.txt'
general_terms = [[] for i in range(5)]

with open(filename_to_write, 'w') as fout:
    # with open(general_keyword_to_write, 'w') as fout1:
        cluster_num = 0
        for weight in clf.coef_:
            keyword_number = 0
            print sum(weight)
            fout.write('cluster number ' + str(cluster_num) + ': \n')
            # fout1.write('cluster number ' + str(cluster_num) + ': \n')
            representativeness_order = np.argsort(abs(np.array(weight)))[::-1]
            for i in representativeness_order:
                if weight[i] == 0:
                    general_terms[cluster_num].append(vectorizer.vocabulary_.keys()[vectorizer.vocabulary_.values().index(i)])
                #     fout1.write(vectorizer.vocabulary_.keys()[vectorizer.vocabulary_.values().index(i)] + '\t' + str(weight[i]) + '\n')
                # if keyword_number > 10:
                #     break
                else:
                    fout.write(vectorizer.vocabulary_.keys()[vectorizer.vocabulary_.values().index(i)] + '\t' + str(weight[i]) + '\n')
                    keyword_number = keyword_number + 1
            cluster_num = cluster_num + 1
            fout.write('\n')
            # fout1.write('\n')
            print 'Non-zero percentage: ' + str(float(keyword_number)/len(tfidf[0]))

result = set(general_terms[0])
for s in general_terms[1:]:
    result.intersection_update(s)
print result

with open(general_keyword_to_write, 'w') as fout:
    for i in result:
        fout.write(i + '\n')

print 'All Done...'

# print 'Running LASSO'
# for alpha in [0, 0.01, 0.1, 0.3, 0.5, 1, 2]:
#     print 'alpha is: ' + str(alpha)
#     clf = Lasso(alpha=alpha)
#     clf.fit(tfidf, training_label)
#     print clf.coef_
#     print sum(clf.coef_)
