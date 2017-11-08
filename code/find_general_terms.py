import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.linear_model import SGDClassifier
import time

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
    i = 0
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

def find_general_terms(input_dir, node_dir, center_names):
    # node_dir = '/Users/wanzheng/Desktop/local-embedding/data/dblp/non-para-00/'
    print('[find_general_terms] Begin finding general terms')
    begin = time.time()
    print 'Reading files...'
    filename = input_dir + 'papers.txt'
    lines = read_file(filename)
    filename1 = node_dir + 'paper_cluster.txt'
    number = max(int(0.01*len(lines)), 40000)
    training_data_index, training_label = read_file1(filename1, number)
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
    ## TODO: loss function can be tuned.
    clf = SGDClassifier(loss="log", penalty="l1", alpha=0.0001, n_jobs=20, max_iter=5)  # Alpha for dblp is
    clf.fit(tfidf, training_label)

    print 'Writing to file...'
    filename_to_write = node_dir + 'repre_keyword.txt'
    general_keyword_to_write = node_dir + 'general_keyword.txt'
    nondetermining_terms = [[] for i in range(len(clf.coef_))]

    with open(filename_to_write, 'w') as fout:
        # with open(general_keyword_to_write, 'w') as fout1:
            cluster_num = 0
            for weight in clf.coef_:
                keyword_number = 0
                print sum(weight)
                fout.write('cluster number ' + str(cluster_num) + ': ' + center_names[cluster_num] + '\n')
                # fout1.write('cluster number ' + str(cluster_num) + ': \n')
                representativeness_order = np.argsort(abs(np.array(weight)))[::-1]
                for i in representativeness_order:
                    if weight[i] == 0:
                        nondetermining_terms[cluster_num].append(vectorizer.vocabulary_.keys()[vectorizer.vocabulary_.values().index(i)])
                        continue
                    #     fout1.write(vectorizer.vocabulary_.keys()[vectorizer.vocabulary_.values().index(i)] + '\t' + str(weight[i]) + '\n')
                    if keyword_number < 20:
                        if weight[i] > 0:  # Only output positive value, since positive value means towards this cluster
                            fout.write(vectorizer.vocabulary_.keys()[vectorizer.vocabulary_.values().index(i)] + '\t' + str(weight[i]) + '\n')
                    keyword_number = keyword_number + 1
                cluster_num = cluster_num + 1
                fout.write('\n')
                # fout1.write('\n')
                print 'Non-zero percentage: ' + str(keyword_number) + '/' + str(len(tfidf[0]))

    # General terms for all clusters
    result = set(nondetermining_terms[0])
    for s in nondetermining_terms[1:]:
        result.intersection_update(s)
    # print result

    general_terms = []
    with open(general_keyword_to_write, 'w') as fout:
        for i in result:
            fout.write(i + '\n')
            general_terms.append(i)

    end = time.time()
    print('[find_general_terms] Finish finding general terms using time %s second' % (end-begin))
    return general_terms

    # print 'Running LASSO'
    # for alpha in [0, 0.01, 0.1, 0.3, 0.5, 1, 2]:
    #     print 'alpha is: ' + str(alpha)
    #     clf = Lasso(alpha=alpha)
    #     clf.fit(tfidf, training_label)
    #     print clf.coef_
    #     print sum(clf.coef_)
