import numpy as np
from sklearn.feature_extraction.text import *
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import normalize
import time

def read_file(filename):
    lines = []
    with open(filename) as file:
        for line in file:
            # line = " ".join(line.split())
            lines.append(line[:-1])  # Git rid of the last character \n
    return lines

def read_interested_file(filename, keywords):  # Read lines with at least one keyword
    lines = []
    with open(filename) as file:
        for line in file:
            lines.append(line[:-1])  # Git rid of the last character \n
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
            if i >= num:
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

def find_general_terms(input_dir, node_dir, center_names, keywords_file):
    # node_dir = '/Users/wanzheng/Desktop/local-embedding/data/dblp/non-para-00/'
    print('[find_general_terms] Begin finding general terms')
    begin = time.time()
    print 'Reading files...'
    keywords = read_file(keywords_file)
    filename = input_dir + 'papers.txt'
    lines = read_file(filename)
    # lines = read_interested_file(filename, keywords)
    filename1 = node_dir + 'paper_cluster.txt'
    number = max(int(0.01*len(lines)), 50000)
    training_data_index, training_label = read_file1(filename1, number)
    lines_used = []
    for i in training_data_index:
        lines_used.append(lines[i])
    del lines  # Save memory

    print 'Initializing tf-idf...'
    vectorizer = TfidfVectorizer(min_df=10)
    A = vectorizer.fit_transform(lines_used)
    print A._shape
    tfidf = A.toarray()

    print 'Trim tf-idf to only interested keywords'
    temp = []
    for i in keywords:
        if i in vectorizer.vocabulary_:
            temp.append(vectorizer.vocabulary_.get(i))
    tfidf = np.array(tfidf[:, temp])

    print 'Original size is: ' + str(tfidf.shape)
    invalid = np.where(np.sum(tfidf, 1) == 0)
    print 'Number of invalid entries is: ' + str(len(invalid[0]))
    tfidf = np.delete(tfidf, invalid, 0)
    training_label = np.delete(np.array(training_label), invalid, 0)
    print 'Truncated size is: ' + str(tfidf.shape)

    print 'Running LASSO...'
    tfidf = normalize(tfidf, norm='l2', axis=0)  # axis: 1: normalize each data; 0: normalize each feature. 0 is better than 1.
    clf = SGDClassifier(loss="hinge", penalty="l1", alpha=0.00005, n_jobs=20, max_iter=5)  # Alpha for dblp is
    clf.fit(tfidf, training_label)
    dimension = len(tfidf[0])
    del tfidf  # To save memory

    print 'Writing to file...'
    filename_to_write = node_dir + 'repre_keyword.txt'
    general_keyword_to_write = node_dir + 'general_keyword.txt'
    nondetermining_terms = [[] for i in range(len(clf.coef_))]
    specific_terms = [[] for i in range(len(clf.coef_))]

    with open(filename_to_write, 'w') as fout:
        cluster_num = 0
        for weight in clf.coef_:
            keyword_number = 0
            print sum(weight)
            fout.write('cluster number ' + str(cluster_num) + ': ' + center_names[cluster_num] + '\n')
            representativeness_order = np.argsort(abs(np.array(weight)))[::-1]
            for i in representativeness_order:
                keyword_i = vectorizer.vocabulary_.keys()[vectorizer.vocabulary_.values().index(temp[i])]
                if weight[i] == 0:  # Write to general terms
                    nondetermining_terms[cluster_num].append(keyword_i)
                    continue
                if weight[i] > 1:  # Write to specific terms. The importance of this word is above threshold
                    specific_terms[cluster_num].append(keyword_i)
                if keyword_number < 20:
                    if weight[i] > 0:  # Only output positive value, since positive value means learning towards this cluster
                        fout.write(keyword_i + '\t' + str(weight[i]) + '\n')
                keyword_number = keyword_number + 1
            cluster_num = cluster_num + 1
            fout.write('\n')
            print 'Non-zero percentage: ' + str(keyword_number) + '/' + str(dimension)

    for i in range(len(specific_terms)):
        print('[find_general_terms] Specific terms: ', specific_terms[i].__len__())
        with open(node_dir + 'repre_cand_' + str(i) + '.txt', 'w') as fout:
            for j in specific_terms[i]:
                fout.write(j + '\n')

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

    print('[find_general_terms] General terms: ', general_terms.__len__())
    end = time.time()
    print('[find_general_terms] Finish finding general terms using time %s second' % (end-begin))
    return general_terms, specific_terms

    # print 'Running LASSO'
    # for alpha in [0, 0.01, 0.1, 0.3, 0.5, 1, 2]:
    #     print 'alpha is: ' + str(alpha)
    #     clf = Lasso(alpha=alpha)
    #     clf.fit(tfidf, training_label)
    #     print clf.coef_
    #     print sum(clf.coef_)
