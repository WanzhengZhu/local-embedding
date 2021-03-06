'''
__author__: Chao Zhang
__description__: A wrapper for spherecluster, implement the term clustering component.
__latest_updates__: 09/25/2017
'''
from collections import defaultdict
from scipy.spatial.distance import cosine
from spherecluster import SphericalKMeans
from dataset import SubDataSet
from find_general_terms import find_general_terms
import numpy as np

class Clusterer:

    def __init__(self, data, keywords, n_cluster):
        self.data = data
        self.keywords = keywords
        self.n_cluster = n_cluster
        self.clus = SphericalKMeans(n_cluster)
        self.clusters = defaultdict(list)  # cluster id -> members
        self.membership = None  # a list contain the membership of the data points
        self.center_ids = None  # a list contain the ids of the cluster centers
        self.inertia_scores = None

    def fit(self):
        self.clus.fit(self.data)
        labels = self.clus.labels_
        print(labels)
        for i in range(self.n_cluster):
            print('Cluster ', i, ': ', sum(labels == i))
        for idx, label in enumerate(labels):
            self.clusters[label].append(idx)
        self.membership = labels
        self.center_ids = self.gen_center_idx()
        self.inertia_scores = self.clus.inertia_
        print('Sum of distances of samples to their closest cluster center:', self.inertia_scores)
        for i in range(self.n_cluster):
            sum_dist = 0
            for j in range(len(self.clusters[i])):
                sum_dist += np.linalg.norm(self.data[self.clusters[i][j]] - self.clus.cluster_centers_[i])**2
            print("Average distances for Cluster " + str(i) + " (" + self.keywords[self.center_ids[i][1]] + "): " + str(sum_dist / sum(labels == i)))

    # find the idx of each cluster center
    def gen_center_idx(self):
        ret = []
        for cluster_id in range(self.n_cluster):
            center_idx = self.find_center_idx_for_one_cluster(cluster_id)
            ret.append((cluster_id, center_idx))
        return ret


    def find_center_idx_for_one_cluster(self, cluster_id):
        query_vec = self.clus.cluster_centers_[cluster_id]
        members = self.clusters[cluster_id]
        best_similarity, ret = -1, -1
        for member_idx in members:
            member_vec = self.data[member_idx]
            cosine_sim = self.calc_cosine(query_vec, member_vec)
            if cosine_sim > best_similarity:
                best_similarity = cosine_sim
                ret = member_idx
        return ret

    def update_center_ids(self, n_cluster):
        # Update the center id of each cluster. Since many similar words (SVM, svms, Support_vector_machines) might drag
        #  the center of a cluster towards themselves, this function do a better selection of a center.
        print('Updating cluster centers')
        center_idx = []
        for cluster_id in range(self.n_cluster):
            center_idx.append(self.update_one_center_id(cluster_id, n_cluster))
        self.new_center_ids = center_idx

    def update_one_center_id(self, cluster_id, n_cluster):
        # Read the data, Perform k-means
        data = self.data[self.clusters[cluster_id]]
        ##  1. Consider the frequency and re-calculate the new mean.
        # new_mean = sum(data)
        ##  2. Just the average of sub-clusters
        clus = SphericalKMeans(n_cluster)
        clus.fit(data)
        new_mean = sum(clus.cluster_centers_)

        # Normalize new_mean
        norm = 0
        for x in new_mean:
            norm += x ** 2
        new_mean = [x / (norm ** 0.5) for x in new_mean]
        # print('Cos sim for old mean and new mean is: ', cossim(new_mean, clus.cluster_centers_[cluster_id]))
        # Find closest index to new_mean
        members = self.clusters[cluster_id]
        best_similarity, ret = -1, -1
        for member_idx in members:
            member_vec = self.data[member_idx]
            cosine_sim = self.calc_cosine(new_mean, member_vec)
            if cosine_sim > best_similarity:
                best_similarity = cosine_sim
                ret = member_idx
        return (cluster_id, ret)


    def write_keywords_to_file(self, keywords, parent_direcotry):
        for i in range(self.n_cluster):
            filename = parent_direcotry + 'cluster_' + str(i) + '.txt'
            with open(filename, 'w') as fout:
                fout.write(keywords[self.center_ids[i][1]] + ': \n')
                for j in range(len(self.clus.labels_)):
                    if self.clus.labels_[j] == i:
                        fout.write(keywords[j] + '\n')

        filename = parent_direcotry + 'label.txt'
        with open(filename, 'w') as fout:
            for i in self.clus.labels_:
                fout.write(str(i) + '\n')

    def calc_cosine(self, vec_a, vec_b):
        return 1 - cosine(vec_a, vec_b)


def run_clustering(dataset, filter_keyword_file, n_cluster, parent_direcotry, parent_description,\
                   cluster_keyword_file, hierarchy_file, doc_membership_file, cluster_keyword_embedding, \
                   cluster_keyword_label, update_center, input_dir):
    print('Start clustering for ', len(dataset.keywords), ' keywords under parent:', parent_description)
    ## TODO: change later here for n_cluster selection from a range
    clus = Clusterer(dataset.embeddings, dataset.keywords, n_cluster)
    clus.fit()
    if update_center:
        clus.update_center_ids(n_cluster)  # To find the general terms
    print('Done clustering for ', len(dataset.keywords), ' keywords under parent:', parent_description)
    # clus.write_keywords_to_file(dataset.keywords, parent_direcotry)
    dataset.write_document_membership(clus, doc_membership_file, parent_direcotry)
    center_names = dataset.write_to_hierarchy(clus, parent_description, hierarchy_file)
    general_terms, specific_terms = find_general_terms(input_dir, parent_direcotry, center_names, filter_keyword_file)
    # general_terms = []
    # specific_terms = []
    dataset.write_cluster_members(clus, cluster_keyword_file, parent_direcotry, cluster_keyword_embedding, cluster_keyword_label, general_terms, specific_terms)
    print('Done saving cluster results for ', len(dataset.keywords), ' keywords under parent:', parent_description)
    return center_names, len(dataset.keywords)
