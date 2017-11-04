'''
__author__: Chao Zhang
__description__: A wrapper for spherecluster, implement the term clustering component.
__latest_updates__: 09/25/2017
'''
from collections import defaultdict

from scipy.spatial.distance import cosine
from spherecluster import SphericalKMeans
from dataset import SubDataSet

class Clusterer:

    def __init__(self, data, n_cluster):
        self.data = data
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
        print('Clustering concentration score:', self.inertia_scores)

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


def run_clustering(full_data, doc_id_file, filter_keyword_file, n_cluster, parent_direcotry, parent_description,\
                   cluster_keyword_file, hierarchy_file, doc_membership_file, cluster_keyword_embedding, cluster_keyword_label, filter_keyword, iter):
    dataset = SubDataSet(full_data, doc_id_file, filter_keyword_file, filter_keyword, iter)
    print('Start clustering for ', len(dataset.keywords), ' keywords under parent:', parent_description)
    ## TODO: change later here for n_cluster selection from a range
    clus = Clusterer(dataset.embeddings, n_cluster)
    clus.fit()
    clus.write_keywords_to_file(dataset.keywords, parent_direcotry)
    print('Done clustering for ', len(dataset.keywords), ' keywords under parent:', parent_description)
    dataset.write_cluster_members(clus, cluster_keyword_file, parent_direcotry, cluster_keyword_embedding, cluster_keyword_label)
    center_names = dataset.write_cluster_centers(clus, parent_description, hierarchy_file)
    dataset.write_document_membership(clus, doc_membership_file, parent_direcotry)
    print('Done saving cluster results for ', len(dataset.keywords), ' keywords under parent:', parent_description)
    return center_names
