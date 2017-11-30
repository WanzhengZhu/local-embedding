'''
__author__: Wanzheng Zhu
__description__: A wrapper for spherecluster, implement the term clustering component.
__latest_updates__: 09/25/2017
'''
from collections import defaultdict
from scipy.spatial.distance import cosine
from spherecluster import SphericalKMeans
from dataset import SubDataSet
from spherical_meanshift import SphericalMeanShift, estimate_bandwidth
import numpy as np
from scipy import stats
from sklearn.datasets.samples_generator import make_blobs
import subprocess


class Meanshifter:

    def __init__(self, data, parent_direcotry):
        self.data = data
        self.n_cluster = None
        filename = 'keyword_embeddings.txt'
        np.savetxt(parent_direcotry + filename, data, header=str(data.shape[0]) + " " + str(data.shape[1]), comments='')
        print('[MeanShift] MeanShift Started...')
        subprocess.call(["../data/synthetic/fams/cmake-build-debug/fams", "30", "46", "100", filename[:-4], parent_direcotry, "-f", "0.05", "10", "1"])
        print('[MeanShift] MeanShift finished...')
        # subprocess.call(["../data/synthetic/fams/cmake-build-debug/fams", "20", "46", "1", filename[:-4], parent_direcotry, "-f", "0.05", "10", "2"])

        self.clus = SphericalKMeans()
        self.clus.cluster_centers_ = np.loadtxt(parent_direcotry + 'modes_' + filename)
        # if self.clus.cluster_centers_.size == 21:  # Just one cluster center
        #     self.clus.cluster_centers_ = np.delete(self.clus.cluster_centers_, 0)
        # else:
        self.clus.cluster_centers_ = np.delete(self.clus.cluster_centers_, np.s_[0:1], axis=1)
        labels = np.loadtxt(parent_direcotry + 'out_' + filename)
        self.clus.labels_ = None
        for i in range(len(labels)):
            new_label = stats.mode(np.where(self.clus.cluster_centers_ == labels[i])[0])[0]
            if new_label.size == 0:
                cos_score = -1
                for j in range(len(self.clus.cluster_centers_)):
                    new_cos_score = self.calc_cosine(self.clus.cluster_centers_[j], self.data[i])
                    # new_cos_score = np.dot(self.clus.cluster_centers_[j], self.data[i])
                    if new_cos_score >= cos_score:
                        new_label = j
                        cos_score = new_cos_score
            self.clus.labels_ = np.append(self.clus.labels_, new_label)
        self.clus.labels_ = np.delete(self.clus.labels_, 0)  # Delete the first None

        # bandwidth = estimate_bandwidth(data, quantile=0.188)
        # self.clus = SphericalMeanShift(bandwidth=1.0, n_jobs=1, cluster_all=True, bin_seeding=True)
        # self.clus.fit(self.data)

        # for quantile in [0.015, 0.1, 0.2, 0.3]:
        #     bandwidth = estimate_bandwidth(data, quantile=quantile)
        #     print('Bandwidth for quantile=' , quantile , ': ' , bandwidth)
        #
        # for bandwidth in [0.8, 0.9, 1]:
        #     self.clus = SphericalMeanShift(bandwidth=bandwidth, n_jobs=1, cluster_all=False, bin_seeding=True)
        #     self.clus.fit(self.data)
        #     print('number of estimated clusters for bandwidth = ', bandwidth, ': ', len(np.unique(self.clus.labels_)))

        self.clusters = defaultdict(list)  # cluster id -> members
        self.n_cluster = len(np.unique(self.clus.labels_))
        print('number of estimated clusters : ', self.n_cluster)
        self.membership = None  # a list contain the membership of the data points
        self.center_ids = None  # a list contain the ids of the cluster centers
        # self.inertia_scores = None

    def fit(self):
        labels = self.clus.labels_
        print(labels)
        for i in range(self.n_cluster):
            print('Cluster ', i, ': ', sum(labels==i))
        for idx, label in enumerate(labels):
            self.clusters[label].append(idx)
        self.membership = labels
        self.center_ids = self.gen_center_idx()
        # self.inertia_scores = self.clus.inertia_
        # print('Clustering concentration score:', self.inertia_scores)

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

    def calc_cosine(self, vec_a, vec_b):
        return 1 - cosine(vec_a, vec_b)


def test_example(choice):
    # #############################################################################
    # Generate sample data
    if choice == 0:  # Spherical samples
        centers = [[1, 0, -0.3], [-0.5, 1, -0.3], [0, 0, 1]]
        X, _ = make_blobs(n_samples=1000, centers=centers, cluster_std=0.3)
        X = [x/np.linalg.norm(x) for x in X]
    elif choice == 1:  # Euclidean 2-D samples
        centers = [[1, 1], [-1, -1], [1, -1]]
        X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

    # #############################################################################
    # Compute clustering with MeanShift

    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

    ms = SphericalMeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X, plot=True)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    # # #############################################################################
    # # Plot result
    # import matplotlib.pyplot as plt
    # from itertools import cycle
    #
    # plt.figure(1)
    # plt.clf()
    #
    # colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    # for k, col in zip(range(n_clusters_), colors):
    #     my_members = labels == k
    #     cluster_center = cluster_centers[k]
    #     plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    #     plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
    #              markeredgecolor='k', markersize=14)
    # plt.title('Estimated number of clusters: %d' % n_clusters_)
    # plt.show()


def run_meanshift(full_data, doc_id_file, filter_keyword_file, parent_direcotry, parent_description,\
                   cluster_keyword_file, hierarchy_file, doc_membership_file, cluster_keyword_embedding, \
                   cluster_keyword_label, filter_keyword, iter, update_center, input_dir):
    # test_example(0)
    # array = np.array([val for (key, val) in full_data.embeddings.iteritems()])
    dataset = SubDataSet(full_data, doc_id_file, filter_keyword_file, filter_keyword, iter)
    print('Start Mean Shifting for ', len(dataset.keywords), ' keywords under parent:', parent_description)
    ## TODO: change later here for n_cluster selection from a range
    clus = Meanshifter(dataset.embeddings, parent_direcotry)
    clus.fit()
    print('Done clustering for ', len(dataset.keywords), ' keywords under parent:', parent_description)
    dataset.write_cluster_members(clus, cluster_keyword_file, parent_direcotry)
    center_names = dataset.write_cluster_centers(clus, parent_description, hierarchy_file)
    dataset.write_document_membership(clus, doc_membership_file, parent_direcotry)
    print('Done saving cluster results for ', len(dataset.keywords), ' keywords under parent:', parent_description)
    return center_names
