'''
__author__: Chao Zhang
__description__: Main function for TaxonGen
__latest_updates__: 09/26/2017
'''
import time
import argparse
from dataset import DataSet
from cluster import run_clustering
from paras import *
from caseslim import main_caseolap
from case_ranker import main_rank_phrase
from local_embedding_training import main_local_embedding
from shutil import copyfile
from distutils.dir_util import copy_tree
from os import symlink
from shutil import rmtree
from meanshift import run_meanshift
from dataset import SubDataSet


MAX_LEVEL = 1

class DataFiles:
    def __init__(self, input_dir, node_dir):
        self.doc_file = input_dir + 'papers.txt'
        self.link_file = input_dir + 'keyword_cnt.txt'
        self.index_file = input_dir + 'index.txt'

        self.embedding_file = node_dir + 'embeddings.txt'
        self.seed_keyword_file = node_dir + 'seed_keywords.txt'
        self.doc_id_file = node_dir + 'doc_ids.txt'

        self.doc_membership_file = node_dir + 'paper_cluster.txt'
        self.hierarchy_file = node_dir + 'hierarchy.txt'
        self.cluster_keyword_file = node_dir + 'cluster_keywords.txt'
        self.cluster_keyword_embedding = node_dir + 'cluster_keywords_embedding.txt'
        self.cluster_keyword_label = node_dir + 'cluster_keywords_label.txt'

        self.caseolap_keyword_file = node_dir + 'caseolap.txt'
        self.filtered_keyword_file = node_dir + 'keywords.txt'


'''
input_dir: the directory for storing the input files that do not change
node_dir: the directory for the current node in the hierarchy
n_cluster: the number of clusters
filter_thre: the threshold for filtering general keywords in the caseolap phase
parent: the name of the parent node
n_expand: the number of phrases to expand from the center
level: the current level in the recursion
'''

def recur(input_dir, node_dir, n_cluster, parent, n_cluster_iter, filter_thre, n_expand, level, caseolap, local_embedding, filter_keyword, update_center):
    if level > MAX_LEVEL:
        return
    print('============================= Running level ', level, ' and node ', parent, '=============================')
    start = time.time()
    df = DataFiles(input_dir, node_dir)
    ## TODO: Everytime we need to read-in the whole corpus, which can be slow.
    full_data = DataSet(df.embedding_file, df.doc_file)
    end = time.time()
    print('[Main] Done reading the full data using time %s seconds' % (end-start))

    # filter the keywords
    if caseolap is False:
        # children = run_clustering(full_data, df.doc_id_file, df.seed_keyword_file, n_cluster, node_dir, parent, df.cluster_keyword_file, df.hierarchy_file, df.doc_membership_file, df.cluster_keyword_embedding, df.cluster_keyword_label, filter_keyword, 0, update_center, input_dir)
        # children = run_meanshift(full_data, df.doc_id_file, df.seed_keyword_file, node_dir, parent, df.cluster_keyword_file, df.hierarchy_file, df.doc_membership_file, df.cluster_keyword_embedding, df.cluster_keyword_label, filter_keyword, 0, update_center, input_dir)

        try:
            dataset = SubDataSet(full_data, df.doc_id_file, df.seed_keyword_file, filter_keyword, 0)
            children, previous_num_of_keywords = run_clustering(dataset, df.seed_keyword_file, n_cluster, node_dir, parent, df.cluster_keyword_file, df.hierarchy_file, df.doc_membership_file, df.cluster_keyword_embedding, df.cluster_keyword_label, update_center, input_dir)

            # children = run_meanshift(full_data, df.doc_id_file, df.seed_keyword_file, node_dir, parent, df.cluster_keyword_file, df.hierarchy_file, df.doc_membership_file)
        except:
            print('Clustering not finished.')
            return
        copyfile(df.seed_keyword_file, df.filtered_keyword_file)
    else:
        ## Adaptive Clustering, maximal n_cluster_iter iterations
        previous_num_of_keywords = 0
        for iter in range(n_cluster_iter):
            if iter > 0:
                df.seed_keyword_file = df.filtered_keyword_file

            # children = run_clustering(full_data, df.doc_id_file, df.seed_keyword_file, n_cluster, node_dir, parent,df.cluster_keyword_file, df.hierarchy_file, df.doc_membership_file, df.cluster_keyword_embedding, df.cluster_keyword_label, filter_keyword, iter, update_center)
            try:
                dataset = SubDataSet(full_data, df.doc_id_file, df.seed_keyword_file, filter_keyword, iter)
                if len(dataset.keywords) == previous_num_of_keywords:  # Convergence achieved!
                    print("Convergence achieved at %s" % str(iter) + '/' + str(n_cluster_iter-1))
                    print("Number of keywords for clustering is: %s" % previous_num_of_keywords)
                    break
                children, previous_num_of_keywords = run_clustering(dataset, df.seed_keyword_file, n_cluster, node_dir, parent,df.cluster_keyword_file, df.hierarchy_file, df.doc_membership_file, df.cluster_keyword_embedding, df.cluster_keyword_label, update_center, input_dir)
                # children = run_meanshift(full_data, df.doc_id_file, df.seed_keyword_file, node_dir, parent, df.cluster_keyword_file, df.hierarchy_file, df.doc_membership_file)
            except:
                print('Clustering not finished.')
                return

            if iter != n_cluster_iter-1:  # Needless to run for the last loop
                print("[Main] Start to run CaseOALP: %s" % str(iter) + '/' + str(n_cluster_iter-1))
                start = time.time()
                main_caseolap(df.link_file, df.doc_membership_file, df.cluster_keyword_file, df.caseolap_keyword_file)
                main_rank_phrase(df.caseolap_keyword_file, df.filtered_keyword_file, filter_thre)
                end = time.time()
                print("[Main] Finish running CaseOALP using %s (seconds)" % (end - start))

    # find_general_terms(node_dir, n_cluster)

    # prepare the embedding for child level
    if level < MAX_LEVEL:
        if local_embedding is False:
            src_file = node_dir + 'embeddings.txt'
            for child in children:
                tgt_file = node_dir + child + '/embeddings.txt'
                # copyfile(src_file, tgt_file)
                symlink(src_file, tgt_file)
        else:
            start = time.time()
            main_local_embedding(node_dir, df.doc_file, df.index_file, parent, n_expand)
            end = time.time()
            print("[Main] Finish running local embedding training using %s (seconds)" % (end - start))

    for child in children:
        recur(input_dir, node_dir + child + '/', n_cluster, child, n_cluster_iter, filter_thre, n_expand, level + 1, caseolap, local_embedding, filter_keyword, update_center)

def main(opt, foldername):
    input_dir = opt['input_dir']
    init_dir = opt['data_dir'] + 'init/'
    n_cluster = opt['n_cluster']
    filter_thre = opt['filter_thre']
    n_expand = opt['n_expand']
    n_cluster_iter = opt['n_cluster_iter']
    level = 0

    # Non-para
    root_dir = opt['data_dir'] + foldername + '/'
    if os.path.exists(root_dir):
        rmtree(root_dir)
    copy_tree(init_dir, root_dir)
    recur(input_dir, root_dir, n_cluster, '*', n_cluster_iter, filter_thre, n_expand, level, caseolap=True, local_embedding=False, filter_keyword=False, update_center=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='main.py', description='')
    parser.add_argument('-dataset', required=False, default='toy', help='toy or dblp or sp')
    parser.add_argument('-foldername', required=False, default='non-para', help='non-para')
    args = parser.parse_args()
    print("Loading " + args.dataset + " dataset...")
    if args.dataset == 'toy':
        opt = load_toy_params()
    elif args.dataset == 'dblp':
        opt = load_dblp_params()
    elif args.dataset == 'sp':
        opt = load_sp_params()
    main(opt, args.foldername)
