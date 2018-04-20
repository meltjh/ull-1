import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import random
import matplotlib.pyplot as plt

# Getting the nouns
def get_nouns(filename):
    w_nouns = []
    with open(filename) as f:
        for line in f:
            w_nouns.append(line.strip())
    return w_nouns

# Returns the matrix of vecs of the nouns
def get_vecs_from_nouns(vecs_dataset, w_nouns):
    w_vecs = []
    for w in w_nouns:
        w_vec = vecs_dataset[w]
        w_vecs.append(w_vec)
    return np.array(w_vecs)

# Returns the list nouns with only occuring words
def get_only_occuring_nouns(vecs_dataset, w_nouns):
    nouns_new = []
    for w in w_nouns:
        if w in vecs_dataset:
            nouns_new.append(w)
    return nouns_new

# Returns lists sorted by the cluster_id's
def list_per_cluster(w_nouns_new, pred, pca_result, n_clusters):
    
    words_per_class = [[] for x in range(n_clusters)]
    pca_vectors_per_class = [[] for x in range(n_clusters)]
    for i in range(len(w_nouns_new)):
        w = w_nouns_new[i]
        c_i = pred[i]
        words_per_class[c_i].append(w)
        pca_vectors_per_class[c_i].append(pca_result[i,:])
        
    return words_per_class, pca_vectors_per_class

# Show some random example words of each class
def print_sample_words_per_class(words_per_class, top_n = 10):
    for c_i in range(len(words_per_class)):
        print("Class {}:".format(c_i))
        for w in random.sample(words_per_class[c_i], top_n):
            print("\t\'{}\'".format(w))
            
# Scatter the nouns in the 2D PCA space.
def scatter_clusters(pca_vectors_per_class, w_vecs):
    for c_i in range(len(pca_vectors_per_class)):
        w_vecs = pca_vectors_per_class[c_i]
        x, y = zip(*w_vecs)
        plt.scatter(x, y, label = c_i, alpha = 0.2)
    plt.legend()
    plt.show()
    
    
# Reeturns the data for plotting the clusters and words per class.
def get_clustering_data(deps_word_vecs, w_nouns, n_clusters):
    
    # Replace the w_nouns with only the occuring nouns
    w_nouns = get_only_occuring_nouns(deps_word_vecs, w_nouns)

    # Represent in a matrix form
    X = get_vecs_from_nouns(deps_word_vecs, w_nouns) 
    
    # Obtain the clusters and corresponding cluster_id's for the X prediction
    pred = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(X)
    
    # Transform the X into a 2D space using PCA
    pca_result = PCA(n_components=2).fit_transform(X)
    
    words_per_class, pca_vectors_per_class = list_per_cluster(w_nouns, pred, pca_result, n_clusters)

    return pca_vectors_per_class, words_per_class, X

# Visualize the clusters and print sample words per class.
def show_results(pca_vectors_per_class, words_per_class, w_vecs, top_n = 10):
    print_sample_words_per_class(words_per_class, top_n)
    scatter_clusters(pca_vectors_per_class, w_vecs)