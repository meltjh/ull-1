import numpy as np
from scipy.stats import spearmanr, pearsonr

# Read the human judgement scores per pair.
def get_word_pairs_human_judgement(filename, dataset):
    word_pairs_human_judgement = dict()
    with open(filename) as f:
        if dataset == 'simlex':
            next(f)
            
        for line in f:
            content = line.split()
            word_pair = (content[0], content[1])
            
            if dataset == 'simlex':
                word_pairs_human_judgement[word_pair] = content[3]
            else:
                word_pairs_human_judgement[word_pair] = content[2]
    return word_pairs_human_judgement

# Compute the cosine similarity between two word vectors.
def get_word_pairs_cosines(word_pairs_human_judgement, word_vecs):
    word_pairs_cosine = dict()
    for (word_1, word_2) in word_pairs_human_judgement.keys():
        # Skip if words are not in the model.
        if not (word_1 in word_vecs) or not (word_2 in word_vecs):
            print(str.format('Warning, one of the words \'{}\' \'{}\' did not occur.', word_1, word_2))
            continue
        
        word_1_vec = word_vecs[word_1]
        word_2_vec = word_vecs[word_2]
        
        # Cosine similarity between the two vectors.
        word_pairs_cosine[(word_1, word_2)] = np.dot(word_1_vec, word_2_vec) / (np.linalg.norm(word_1_vec) * np.linalg.norm(word_2_vec))

    return word_pairs_cosine

# Convert the dicts with scores to vectors.
def get_similarity_vecs(word_pairs_human_judgement, word_pairs_cosine):
    num_pairs = len(word_pairs_cosine.keys())
    hj_vec = np.zeros([num_pairs])
    cosine_vec = np.zeros([num_pairs])
    
    for i, (pair, cosine_value) in enumerate(word_pairs_cosine.items()):
        hj_value = word_pairs_human_judgement[pair]
        hj_vec[i] = hj_value
        cosine_vec[i] = cosine_value
    return hj_vec, cosine_vec

# Get the Spearman & Pearson correlations based on the scores.
def get_correlations(vec1, vec2):
    spearman_corr = spearmanr(vec1, vec2)[0]
    pearson_corr = pearsonr(vec1, vec2)[0]
    return [spearman_corr, pearson_corr]

# Print the correlations.
def print_correlations(vec1, vec2, word_embedding_model_name, dataset_name):
    [spearman_corr, pearson_corr] = get_correlations(vec1, vec2)
    print("For {} and {}, Spearman correlation: {}, Pearson correlation: {}"
          .format(word_embedding_model_name, dataset_name, spearman_corr, pearson_corr))