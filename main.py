# Unsupervised Language Learning: Assignment 1
# Richard Olij (10833730) & Melissa Tjhia (10761071)

import word_similarity_task as wst
import word_analogy_task as wat
import clustering_word_vector_task as cwvt
import functions as fn

# ========== 2 Getting the pre-trained word vectors ==========

print("Getting the pre-trained word vectors")
bow2_word_vecs = fn.get_word_vectors('bow2.words')
bow5_word_vecs = fn.get_word_vectors('bow5.words')
deps_word_vecs = fn.get_word_vectors('deps.words')

# ========== 3 Word Similarity Task ==========

print("=== Word similarity task ===")
simlex_pairs = wst.get_word_pairs_human_judgement('SimLex-999/SimLex-999.txt', 'simlex')
men_pairs = wst.get_word_pairs_human_judgement('MEN/MEN_dataset_natural_form_full', 'men')

print("\t Computing cosine similarity")
bow2_simlex_cos = wst.get_word_pairs_cosines(simlex_pairs, bow2_word_vecs)
bow5_simlex_cos = wst.get_word_pairs_cosines(simlex_pairs, bow5_word_vecs)
deps_simlex_cos = wst.get_word_pairs_cosines(simlex_pairs, deps_word_vecs)

bow2_men_cos = wst.get_word_pairs_cosines(men_pairs, bow2_word_vecs)
bow5_men_cos = wst.get_word_pairs_cosines(men_pairs, bow5_word_vecs)
deps_men_cos = wst.get_word_pairs_cosines(men_pairs, deps_word_vecs)

######## SIMLEX
print("\t Comparing with SimLex")

# BOW2
v1_bow2_simlex, v2_bow2_simlex = wst.get_similarity_vecs(simlex_pairs, bow2_simlex_cos)
wst.print_correlations(v1_bow2_simlex, v2_bow2_simlex, "BOW2", "SimLex")

# BOW5
v1_bow5_simlex, v2_bow5_simlex = wst.get_similarity_vecs(simlex_pairs, bow5_simlex_cos)
wst.print_correlations(v1_bow5_simlex, v2_bow5_simlex, "BOW5", "SimLex")

# Dependencies
v1_deps_simlex, v2_deps_simlex = wst.get_similarity_vecs(simlex_pairs, deps_simlex_cos)
wst.print_correlations(v1_deps_simlex, v2_deps_simlex, "Dependency based", "SimLex")

######## MEN
print("\t Comparing with MEN")

# BOW2
v1_bow2_men, v2_bow2_men = wst.get_similarity_vecs(men_pairs, bow2_men_cos)
wst.print_correlations(v1_bow2_men, v2_bow2_men, "BOW2", "MEN")

# BOW5
v1_bow5_men, v2_bow5_men = wst.get_similarity_vecs(men_pairs, bow5_men_cos)
wst.print_correlations(v1_bow5_men, v2_bow5_men, "BOW5", "MEN")

# Dependencies
v1_deps_men, v2_deps_men = wst.get_similarity_vecs(men_pairs, deps_men_cos)
wst.print_correlations(v1_deps_men, v2_deps_men, "Dependency based", "MEN")

# ========== 4 Word Analogy Task ==========

print("\n=== Word analogy task ===")
analogy_words = wat.get_analogy_data('Google analogy test set (State of the art)_questions-words.txt')

print("\t BOW 2")
MRR_bow2, accuracy_bow2 = wat.get_results_analogy_task(analogy_words, bow2_word_vecs, True)

print("\t BOW 5")
MRR_bow5, accuracy_bow5 = wat.get_results_analogy_task(analogy_words, bow5_word_vecs, True)

print("\t Dependency")
MRR_deps, accuracy_deps = wat.get_results_analogy_task(analogy_words, deps_word_vecs, True)

# ========== 5 Clustering word vectors ==========

print("\n=== Clustering word vectors ===")
n_clusters = 15
w_nouns = cwvt.get_nouns('2000_nouns_sorted.txt')

# BOW2
print("\t BOW 2")
BOW2_pca_vectors_per_class, BOW2_words_per_class, BOW2_X = cwvt.get_clustering_data(bow2_word_vecs, w_nouns, n_clusters)
cwvt.show_results(BOW2_pca_vectors_per_class, BOW2_words_per_class, BOW2_X, 10)

# BOW5
print("\t BOW 5")
BOW5_pca_vectors_per_class, BOW5_words_per_class, BOW5_X = cwvt.get_clustering_data(bow5_word_vecs, w_nouns, n_clusters)
cwvt.show_results(BOW5_pca_vectors_per_class, BOW5_words_per_class, BOW5_X, 10)

# Dependency
print("\t Dependency")
DEPS_pca_vectors_per_class, DEPS_words_per_class, DEPS_X = cwvt.get_clustering_data(deps_word_vecs, w_nouns, n_clusters)
cwvt.show_results(DEPS_pca_vectors_per_class, DEPS_words_per_class, DEPS_X, 10)