import numpy as np
from collections import defaultdict

# Returns the analogy data in a defaultdict (w1, w2): [(w3, w4), ...]
def get_analogy_data(filename, ignore_character = ':'):
    analogy_words = defaultdict(lambda: [])
    with open(filename) as f:
        for line in f:
            # Ignore comments
            if line.startswith(ignore_character):
                continue
            content = line.split()
            analogy_words[content[0], content[1]].append((content[2], content[3]))
    return analogy_words

# Sort the words based on their cosine similarity value, in descending order.
def sort_dict_on_value(data, print_temp_res):
    sorted_data = sorted(data.items(), key=lambda x: x[1], reverse = True)
    
    # Print the top 10 words with the highest cosine similarity.
    if print_temp_res:
        for key, value in sorted_data[:10]:
            print(str.format('\t\t{} ({}):', key, value))
    
    # Second one are cosines, but not useful here
    words, _ = zip(*sorted_data) 
    
    return words

# Returns a sorted list of words and their cosine similarity, in descending order.
def get_cosine_similarities(pred_vec, vecs_dataset, w_a, w_a2, w_b, print_temp_res):
    cosines = dict()
    for word, vec in vecs_dataset.items():
        
        # Skip if words are not in the model.
        if ((word == w_a) or (word == w_a2) or (word == w_b)):
            continue
        
        # Cosine similarity between the two vectors.
        cosines[word] = np.dot(pred_vec, vec) / (np.linalg.norm(vec) * np.linalg.norm(pred_vec))
    return sort_dict_on_value(cosines, print_temp_res)

 # Return the rank of the target word.
 # w_results dict[target_word, [w1,...]]
def get_all_ranks(w_results):
    indices = dict()
    # Foreach target word in the w_results
    for w_target, w_rankings in w_results. items():
        rank = get_single_rank(w_target, w_rankings)
        indices[w_target] = rank;
    return indices
    
# The rank of the target word is returned, note this is index + 1
def get_single_rank(w_target, w_rankings):
    for i_rank, w in enumerate(w_rankings):
        if w == w_target:
            return i_rank + 1
    print(str.format("WARNING: Could not find \'{}\' in list \n{}\n\n", w_target, w_rankings))

# Returns the MRR and the accuracy for the analogy task.
def get_results_analogy_task(analogy_words, vecs_dataset, print_temp_res):
    
    correct = 0
    RR_i = 0
    RR_total = 0
    j = 0 # Only for printing progress.
    
    # Get all the first two words of the analogies.
    for (w_a, w_a2), list_analogies in analogy_words.items():
        j += 1
        
        # Skip if words are not in the model.
        if not (w_a in vecs_dataset) or not (w_a2 in vecs_dataset):
            continue
        
        # Print progress.
        if print_temp_res:
            print(str.format('What {} -> {}, {}/{}:', w_a, w_a2, j, len(analogy_words.items())))

        w_a_vec = vecs_dataset[w_a] # a
        w_a2_vec = vecs_dataset[w_a2] # a*
        
        offset_vec = w_a2_vec - w_a_vec # a* - a
    
        # Get all the analogies given those two words
        i = 0
        for (w_b, w_b2) in list_analogies:
            i +=1
        
            # Skip if words are not in the model.
            if not (w_b in vecs_dataset) or not (w_b2 in vecs_dataset):
                continue
            
            if print_temp_res:
                print(str.format('\t{}/{}, {}/{} What {} -> ? ({}):', j, len(analogy_words.items()), i, len(list_analogies), w_b, w_b2))
            w_b_vec = vecs_dataset[w_b] # b

            # Predicted analogy word vector
            pred_vec = w_b_vec + offset_vec
            
            # Get for all words the cosine similarity of its vector with the actual target word vector.
            # Order the similarities.
            similarities = get_cosine_similarities(pred_vec, vecs_dataset, w_a, w_a2, w_b, print_temp_res)               
            
            rank = get_single_rank(w_b2, similarities)
            
            # For accuracy.
            if rank == 1:
                correct += 1
            
            # For Mean Recipropal Rank.
            RR = 1/rank
            RR_total += RR
            RR_i += 1
            
            # Print intermediate results.
            if print_temp_res:
                print(str.format("Avg MRR so far: {}", RR_total / RR_i))
                print(str.format("Accuracy so far: {}", correct / RR_i))

    MRR = RR_total / RR_i
    accuracy = correct/RR_i
    return MRR, accuracy
