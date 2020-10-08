import os

from gensim.models import KeyedVectors

from anymat2vec.tasks.analogy_testing.data import load_excluded_materials_list, get_analogies_filename


def run_analogy_test(words, vectors, analogy_set_fname, print_results=True):
    """
    Run a test on a set of analogies, and print results to console.

    Args:
        words ([str]): The set of words, ordered according to index. The words
            should be normalized.
        vectors ([array]): The set of vectors, ordered according to index. The
            order of vectors should match the order of words, so
                                word[0] -> vector[0]
        analogy_set: Local filename of the analogy set to load within folder.
            For example, "analogies.txt"
        print_results (bool): Print accuracy of analogies by category.

    Returns:
        [dict]: A list of dicts containing the correct and incorrect answers
            by analogy section.
    """
    analogy_set_fname = get_analogies_filename(analogy_set_fname)
    kv = KeyedVectors(vectors.shape[1])
    kv.add(words, vectors)
    analogies_score, sections = kv.evaluate_word_analogies(analogy_set_fname)
    for section in sections:
        n_correct = len(section['correct'])
        n_incorrect = len(section['incorrect'])
        n_total = n_correct + n_incorrect
        if print_results:
            print(f"{section['section']}: {n_correct / n_total}")
    return sections


if __name__ == "__main__":

    # Test the full, ground truth model
    # replace this with true (test words in corpus) hidden rep object
    # should accept words and return embeddings
    model_truth = "model_trained_with_excluded_materials"
    words = model_truth.words
    embeddings = model_truth.embeddings
    print("Ground truth model analogy performance:")
    run_analogy_test(words, embeddings, "analogies.txt")


    # Test the restricted model
    # replace this with the expected (test words not in corpus) hidden rep object
    model_expected = "model_trained_without_excluded_materials"
    words_training = model_expected.words
    embeddings_training = model_expected.embeddings

    excluded_materials = load_excluded_materials_list()
    contaminated_materials = []
    for em in excluded_materials:
        if em in words_training:
            contaminated_materials.append(em)
    if contaminated_materials:
        raise ValueError(f"The following words/mats were supposed to be excluded but were not: {contaminated_materials}.")


    embeddings_test = model_expected.get_embeddings(excluded_materials)
    all_words = words_training + excluded_materials
    all_embeddings = embeddings_training + embeddings_test
    print("Expected model analogy test performance:")
    run_analogy_test(all_words, all_embeddings, get_analogies_filename("analogies_compounds_only.txt"))

