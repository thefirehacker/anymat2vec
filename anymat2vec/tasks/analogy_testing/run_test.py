import os

from gensim.models import KeyedVectors


def run_analogy_test(words, vectors, analogy_set="full", print_results=True):
    """
    Run a test on a set of analogies, and print results to console.

    Args:
        words ([str]): The set of words, ordered according to index. The words
            should be normalized.
        vectors ([array]): The set of vectors, ordered according to index. The
            order of vectors should match the order of words, so
                                word[0] -> vector[0]
        analogy_set: Use "full" for testing on the full set of analogies. Use
            "restricted" for testing on the anymat2vec analogies alone.
        print_results (bool): Print accuracy of anaologies by category.

    Returns:
        [dict]: A list of dicts containing the correct and incorrect answers
            by analogy section.
    """
    analogy_files = {
        "full": "analogies.txt",
        "compounds_only": "analogies_compounds_only.txt"
    }
    if analogy_set not in analogy_files:
        raise KeyError(f"Analogy set {analogy_set} not found; choose from"
                       f" {list(analogy_files.keys())}")
    this_dir = os.path.dirname(os.path.abspath(__file__))
    analogy_path = os.path.join(this_dir, analogy_files[analogy_set])

    kv = KeyedVectors(vectors.shape[1])
    kv.add(words, vectors)
    analogies_score, sections = kv.evaluate_word_analogies(analogy_path)
    for section in sections:
        n_correct = len(section['correct'])
        n_incorrect = len(section['incorrect'])
        n_total = n_correct + n_incorrect
        if print_results:
            print(f"{section['section']}: {n_correct / n_total}")
    return sections
