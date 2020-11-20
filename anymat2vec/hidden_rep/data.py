"""
Parts of this code were adapted from https://github.com/Andras7/word2vec-pytorch/blob/master/word2vec/model.py

"""

import torch
import torch.nn as nn
from pymatgen import Composition, Element
import numpy as np
from torch.utils.data import Dataset
from mat2vec.processing.process import MaterialsTextProcessor
from collections import defaultdict
import os
from roost.roost.data import parse_roost
from roost.core import LoadFeaturiser
from collections import defaultdict

from anymat2vec.common_data.files import msch_emb_path

processor = MaterialsTextProcessor()


def _process_tokenizer(sentence):
    """
    Mat2vec tokenization and pre-processing, gets passed to torchtext tokenizer

    Args:
        sentence (str): sentence to be tokenized
    Returns:
        (processed_tokens, material_list)
    """
    processed_sentence = processor.process(
        sentence,
        exclude_punct=False,
        convert_num=True,
        normalize_materials=True,
        remove_accents=True,
        make_phrases=False,
        split_oxidation=True)

    return processed_sentence


def tokenize(sentence):
    """ Takes in a sentence and returns list of tokens.
    Args:
        sentence (str): sentence to be tokenized
    """
    return _process_tokenizer(sentence)

def get_stoichiometry_vector(formula, normalize=True):
    composition_dict = Composition(formula).get_el_amt_dict()
    vec = np.zeros(118)  # 118 elements on periodic table
    for el, amt in composition_dict.items():
        vec[Element(el).number - 1] = amt
    if normalize:
        vec = vec / np.sum(vec)
    return vec, composition_dict


def get_stoichiometry_sparse(formula):
    vec, composition_dict = get_stoichiometry_vector(formula, normalize=False)
    output_dict = {}
    composition_sum = float(np.sum(vec))
    for el, amt in composition_dict.items():
        vec[Element(el).number - 1] = amt
        output_dict[Element(el).number - 1] = amt / composition_sum
    return output_dict

np.random.seed(12345)


class DataReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, corpus_file, min_count, allow_discard_materials=True, n_elements=118, downsampling=0.0001):
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.device = "cpu"

        self.allow_discard_materials = allow_discard_materials
        self.negatives = []
        self.discards = []
        self.negpos = 0
        self.n_elements = n_elements
        self.downsampling = downsampling

        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()
        self.materials = set()
        self.num_regular_words = None

        self.elem_features = LoadFeaturiser(msch_emb_path)
        self.elem_emb_len = self.elem_features.embedding_size

        self.input_file = corpus_file
        self.read_words(min_count)
        self.init_table_negatives()
        self.init_table_discards()
        self.original_discards = self.discards.copy()
        self.original_negatives = self.negatives.copy()
        self.stoichiometries = None
        self.load_stoichiometries()


    def read_words(self, min_count):
        """
        Corpus is one 'sentence' per line (could also be an abstract maybe)

        Args:

            min_count (int): minimum number of occurences in corpus to be
                included in the vocabulary.
        """
        print("Building word frequency tables...\n")
        if isinstance(self.materials, list):
            self.materials = set(self.materials)

        word_frequency = defaultdict(int)
        material_frequency = defaultdict(int)

        for line in open(self.input_file, encoding="utf8"):
            tokens, formulas = tokenize(line)
            if len(tokens) > 1:
                self.sentences_count += 1
                self.materials.update([f[1] for f in formulas])
                for word in tokens:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency[word] + 1
                        if self.token_count % 1000000 == 0:
                            print("\rRead " + str(int(self.token_count / 1000000)) + "M words.")
        print("\n")

        #filter out materials with non-allowed elements
        filtered_materials = []
        for m in self.materials:
            elements, _ = parse_roost(m)
            if all([e in self.elem_features.allowed_types for e in elements]) and len(elements) > 1:
                filtered_materials.append(m)
            else:
                print(elements)

        self.materials = set(filtered_materials)
        # Build vocabulary, filter out materials
        wid = 0
        for w, c in word_frequency.items():
            if w in self.materials:
                material_frequency[w] = c
            elif c < min_count:
                continue
            else:
                self.word2id[w] = wid
                self.id2word[wid] = w
                self.word_frequency[wid] = c
                wid += 1
        self.num_regular_words = len(self.word2id)

        # Lock in ordering of materials in set
        self.materials = list(self.materials)
        # Add all materials to end of vocabulary
        for w, c in material_frequency.items():
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1

        print("Total words: " + str(self.num_regular_words))
        print("Total materials: " + str(len(material_frequency)))

    def init_table_discards(self):

        # Normalized frequency of each word
        f = np.array(list(self.word_frequency.values())) / self.token_count

        self.discards = np.sqrt(self.downsampling / f) + (self.downsampling / f)

        if not self.allow_discard_materials:
            # Never discard materials
            self.discards[self.num_regular_words::] = 0

    def init_table_negatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def get_negatives(self, target, size):  # TODO check equality with target
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response

    def load_stoichiometries(self):
        indices = []
        values = []
        for i, material in enumerate(self.materials):
            sparse_mat = get_stoichiometry_sparse(material)
            for key, value in sparse_mat.items():
                indices.append([i, key])
                values.append(value)
        self.indices = torch.LongTensor(indices)
        self.values = torch.FloatTensor(values)
        self.dimensions = torch.Size([len(self.materials), self.n_elements])
        self.stoichiometries = torch.sparse.FloatTensor(self.indices.t(), self.values, self.dimensions)

    # def load_stoichiometries(self):
    #     #default crystal is H2O
    #     default_value = (*self.get_stoichiometry_vector('H2O'), 0)
    #     self.stoichiometries = defaultdict(lambda:default_value)
    #     for i, material in enumerate(self.materials):
    #         atom_weights, atom_fea, self_fea_idx, nbr_fea_idx = self.get_stoichiometry_vector(material)
    #         inputs_dict = (atom_weights, atom_fea, self_fea_idx, nbr_fea_idx, i + 1)

    #         self.stoichiometries[i] = inputs_dict

    def discard_materials(self, discard_list):
        """
        Removes materials from discard_list from the training set. Use for cross-validation.

        Args:
            discard list (list): formulas (mat2vec preprocessed)
                or formula indices as they are in self.materials.
        """
        self.original_discards = self.discards.copy()
        self.original_negatives = self.negatives.copy()
        if isinstance(discard_list[0], int):
            for idx in discard_list:
                self.discards[idx + self.num_regular_words] = 1
                self.negatives[idx + self.num_regular_words] = 0
        elif isinstance(discard_list[0], str):
            for mat in discard_list:
                self.discards[self.word2id[mat] + self.num_regular_words] = 1
                self.negatives[self.word2id[mat] + self.num_regular_words] = 0

    def restore_discarded(self):
        """ Restores the dataset's original discard and sampling frequencies """
        self.discards = self.original_discards
        self.negatives = self.original_negatives

    def get_stoichiometry_vector(self, formula, normalize=True):

        elements, weights = parse_roost(formula)
        weights = np.atleast_2d(weights).T / np.sum(weights)
        try:
            atom_fea = np.vstack(
                [self.elem_features.get_fea(element) for element in elements]
            )
        except AssertionError:
            raise AssertionError(
                f"[{formula}] contains element types not in embedding"
            )
        except ValueError:
            raise ValueError(
                f"[{formula}] composition cannot be parsed into elements"
            )

        env_idx = list(range(len(elements)))
        self_fea_idx = []
        nbr_fea_idx = []
        nbrs = len(elements) - 1
        for i, _ in enumerate(elements):
            self_fea_idx += [i] * nbrs
            nbr_fea_idx += env_idx[:i] + env_idx[i + 1 :]

        # convert all data to tensors
        atom_weights = torch.Tensor(weights).to(self.device)
        atom_fea = torch.Tensor(atom_fea).to(self.device)
        self_fea_idx = torch.LongTensor(self_fea_idx).to(self.device)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx).to(self.device)
        
        return atom_weights, atom_fea, self_fea_idx, nbr_fea_idx

    def save(self, filepath):
        torch.save(self, filepath)

    @staticmethod
    def from_save(filepath):
        data = torch.load(filepath)
        data.input_file = filepath.replace(".pt", ".txt")
        return data

class HiddenRepDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.input_file, encoding="utf8")

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()

                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    boundary = np.random.randint(1, self.window_size)
                    return [(u, v, self.data.get_negatives(v, 5)) for i, u in enumerate(word_ids) for j, v in
                            enumerate(word_ids[max(i - boundary, 0):i + boundary]) if u != v]

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _, _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)
