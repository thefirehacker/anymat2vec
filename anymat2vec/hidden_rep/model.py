"""
Parts of this code were adapted from https://github.com/Andras7/word2vec-pytorch/blob/master/word2vec/model.py

"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from roost.roost.model import DescriptorNetwork

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
device = "cpu"

def collate_batch(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, self_fea_idx, nbr_fea_idx)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      self_fea_idx: torch.LongTensor shape (n_i, M)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)


    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_weights: torch.Tensor shape (N, 1)
    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
        Atom features from atom type
    batch_self_fea_idx: torch.LongTensor shape (N, M)
        Indices of mapping atom to copies of itself
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
        Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
        Mapping from the crystal idx to atom idx
    """
    # define the lists
    batch_atom_weights = []
    batch_atom_fea = []
    batch_self_fea_idx = []
    batch_nbr_fea_idx = []
    crystal_atom_idx = []
    batch_cry_ids = []

    cry_base_idx = 0
    for i, inputs in enumerate(dataset_list):
        atom_weights, atom_fea, self_fea_idx, nbr_fea_idx, cry_id = inputs
        # number of atoms for this crystal
        n_i = atom_fea.shape[0]

        # batch the features together
        batch_atom_weights.append(atom_weights)
        batch_atom_fea.append(atom_fea)

        # mappings from bonds to atoms
        batch_self_fea_idx.append(self_fea_idx + cry_base_idx)
        batch_nbr_fea_idx.append(nbr_fea_idx + cry_base_idx)

        # mapping from atoms to crystals
        crystal_atom_idx.append(torch.tensor([i] * n_i))

        # batch the targets and ids
        batch_cry_ids.append(cry_id)

        # increment the id counter
        cry_base_idx += n_i

    return (
            torch.cat(batch_atom_weights, dim=0).to(device),
            torch.cat(batch_atom_fea, dim=0).to(device),
            torch.cat(batch_self_fea_idx, dim=0).to(device),
            torch.cat(batch_nbr_fea_idx, dim=0).to(device),
            torch.cat(crystal_atom_idx).to(device),
        )



class HiddenRepModel(nn.Module):
    """
    u_embedding: Embedding for center word.
    v_embedding: Embedding for context words.
    """

    def __init__(self, emb_size, emb_dimension, hidden_size, stoichiometries, num_regular_words):
        """
        Note: The vocabulary used by this model needs to be a bit special.
        The total vocab size, V = W + M, where W is the number of regular words, and
        M is the number of materials formulae. `emb_size` must be equal to W,
        not V. The reason for this is how we check if a sample is a material or not.
        If the index of the target (pos_u) is greater than the vocab size,
        we look it up in the stoichiometries matrix. Likewise for context and
        negative samples. If you want to train a normal word2vec (no hidden rep training),
        don't remove formulas and V will be equal to W (M = 0) and pos_u/pos_v/neg_v
        will always be <= emb_size.

        """
        super(HiddenRepModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.hidden_size = hidden_size
        self.stoichiometries = stoichiometries
        self.stoich_size = len(stoichiometries)
        self.stoichiometries = stoichiometries
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.num_regular_words = num_regular_words
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

        self.shared_generator = DescriptorNetwork(200,
        elem_fea_len=self.hidden_size,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256])


        # target material embedding generator head
        self.tmeg = torch.nn.Linear(self.hidden_size, self.emb_dimension)
        # context material embedding generator head
        self.cmeg = torch.nn.Linear(self.hidden_size, self.emb_dimension)

    def _generate_embedding(self, uv, context=False):
        stoich = [self.stoichiometries[u] for u in uv]
        roost_input = collate_batch(stoich)
        hrelu = self.shared_generator(*roost_input)

        if context:
            return self.cmeg(hrelu)
        else:
            return self.tmeg(hrelu)

    def _masked_score(self, emb_u, emb_v, emb_neg_v, u_mask, v_mask, neg_v_mask):

        pos_score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        pos_score = torch.clamp(pos_score, max=10, min=-10)
        pos_score = -F.logsigmoid(pos_score)
        # Mask out scores contributed by unwanted positive context words
        pos_score[v_mask] = 0
        pos_score = pos_score 

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = F.logsigmoid(-neg_score)
        # Mask out scores contributed by unwanted negative samples
        neg_score[neg_v_mask] = 0
        neg_score = -torch.sum(neg_score, dim=1)

        total_score = pos_score + neg_score
        # Mask out scores contributed by unwanted target words
        total_score[u_mask] = 0

        # Account for scores being counted twice
        return total_score / 2

    def forward(self, pos_u, pos_v, neg_v):
        """
        There are 4 cases:
            word in center, word in context
            word in center, material in context
            material in center, word in context
            material in center, material in context

        If the index of the target (pos_u) is greater than the vocab size,
        we look it up in the stoichiometries matrix. Likewise for context and
        negative samples.

        Args:
            pos_u: Index of target word.
            pos_v: Index of context word.
            neg_v: Indices of negative sampled words.
        """
        emb_u_w = self.u_embeddings(pos_u)
        emb_v_w = self.v_embeddings(pos_v)
        emb_neg_v_w = self.v_embeddings(neg_v)

        emb_u_m = self._generate_embedding(pos_u)
        emb_v_m = self._generate_embedding(pos_v, context=True)
        emb_neg_v_m = torch.stack([self._generate_embedding(n, context=True) for n in neg_v])

        emb_u_mask_pairs = [(emb_u_w, pos_u.ge(self.num_regular_words), "w"),
                            (emb_u_m, pos_u.lt(self.num_regular_words), "m")]
        emb_v_mask_pairs = [(emb_v_w, pos_v.ge(self.num_regular_words), "w"),
                            (emb_v_m, pos_v.lt(self.num_regular_words), "m")]
        emb_neg_v_mask_pairs = [(emb_neg_v_w, neg_v.ge(self.num_regular_words), "w"),
                                (emb_neg_v_m, neg_v.lt(self.num_regular_words), "m")]

        subscores = {}
        for emb_u, u_mask, u_type in emb_u_mask_pairs:
            for emb_v, v_mask, v_type in emb_v_mask_pairs:
                for emb_neg_v, neg_v_mask, neg_v_type in emb_neg_v_mask_pairs:
                    s = self._masked_score(emb_u, emb_v, emb_neg_v, u_mask, v_mask, neg_v_mask)
                    subscores[u_type + v_type + neg_v_type] = s

        all_scores = []
        for key, s in subscores.items():
            all_scores.append(s)
            subscores[key] = torch.mean(s)
        all_scores = torch.stack(all_scores)
        all_scores = torch.sum(all_scores, dim=0)
        return torch.mean(all_scores), subscores

    def save(self, filepath, overwrite=False):
        if os.path.exists(filepath) and not overwrite:
            yn = input(f"Save file already exists at {filepath}. Overwrite?\nY/N: ")
            if yn.lower() != "y":
                return None
        else:
            os.mkdir(os.path.dirname(filepath))
        torch.save(self, filepath)

    def save_keyed_vectors(self, id2word, file_name):
        """
        Save embeddings and materials in gensim kv format
        """
        def fetch_or_generate_target_embedding(u):
            if u >= self.num_regular_words:
                 stoich = self.stoichiometries.weight[u].transpose(-1, 0)
                 return self.tmeg(stoich.unsqueeze(0)).squeeze()
            else:
                return self.u_embeddings.weight[u]
        with open(file_name, 'w') as f:
                f.write('%d %d\n' % (len(id2word), self.emb_dimension))
                for wid, w in id2word.items():
                    e = ' '.join(map(lambda x: str(x), fetch_or_generate_target_embedding(wid).cpu().detach().numpy()))
                    f.write('%s %s\n' % (w, e))

    @staticmethod
    def load_from_file(filepath):
        return torch.load(filepath)
