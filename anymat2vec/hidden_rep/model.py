"""
Parts of this code were adapted from https://github.com/Andras7/word2vec-pytorch/blob/master/word2vec/model.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import os


class HiddenRepModel(nn.Module):
    """
    u_embedding: Embedding for center word.
    v_embedding: Embedding for context words.
    """

    def __init__(self, emb_size, emb_dimension, hidden_size, stoichiometries):
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
        self.stoich_size = stoichiometries.shape[0]
        self.stoich_dimension = stoichiometries.shape[1]
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)
        self.stoichiometries = stoichiometries
        # target material embedding generator
        self.tmeg = self.make_stoich_to_emb_nn()
        # context material embedding generator
        self.cmeg = self.make_stoich_to_emb_nn()

    def make_stoich_to_emb_nn(self):
        """ Initializes the neural network for turning a stoichiometry vector
        into an embedding

        TODO: fine-tune this architecture based on cross-validation.
        I expect some regularization might be called for.

        """
        model = torch.nn.Sequential(
            torch.nn.Linear(self.stoich_dimension, self.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_size, self.emb_dimension))
        return model

    def _fetch_or_generate_context_embedding(self, v):
        if v >= self.emb_size:
            stoich = self.stoichiometries[v - self.emb_size].transpose(-1, 0)
            return self.cmeg(stoich)
        else:
            return self.v_embeddings.weight[v]

    def _fetch_or_generate_target_embedding(self, u):
        if u >= self.emb_size:
            stoich = self.stoichiometries[u - self.emb_size].transpose(-1, 0)
            return self.tmeg(stoich)
        else:
            return self.u_embeddings.weight[u]

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
        emb_v = torch.stack([self._fetch_or_generate_context_embedding(neg_v_i) for neg_v_i in pos_v])
        emb_u = torch.stack([self._fetch_or_generate_target_embedding(neg_v_i) for neg_v_i in pos_u])
        emb_neg_v = torch.stack(
            [torch.stack([self._fetch_or_generate_context_embedding(neg_v_i) for neg_v_i in neg_v_r]) for neg_v_r in
             neg_v])

        # cross product of the context word with center word
        pos_score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        pos_score = torch.clamp(pos_score, max=10, min=-10)
        pos_score = -F.logsigmoid(pos_score)

        # batch matrix-matrix product, calculates the cross products
        # of the negative samples with center word
        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(pos_score + neg_score)

    def save(self, filepath, overwrite=False):
        if os.path.exists(filepath) and not overwrite:
            yn = input(f"Save file already exists at {filepath}. Overwrite?\nY/N: ")
            if yn.lower() != "y":
                return None
        else:
            os.mkdir(os.path.dirname(filepath))
        torch.save(self, filepath)

    @staticmethod
    def load_from_file(filepath):
        return torch.load(filepath)
