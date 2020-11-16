"""
Parts of this code were adapted from https://github.com/Andras7/word2vec-pytorch/blob/master/word2vec/model.py

"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


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
        self.stoich_size = stoichiometries.shape[0]
        self.stoich_dimension = stoichiometries.shape[1]
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension)
        self.num_regular_words = num_regular_words
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)
        self.stoichiometries = nn.Embedding.from_pretrained(stoichiometries, freeze=True)
        self.shared_generator = self.make_stoich_to_emb_nn()
        # target material embedding generator head
        self.tmeg = torch.nn.Linear(self.hidden_size, self.emb_dimension)
        # context material embedding generator head
        self.cmeg = torch.nn.Linear(self.hidden_size, self.emb_dimension)

    def make_stoich_to_emb_nn(self):
        """ Initializes the neural network for turning a stoichiometry vector
        into an embedding

        TODO: fine-tune this architecture based on cross-validation.
        I expect some regularization might be called for.

        """
        model = torch.nn.Sequential(
            torch.nn.Tanh(),
            torch.nn.Linear(self.stoich_dimension, self.hidden_size),
            torch.nn.Tanh(),
            #torch.nn.Linear(self.hidden_size, self.hidden_size),
            #torch.nn.Tanh()
        )
        return model

    def _generate_embedding(self, uv, context=False):
        stoich = self.stoichiometries(uv)
        hrelu = self.shared_generator(stoich)
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
        emb_neg_v_m = self._generate_embedding(neg_v, context=True)

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

        L2Loss = 0
        for param in self.shared_generator.parameters():
            L2Loss += torch.linalg.norm(param,ord=1)


        weight_decay = 1.0

        all_scores = torch.mean(all_scores) + weight_decay*L2Loss
        return all_scores, subscores

    def save(self, filepath, overwrite=True):
        if os.path.exists(filepath) and not overwrite:
            yn = input(f"Save file already exists at {filepath}. Overwrite?\nY/N: ")
            if yn.lower() != "y":
                return None
        else:
            try:
                os.mkdir(os.path.dirname(filepath))
            except FileExistsError:
                pass
        torch.save(self, filepath)

    def save_keyed_vectors(self, id2word, file_name):
        """
        Save embeddings and materials in gensim kv format
        """
        def fetch_or_generate_target_embedding(u):
            if u >= self.num_regular_words:
                 stoich = self.stoichiometries.weight[u].unsqueeze(0)
                 stoich = self.shared_generator(stoich)
                 return self.tmeg(stoich).squeeze()
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
