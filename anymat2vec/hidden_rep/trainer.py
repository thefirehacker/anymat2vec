"""
Parts of this code were adapted from https://github.com/Andras7/word2vec-pytorch/blob/master/word2vec/model.py

"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from anymat2vec import MODELS_DIR
from anymat2vec.hidden_rep.data import DataReader, HiddenRepDataset
from anymat2vec.hidden_rep.model import HiddenRepModel
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class HiddenRepTrainer:
    """
    Parts adapted from https://github.com/Andras7/word2vec-pytorch/blob/master/word2vec/trainer.py
    """

    def __init__(self, input_file, save_directory_name="hr_save", emb_dimension=200, hidden_size=20, batch_size=32,
                 window_size=8, n_epochs=30, initial_lr=0.001, min_count=10, use_vanilla_word2vec=False):

        _, file_extension = os.path.splitext(input_file)

        if file_extension == ".pt":
            print("Loading preprocessed dataset. Please ensure that the original corpus file is also available \n"
                  "in the same directory and has the same name as the preprocessed input file except a '.txt' extension.")
            self.data = DataReader.from_save(input_file)
        else:
            self.data = DataReader(input_file, min_count)
            self.data.save(input_file.replace(".txt", ".pt"))
        dataset = HiddenRepDataset(self.data, window_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)
        self.save_directory_name = save_directory_name
        self.emb_size = len(self.data.word2id)
        if use_vanilla_word2vec:
            self.num_regular_words = self.emb_size
        else:
            self.num_regular_words = self.data.num_regular_words

        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.initial_lr = initial_lr
        self.hidden_size = hidden_size
        self.stoichiometries = torch.cat((torch.zeros((self.data.num_regular_words,
                                                       self.data.stoichiometries.size()[1]),
                                                      dtype=self.data.stoichiometries.dtype),
                                          self.data.stoichiometries.to_dense()))
        self.hidden_rep_model = HiddenRepModel(self.emb_size,
                                               self.emb_dimension,
                                               self.hidden_size,
                                               self.stoichiometries,
                                               self.num_regular_words)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.hidden_rep_model.cuda()

    def train(self):

        writer = SummaryWriter()

        # Send stoichiometry tensor to GPU
        self.stoichiometries = self.stoichiometries.to(self.device)
        optimizer = optim.Adam(self.hidden_rep_model.parameters(), lr=self.initial_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader) * self.n_epochs)

        for epoch in range(self.n_epochs):
            print("\n\n\nEpoch: " + str(epoch + 1))
            running_loss = 0.0
            for i, sample_batched in enumerate(tqdm(self.dataloader)):

                if len(sample_batched[0]) > 1:

                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.hidden_rep_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    if i > 0 and i % 100 == 0:
                        print(" Loss: " + str(running_loss))
                        writer.add_scalar('Loss', loss.item(), i + epoch*len(self.dataloader.dataset))
            hrt.save_model(checkpoint_number=epoch)

    def save_model(self, save_dir=os.path.join(MODELS_DIR, "hr_checkpoints"), checkpoint_number=None):
        if checkpoint_number:
            fn = f"checkpoint_epoch_{checkpoint_number}.pt"
        else:
            fn = "checkpoint.pt"
        self.hidden_rep_model.save(os.path.join(save_dir, fn))


if __name__ == '__main__':
    hrt = HiddenRepTrainer(input_file='data/relevant_abstracts.pt', batch_size=32)
    hrt.train()
    # hrt.data.save("data/tiny_corpus_loaded.pt")
