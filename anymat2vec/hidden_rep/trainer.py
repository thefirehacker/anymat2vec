"""
Parts of this code were adapted from https://github.com/Andras7/word2vec-pytorch/blob/master/word2vec/model.py

"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from anymat2vec.hidden_rep.data import DataReader, HiddenRepDataset
from anymat2vec.hidden_rep.model import HiddenRepModel

class HiddenRepTrainer:
    """
    Parts adapted from https://github.com/Andras7/word2vec-pytorch/blob/master/word2vec/trainer.py
    """

    # MIN_COUNT SET BACK TO 12, BATCH_SIZE BACK TO 32
    def __init__(self, input_file, save_directory_name="hr_save", emb_dimension=100, hidden_size=20, batch_size=1,
                 window_size=5,
                 epochs=3, initial_lr=0.001, min_count=2):

        self.data = DataReader(input_file, min_count)
        dataset = HiddenRepDataset(self.data, window_size)
        self.dataloader = DataLoader(dataset, batch_size=batch_size,
                                     shuffle=False, num_workers=0, collate_fn=dataset.collate)
        self.save_directory_name = save_directory_name
        self.emb_size = self.data.num_regular_words
        self.emb_dimension = emb_dimension
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_lr = initial_lr
        self.hidden_size = hidden_size
        self.stoichiometries = self.data.stoichiometries
        self.hidden_rep_model = HiddenRepModel(self.emb_size,
                                               self.emb_dimension,
                                               self.hidden_size,
                                               self.stoichiometries)

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        if self.use_cuda:
            self.hidden_rep_model.cuda()

    def train(self):
        # Send stoichiometry tensor to GPU
        self.stoichiometries.to(self.device)
        for epoch in range(self.epochs):
            print("\n\n\nEpoch: " + str(epoch + 1))
            optimizer = optim.Adam(self.hidden_rep_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(self.dataloader))

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
                    if i > 0 and i % 500 == 0:
                        print(" Loss: " + str(running_loss))

    def save_model(self):
        self.hidden_rep_model.save(self.data.id2word)


if __name__ == '__main__':
    hr = HiddenRepTrainer(input_file='anymat2vec/relevant_abstracts.txt')
    hr.train()