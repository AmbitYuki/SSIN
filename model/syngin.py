import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tree import head_to_tree, tree_to_adj


class SynGINClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()
        in_dim = opt.hidden_dim
        self.opt = opt
        self.gin_model = GINAbsaModel(embedding_matrix=embedding_matrix, opt=opt)
        self.classifier = nn.Linear(in_dim, opt.polarities_dim)

    def forward(self, inputs):
        outputs = self.gin_model(inputs)
        logits = self.classifier(outputs)
        return logits, None

import torch
import torch.nn as nn
import torch.nn.functional as F

class GIN(nn.Module):
    def __init__(self, num_layers, in_dim, out_dim):
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(), 
            nn.Linear(out_dim, out_dim)
        )
        self.epsilon = nn.Parameter(torch.Tensor([0.]))
        
    def forward(self, x, adj):
        for l in range(self.num_layers):
            x = x + self.epsilon * adj.matmul(x)
            x = self.mlp(x)
        return x

class GINAbsaModel(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super().__init__()  
        self.emb = nn.Embedding.from_pretrained(embedding_matrix) 
        self.gin = GIN(opt.num_layers, opt.input_dim, opt.hidden_dim)
        
    def forward(self, inputs, adj):
        x = self.emb(inputs) 
        x = self.gin(x, adj)
        return x.mean(1)

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.opt.rnn_hidden, self.opt.rnn_layers, self.opt.bidirect)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        tok, asp, pos, head, deprel, post, mask, l, _ = inputs           # unpack inputs
        # embedding
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.opt.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.opt.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        self.rnn.flatten_parameters()
        gin_inputs = self.rnn_drop(self.encode_with_rnn(embs, l, tok.size()[0]))
        
        # gin layer
        denom = adj.sum(2).unsqueeze(2) + 1

        for l in range(self.layers):
            Ax = adj.bmm(gin_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            gin_inputs = self.gin_drop(gAxW) if l < self.layers - 1 else gAxW
 
        return gin_inputs


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()
