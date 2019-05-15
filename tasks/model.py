import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
# import torch.legacy as legacy

from torch.autograd import Variable
from torch.nn.parameter import Parameter

import numpy as np
import collections
import math
import sys
import logging

class Model(nn.Module):
    def __init__(self, input_dim, category_emb, category_dim, hidden_dim, embed_dim, output_dim,
                linear_dropout, dist_fn, learning_rate, weight_decay):

        super(Model, self).__init__()

        # Save model configs
        self.embed_dim = embed_dim
        self.dist_fn = dist_fn

        self.category_emb = category_emb
        self.category_dim = category_dim

        if self.category_emb:
            wide = embed_dim*embed_dim+embed_dim+34
            print("category_emb switched ON")

        else:
            wide = embed_dim*embed_dim+embed_dim
            print("category_emb switched OFF")

        # Model 0 : Basic Siamese Model
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # 300 x 300
            nn.Dropout(linear_dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(linear_dropout),
            nn.ReLU()
        )
        self.print_network(self.encoder, "encoder")\

        # Model 1 : COSINE

        # Model 2 : Concat (2 Layers)
        if self.dist_fn == "concat":
            self.dist_concat = nn.Sequential(
                nn.Linear(embed_dim*2, embed_dim),
                nn.Dropout(linear_dropout),
                nn.ReLU(),
                nn.Linear(embed_dim, int(embed_dim)),
				nn.Dropout(linear_dropout),
                nn.ReLU(),
				nn.Linear(int(embed_dim), output_dim),
            )
            self.print_network(self.dist_concat, "dist_concat")

        # Model 3: Wide and Deep
        elif self.dist_fn == "widedeep":
            self.deep = nn.Sequential(
                nn.Linear(embed_dim*2, embed_dim),
                nn.Dropout(linear_dropout),
                nn.ReLU(),
		        nn.Linear(embed_dim, embed_dim),
                nn.Dropout(linear_dropout),
                nn.ReLU()
            )
            self.print_network(self.deep, "deep")
            self.wide_deep = nn.Sequential(
                nn.Linear(wide, output_dim),
            )
            self.print_network(self.wide_deep, "wide_deep")

        # init_layers()
        self.init_layers()

        # Optimizer
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                    lr=learning_rate, weight_decay=weight_decay)

        # Loss Functions
        mse_crit = nn.MSELoss().cuda()
        cosine_crit = nn.CosineEmbeddingLoss(0.1).cuda()
        self.criterion = [mse_crit, cosine_crit]

    def forward(self, d1_r, d1_c, d1_l, d2_r, d2_c, d2_l):
        siamese_embed1 = self.model_siamese(d1_r)
        siamese_embed2 = self.model_siamese(d2_r)
        outputs = [siamese_embed1, siamese_embed2]

        if self.dist_fn == "concat":
            output_concat = self.model_concat(siamese_embed1, siamese_embed2)
            output = output_concat
            outputs.append(output)

        elif self.dist_fn == "cos":
            output_cos = F.cosine_similarity(siamese_embed1 + 1e-16, siamese_embed2 + 1e-16, dim=-1)
            output = output_cos
            outputs.append(output)

        elif self.dist_fn == "widedeep":
            output_widedeep = self.model_widedeep(siamese_embed1, siamese_embed2, d1_c, d2_c)
            output = output_widedeep
            outputs.append(output)

        return outputs

    def get_loss(self, outputs, targets):
        preds = outputs[2]
        loss = self.criterion[0](preds, targets)
        loss = torch.sqrt(loss)
        return loss

    # Model 0
    def model_siamese(self, d_r):
        inputs = d_r
        return self.encoder(inputs.float())

    # Model 2 : Concat (2 Layers)
    def model_concat(self, siamese_embed1, siamese_embed2):
        x = torch.cat((siamese_embed1, siamese_embed2), 1)
        x = self.dist_concat(x)
        #x = self.norm(x)
        output = torch.squeeze(x, 1)
        return output

    # Model 3: Wide and Deep
    def model_widedeep(self, siamese_embed1, siamese_embed2, d1_c, d2_c):
        batch_size = list(siamese_embed1.size())[0]

        # Wide
        w = torch.bmm(siamese_embed1.unsqueeze(2), siamese_embed2.unsqueeze(1))
        w = w.reshape(batch_size, -1)

        # Deep
        d = torch.cat((siamese_embed1, siamese_embed2), 1)
        d = self.deep(d)

        # Wide & Deep
        if self.category_emb:
            wd = torch.cat((w, d, d1_c, d2_c), 1)
        else:
            wd = torch.cat((w, d), 1)

        y = self.wide_deep(wd)
        output = torch.squeeze(y, 1)
        return output

    def init_layers(self):
        nn.init.xavier_normal_(self.encoder[0].weight.data)
        nn.init.xavier_normal_(self.encoder[3].weight.data)

        if self.dist_fn == "concat":
            nn.init.xavier_normal_(self.dist_concat[0].weight.data)
            nn.init.xavier_normal_(self.dist_concat[3].weight.data)
            nn.init.xavier_normal_(self.dist_concat[6].weight.data)

        elif self.dist_fn == "widedeep":
            nn.init.xavier_normal_(self.deep[0].weight.data)
            nn.init.xavier_normal_(self.deep[3].weight.data)
            nn.init.xavier_normal_(self.wide_deep[0].weight.data)




    def norm(input, p=2, dim=1, eps=1e-12):
        # Needs to be fixed!
        return input / input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)

    def print_network(self, model, name):

        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print("Model Name: \"{}\"".format(name))
        print(model)
        print("The number of parameters: {}".format(num_params))

    def save_checkpoint(self, state, checkpoint_dir, filename):
        filename = checkpoint_dir + filename
        torch.save(state, filename)

    def load_checkpoint(self, checkpoint_dir, filename):
        filename = checkpoint_dir + filename
        checkpoint = torch.load(filename)

        self.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
