# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import torch
import torch.nn as nn
import esm
from domainbed.scripts.utils import exists
from domainbed.lib.modules import AbAgRotaryEmbedding


class ESMModel(torch.nn.Module):
    """ESM2 with linear layers"""
    def __init__(self, pretrained_esm2_model='esm2_t6_8M_UR50D'):
        super(ESMModel, self).__init__()
        self.pretrained_esm2_model = pretrained_esm2_model
        if self.pretrained_esm2_model not in [
            'esm2_t6_8M_UR50D',
            'esm2_t12_35M_UR50D',
            'esm2_t30_150M_UR50D',
            'esm2_t33_650M_UR50D',
            'esm2_t36_3B_UR50D',
            'esm2_t48_15B_UR50D']:
            raise ValueError("Pretrained model doesn't exist.")
        self.esm, self.alphabet = getattr(
            esm.pretrained, self.pretrained_esm2_model)()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.final_layer = int(
            self.pretrained_esm2_model.split('_')[1][1:])
        self.n_outputs = self.esm.embed_dim
        for layer_i in range(self.esm.num_layers):
            # Swap out rotary embedding
            self.esm.layers[layer_i].self_attn.rot_emb = AbAgRotaryEmbedding(
                self.esm.layers[layer_i].self_attn.head_dim
            )

    def get_batch_tensor_x(self, x):
        # # Convert output of default collate_fn to length batch_size list of 2-tuples
        _, _, x = self.batch_converter(x)
        return x

    def forward(self, x):
        batch_lens = (x != self.alphabet.padding_idx).sum(1)  # [B,]
        x = self.esm(
            x, repr_layers=[self.final_layer]
            )["representations"][self.final_layer]  # [B, L, n_outputs]
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(
                x[i, 1: tokens_len - 1].mean(0))
        out = torch.stack(sequence_representations)  # [B, n_outputs]
        return out  # [B, n_outputs]


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if input_shape == 'use_esm':
        return ESMModel()  # new ESM Ab8 setting
    else:
        return SeqCNN(input_shape[0], hparams)


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)


class WholeFish(nn.Module):
    def __init__(self, input_shape, num_classes, hparams, weights=None):
        super(WholeFish, self).__init__()
        self.featurizer = Featurizer(input_shape, hparams)
        self.classifier = Classifier(
            self.featurizer.n_outputs,
            num_classes,
            hparams['nonlinear_classifier'])
        self.net = nn.Sequential(
            self.featurizer, self.classifier
        )
        if weights is not None:
            self.load_state_dict(copy.deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(copy.deepcopy(weights))

    def forward(self, x):
        return self.net(x)


class SeqCNN(nn.Module):
    def __init__(self, vocab_size=22, lr=0.001, d_model=64, ksizes=[10, 5], stride=2, padding=0, regression=False, wd=1e-4):
        super(SeqCNN, self).__init__()
        self.lr = lr
        self.wd = wd
        self.regression=regression
        self.stride=stride
        self.padding=padding
        self.ks0=ksizes[0]
        self.ks1=ksizes[1]
        self.d_model = d_model
        self.n_outputs = 256        
        self.embed = nn.Embedding(vocab_size+1, self.d_model, padding_idx=0)

        self.emb_ab = nn.Embedding(22, self.d_model, padding_idx=0)
        self.emb_ag = nn.Embedding(22, self.d_model, padding_idx=0)

        # conv layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(self.d_model, self.d_model, kernel_size=self.ks0, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            nn.Conv1d(self.d_model, 2*self.d_model, kernel_size=self.ks1, stride=self.stride, padding=self.padding),
            nn.ReLU()
        )
        
        # feed forward
        self.ffd = nn.Linear(2*2*self.d_model, self.n_outputs)        


    def features(self, x):
        x_heavy, x_light = x
        
        # embed integers into vectors
        heavy = self.embed(x_heavy)
        light = self.embed(x_light)
        
        # conv layers
        h_heavy = self.conv_layers(heavy.permute(0, 2, 1))  # (batch_size, dim, L)
        h_light = self.conv_layers(light.permute(0, 2, 1))  # (batch_size, dim, L)
                
        # max pooling separately Heavy & Light Chain && concatenate
        h = torch.cat([torch.max(h_heavy, dim=-1)[0], torch.max(h_light, dim=-1)[0]], dim=-1)
        
        return h
        
    def forward(self, x):
        x_ab = x[:, :298]
        x_ag = x[:, 298:]

        e_ab = self.emb_ab(x_ab.long()) # [bs, 298, 256]
        e_ag = self.emb_ag(x_ag.long()) # [bs, 915, 256]

        # conv layers
        h_ab = self.conv_layers(e_ab.permute(0, 2, 1))  # (batch_size, dim, L)
        h_ag = self.conv_layers(e_ag.permute(0, 2, 1))  # (batch_size, dim, L)
                
        # max pooling separately Heavy & Light Chain && concatenate
        h = self.ffd(torch.cat([torch.max(h_ab, dim=-1)[0], torch.max(h_ag, dim=-1)[0]], dim=1))

        return h
