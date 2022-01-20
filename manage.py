#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import torch
from torch import nn
import torch.nn.functional as f
import numpy as np
class TransformerCNN(nn.Module):
    def __init__(
        self, embed_weights, n_heads,
        n_feed, dropout, n_layers, seq_length, kernal_size
        ):

        super().__init__()
        
        embed_weight = torch.from_numpy(np.zeros((12784,300)))

        self.embed_dim = 300

        self.embedding = nn.Embedding.from_pretrained(
                                embed_weight, freeze=False)
        

        self.trans_enc_layer = nn.TransformerEncoderLayer(
                                d_model = self.embed_dim,
                                nhead = n_heads,
                                dim_feedforward = n_feed,
                                dropout = dropout,
                                batch_first = True,
                                )
        
        self.trans_encoder = nn.TransformerEncoder(
                                encoder_layer = self.trans_enc_layer,
                                num_layers = n_layers,
                                )
        

        self.seq_length = seq_length

        self.kernal_size = kernal_size
        self.conv_1d = nn.Conv1d(
                                in_channels=self.embed_dim,
                                out_channels = self.seq_length,
                                kernel_size  = kernal_size
                                )


        self.pool_1d = nn.AvgPool1d(kernel_size=self.seq_length-kernal_size)

        self.linear = nn.Linear(self.seq_length, 13)



    def forward(self, ids, masks):
        embed_out = self.embedding(ids)
        trans_out = self.trans_encoder(embed_out, src_key_padding_mask = masks)
        conv_out = self.conv_1d(trans_out.reshape(-1,self.embed_dim, self.seq_length))
        pool_out = self.pool_1d(f.dropout(f.relu(conv_out),0.2))
        final_out = self.linear(f.dropout(f.relu(pool_out.view(-1,self.seq_length)),0.2))
        return final_out


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'NewsClassifier.settings')
    try:
        from django.core.management import execute_from_command_line
        import django
        django.setup()

        # Override default port for `runserver` command
        from django.core.management.commands.runserver import Command as runserver
        runserver.default_port = "80"

    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
