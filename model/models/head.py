import torch
import torch.nn as nn
import numpy as np

class Head(nn.Module):

    def __init__(
        self,
        nbr_max_car,
        nbr_points,
        nbr_variable,
        bn_momentum = 0.1,
        add_positional_encoding = True,
        nhead=4,
        num_layers=3,
    ):
        super().__init__()

        neck_size = 1024
        embed_size = 256
        position_size = 30

        self.queries = torch.nn.Parameter(torch.rand(
            nbr_points * nbr_max_car,
            embed_size,
        ))

        self.add_positional_encoding = add_positional_encoding
        if(add_positional_encoding):
            self.positional_encoding = torch.nn.Parameter(torch.rand(
                embed_size,
                position_size,
                position_size,
            ))

        self.conv = nn.Conv2d(
            in_channels=neck_size,
            out_channels=embed_size,
            kernel_size=1
        )

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_size, nhead=nhead, batch_first=True),
            num_layers=num_layers)

        self.final = nn.Sequential(
            nn.BatchNorm1d(embed_size, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=embed_size,
                out_channels=embed_size//2,
                kernel_size=1
            ),
            nn.BatchNorm1d(embed_size//2, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=embed_size//2,
                out_channels=nbr_points + 1 + nbr_variable,
                kernel_size=1
            ),
        )

    def cat_positional_encoding(self, x):
        device = x.device
        x_  = torch.tensor(range(x.shape[2]), dtype=torch.float64, device=device).expand(x.shape[0],1,x.shape[3],x.shape[2]).permute(0,1,3,2)
        y_  = torch.tensor(range(x.shape[3]), dtype=torch.float64, device=device).expand(x.shape[0],1,x.shape[2],x.shape[3])

        x = x[:,:-2,:,:]
        x = torch.cat((x, x_, y_), 1)
        return x.view(*x.shape[0:2],-1).permute(0,2,1)


    def forward(self, x):
        x = self.conv(x)
        if(self.add_positional_encoding):
            positional_encoding = self.positional_encoding[:, :x.shape[2], :x.shape[3]]
            positional_encoding = positional_encoding.expand(x.shape[0], *positional_encoding.shape)
            x += positional_encoding
            x = x.view(*x.shape[0:2],-1).permute(0,2,1)
        else:
            x = self.cat_positional_encoding(x)


        query = self.queries.expand(x.shape[0], *self.queries.shape)
        output = self.transformer_decoder(query, x)

        output = self.final(output.permute(0,2,1)).permute(0,2,1)
        
        return output