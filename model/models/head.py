import torch
import torch.nn as nn

class Head(nn.Module):

    def __init__(
        self,
        nbr_max_car,
        nbr_points,
        nbr_variable,
        bn_momentum = 0.1,
    ):
        super().__init__()

        neck_size = 1024
        embed_size = 256
        position_size = 2

        self.queries = torch.nn.Parameter(torch.rand(
            1,
            nbr_points * nbr_max_car,
            embed_size,
        ))

        self.conv = nn.Conv2d(
            in_channels=neck_size,
            out_channels=embed_size - position_size,
            kernel_size=1
        )

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_size, nhead=1, batch_first=True),
            num_layers=3)

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

    def add_positional_encoding(self, x):
        x_  = torch.tensor(range(x.shape[2])).expand(x.shape[0],1,x.shape[3],x.shape[2]).permute(0,1,3,2)
        y_  = torch.tensor(range(x.shape[3])).expand(x.shape[0],1,x.shape[2],x.shape[3])

        x = torch.cat((x, x_, y_), 1)
        return x.view(*x.shape[0:2],-1).permute(0,2,1)


    def forward(self, x):
        query = self.queries.expand(x.shape[0], *self.queries.shape[1:3])
        x = self.conv(x)
        x = self.add_positional_encoding(x)

        output = self.transformer_decoder(query, x)

        output = output.permute(0,2,1)
        output = self.final(output)
        output = output.permute(0,2,1)
        
        return output