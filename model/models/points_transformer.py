import torch
import torch.nn as nn

class PointsTransformer(nn.Module):
    def __init__(
        self,
        nbr_max_car,
        nbr_points,
        nbr_variable,
        bn_momentum = 0.1,
    ):
        super().__init__()

        self.nbr_max_car = nbr_max_car
        self.nbr_points = nbr_points
        self.nbr_variable = nbr_variable
        self.bn_momentum = bn_momentum


        head_size = 1024

        self.queries = torch.nn.Parameter(torch.rand(
            1,
            nbr_points * nbr_max_car,
            head_size,
        ))

        self.keys = nn.Conv1d(
            in_channels=head_size,
            out_channels=head_size,
            kernel_size=1
        )

        self.values = nn.Conv1d(
            in_channels=head_size,
            out_channels=head_size,
            kernel_size=1
        )

        self.multihead_attn = nn.MultiheadAttention(head_size, 1, batch_first=True)
        self.final = nn.Sequential(
            nn.BatchNorm1d(head_size, momentum=bn_momentum),
            nn.Conv1d(
                in_channels=head_size,
                out_channels=nbr_points + 1 + nbr_variable,
                kernel_size=1
            ),
            nn.BatchNorm1d(nbr_points + 1 + nbr_variable, momentum=bn_momentum),
        )

    def forward(self, x):
        query = self.queries.expand(x.shape[0], *self.queries.shape[1:3])
        x = x.view(*x.shape[0:2],-1)
        
        key = self.keys(x).permute(0,2,1)
        value = self.values(x).permute(0,2,1)

        x = self.multihead_attn(query, key, value)[0]

        x = x.permute(0,2,1)
        x = self.final(x)
        x = x.permute(0,2,1)
        
        return x