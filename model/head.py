import torch
import torch.nn as nn
import numpy as np

class Head(nn.Module):

    def __init__(
        self,
        nbr_max_car,
        nbr_points,
        nbr_variable,
        nhead = 4,
        num_layers = 3,
        use_matcher = True,
        normalize_position=True,
        embed_size = 480,
    ):
        super().__init__()
        self.normalize_position = normalize_position
        self.use_matcher = use_matcher
        self.nbr_points = nbr_points
        self.nbr_variable = nbr_variable

        #Initializes random queries with size of nbr_points * nbr_max_car multiplied by the embedding size
        self.queries = torch.nn.Parameter(torch.rand(
            nbr_points * nbr_max_car,
            embed_size,
        ))

        #Initializes a transformer decoder layer
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embed_size, nhead=nhead, batch_first=True),
            num_layers=num_layers)

        self.final_classification = nn.Sequential(
                nn.Linear(embed_size,self.nbr_points+1 if self.use_matcher else 1)
            )

        self.final_position = nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.ReLU(),
                nn.Linear(embed_size, embed_size),
                nn.ReLU(),
                nn.Linear(embed_size, embed_size),
                nn.ReLU(),
                nn.Linear(embed_size, nbr_variable)
            )

    def forward(self, x):
        #Expands the queries to match the size of x
        query = self.queries.expand(x.shape[0], *self.queries.shape)
        assert(torch.all(query[0]==query[1]))
        #Applies the transformer decoder
        output = self.transformer_decoder(query, x)

        #Applies a sequence of operations on the output of the transformer decoder
        output_cls = self.final_classification(output)
        output_pos = self.final_position(output).sigmoid() if self.normalize_position else self.final_position(output)
        output = torch.cat([output_pos, output_cls],dim=2)
        #Returns the output
        return output