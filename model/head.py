import torch
import torch.nn as nn
import numpy as np

class Head(nn.Module):

    def __init__(
        self,
        nbr_max_car,
        nbr_points,
        nbr_variable,
        add_positional_encoding = True,
        nhead = 4,
        num_layers = 3,
        use_matcher = True,
        normalize_position=True,
        neck_size = 1024,
    ):
        super().__init__()
        self.normalize_position = normalize_position
        self.use_matcher = use_matcher
        self.nbr_points = nbr_points
        self.nbr_variable = nbr_variable
        #Sets the size of the embeddings to 256
        embed_size = 256
        #Sets the size of the postion to 30
        position_size = 30

        #Initializes random queries with size of nbr_points * nbr_max_car multiplied by the embedding size
        self.queries = torch.nn.Embedding(
            nbr_points * nbr_max_car,
            embed_size,
        )

        #Sets a boolean for the positional encoding to true
        self.add_positional_encoding = add_positional_encoding
        #If the positional encoding is true
        if(add_positional_encoding):
            #Initializes random positional encoding with size of the embedding size multiplied by position size multiplied by position size
            self.positional_encoding = torch.nn.Parameter(torch.rand(
                embed_size,
                position_size,
                position_size,
            ))

        #Initializes a 2d convolutional layer to reduce size
        self.conv = nn.Conv2d(
            in_channels=neck_size,
            out_channels=embed_size,
            kernel_size=1
        )

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

    def cat_positional_encoding(self, x):
        #Gets the device and the data type of x
        device = x.device
        dtype = x.dtype
        #Creates a tensor with the range of x's shape[2]
        x_  = torch.tensor(range(x.shape[2]), dtype=dtype, device=device).expand(x.shape[0],1,x.shape[3],x.shape[2]).permute(0,1,3,2)
        #Creates a tensor with the range of x's shape[3]
        y_  = torch.tensor(range(x.shape[3]), dtype=dtype, device=device).expand(x.shape[0],1,x.shape[2],x.shape[3])

        x[:,-2:-1,:,:] = x_
        x[:,-1:,:,:] = y_

        #Reshapes x
        return x.view(*x.shape[0:2],-1).permute(0,2,1)


    def forward(self, x):
        #Applies the convolutional layer to reduce size
        x = self.conv(x)
        #If the positional encoding is true
        if(self.add_positional_encoding):
            #Gets the positional encoding
            positional_encoding = self.positional_encoding[:, :x.shape[2], :x.shape[3]]
            #Expands the positional encoding to match the size of x
            positional_encoding = positional_encoding.expand(x.shape[0], *positional_encoding.shape)
            #Adds the positional encoding to x
            x += positional_encoding
            #Reshapes x
            x = x.view(*x.shape[0:2],-1).permute(0,2,1)
        else:
            #Applies the cat_positional_encoding to x
            x = self.cat_positional_encoding(x)

        #Expands the queries to match the size of x
        query = self.queries.weight.expand(x.shape[0], *self.queries.weight.shape)
        assert(torch.all(query[0]==query[1]))
        #Applies the transformer decoder
        output = self.transformer_decoder(query, x)

        #Applies a sequence of operations on the output of the transformer decoder
        output_cls = self.final_classification(output)
        output_pos = self.final_position(output).sigmoid() if self.normalize_position else self.final_position(output)
        output = torch.cat([output_pos, output_cls],dim=2)
        #Returns the output
        return output