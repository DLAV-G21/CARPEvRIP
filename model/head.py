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
        nhead = 4,
        num_layers = 3,
        use_matcher = True,
    ):
        super().__init__()
        #Sets the size of the neck (the middle layer) to 1024
        neck_size = 1024
        #Sets the size of the embeddings to 256
        embed_size = 256
        #Sets the size of the postion to 30
        position_size = 30

        #Initializes random queries with size of nbr_points * nbr_max_car multiplied by the embedding size
        self.queries = torch.nn.Parameter(torch.rand(
            nbr_points * nbr_max_car,
            embed_size,
        ))

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

        #A sequence of operations on the output of the transformer decoder
        self.final = nn.Sequential(
            #Batch normalization with a momentum of 0.1
            nn.BatchNorm1d(embed_size, momentum=bn_momentum),
            #ReLU activation
            nn.ReLU(),
            #1d convolutional layer 
            nn.Conv1d(
                in_channels=embed_size,
                out_channels=embed_size//2,
                kernel_size=1
            ),
            #Batch normalization with a momentum of 0.1
            nn.BatchNorm1d(embed_size//2, momentum=bn_momentum),
            #ReLU activation
            nn.ReLU(),
            #1d convolutional layer 
            nn.Conv1d(
                in_channels=embed_size//2,
                #Output channels will be the number of points plus one plus the number of variables if use matcher is true. Otherwise, it will be the number of variables plus one
                out_channels=nbr_points + 1 + nbr_variable if use_matcher else nbr_variable + 1,
                kernel_size=1
            ),
        )

    def cat_positional_encoding(self, x):
        #Gets the device and the data type of x
        device = x.device
        dtype = x.dtype
        #Creates a tensor with the range of x's shape[2]
        x_  = torch.tensor(range(x.shape[2]), dtype=dtype, device=device).expand(x.shape[0],1,x.shape[3],x.shape[2]).permute(0,1,3,2)
        #Creates a tensor with the range of x's shape[3]
        y_  = torch.tensor(range(x.shape[3]), dtype=dtype, device=device).expand(x.shape[0],1,x.shape[2],x.shape[3])

        #Removes the last two shape of x
        x = x[:,:-2,:,:]
        #Concatenates x, x_, and y_
        x = torch.cat((x, x_, y_), 1)
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
        query = self.queries.expand(x.shape[0], *self.queries.shape)
        #Applies the transformer decoder
        output = self.transformer_decoder(query, x)

        #Applies a sequence of operations on the output of the transformer decoder
        output = self.final(output.permute(0,2,1)).permute(0,2,1)
        
        #Returns the output
        return output