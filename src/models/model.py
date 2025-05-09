# MIT LICENSE
# Written by Dr. Inal Mashukov
# Affiliation: 
# University of Massachusetts Boston,
# Department of Computer Science,
# Artificial Intelligence Laboratory 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import copy, math


def _get_activation_fn(activation):
    if activation =='relu':
        return F.relu
    elif activation =='gelu':
        return F.gelu 
    else:
        raise ValueError(f"Activation should be relu or gelu, not {activation}")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



class PositionalEncoding(nn.Module):
    '''
    Positional Encoding Layer
    Args:
        -seq_len: the length of the input sequences
        -d_model: the embedding dimension
        -dropout: the dropout rate
    '''
    def __init__(self, seq_len: int, d_model: int, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        max_len = max(5000, seq_len)
        self.dropout = nn.Dropout(p = dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype= torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model,2).float() * (-math.log(10000.0)/ d_model))
        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model %2 ==0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position*div_term)[:, 0:-1]
        
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Permute(torch.nn.Module):
    def forward(self, x):
        return x.permute(1,0)


class TransformerEncoderLayer(nn.Module):

    '''
    Transformer Encoder Layer.
    Args:
        -d_model: embedding dimension, the number of expected features in the input
        -nhead: the number of heads in the multihead attention layer
        -dim_feedforward: the dimension (#neurons) of the feedforward network
        -dropout: the dropout rate
        -activation: the activation function
        -layer_norm_eps: the epsilon value in the layer normalization
    '''

    def __init__(self, d_model: int, nhead: int, 
                 dim_feedforward: int = 2048, dropout: float=0.1,
                 activation: str = 'relu', layer_norm_eps: float = 1e-5):
         super(TransformerEncoderLayer, self).__init__()
         # d_model is the embedding dimension
         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
         # feed forward network
         self.linear1 = nn.Linear(d_model, dim_feedforward)
         self.dropout = nn.Dropout(dropout)
         self.linear2 = nn.Linear(dim_feedforward, d_model)

         self.norm1 = nn.LayerNorm(d_model, eps = layer_norm_eps)
         self.norm2 = nn.LayerNorm(d_model, eps= layer_norm_eps)
         self.dropout1 = nn.Dropout(dropout)
         self.dropout2 = nn.Dropout(dropout)

         self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu 
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        '''
        Forward method passing the input through the encoder layer.
        Args:
            -src: the sequence to the encoder layer
            -src_mask: the mask for the src sequence (optional)
            -src_key_padding_mask: the mask for the src keys per batch (optional)
        '''
        src2, attn = self.self_attn(src, src, src, attn_mask = src_mask,
                                    key_padding_mask = src_key_padding_mask)
        
        src = src + self.dropout1(src2)
        
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
        src = src + self.dropout2(src2)

        src = self.norm2(src)

        return src, attn


class TransformerEncoder(nn.Module):

    '''
    Transformer Encoder is a stack of N encoder layers.
    Args:
        -encoder_layer: an instance of the TransformerEncoderLayer() class 
        -num_layers: the number of encoder layers in the stack
        -norm: layer normalization
    '''
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, device, norm = None):
        super(TransformerEncoder, self).__init__()
        self.device = device 
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers 
        self.norm = norm 

    def forward(self, src, mask = None, src_key_padding_mask = None):
        ''' 
        Pass the input through the encoder layers.
        Args:
            -src: the sequence inputs to the encoder
            -mask: the mask for the src sequence (optional)
            -src_key_padding_mask: the mask for the src keys per batch (optional)
        '''
        output = src 
        attn_output = torch.zeros((src.shape[1], src.shape[0], src.shape[0]), 
                                  device = self.device) # batch_size, seq_len, seq_len
        
        for mod in self.layers:
            output, attn = mod(output, src_mask = mask,
                                src_key_padding_mask = src_key_padding_mask)
            attn_output += attn 

        if self.norm is not None:
            output = self.norm(output)

        return output, attn_output
    


class Transformer(nn.Module):

    '''
    Encoder-only Transformer Model.
    Args:
        -device: device (cuda or cpu)
        -nclasses: number of classes for the classification task
        -seq_len: length of input sequence
        -batch: batch size
        -input_size: input dimension (#features)
        -emb_size: embedding dimension
        -nhead: number of attention heads
        -nhid: dimension (#neurons) of the hidden layers
        -nlayers: number of hidden layers and encoder stacks
        -dropout: dropout rate
    '''

    def __init__(self, device, nclasses: int, seq_len: int,
                 batch: int, input_size: int, emb_size: int,
                 nhead: int, nhid: int, nlayers: int, dropout: float = 0.1):
        super(Transformer, self).__init__()

        self.input_net = nn.Sequential(
            nn.Linear(input_size, emb_size),
            nn.BatchNorm1d(batch),
            PositionalEncoding(seq_len, emb_size, dropout),
            nn.BatchNorm1d(batch))
        
        encoder_layers = TransformerEncoderLayer(emb_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers, device)

        self.batch_norm = nn.BatchNorm1d(batch)

        self.class_net = nn.Sequential(
            nn.Linear(emb_size, nhid),
            nn.ReLU(),
            Permute(),
            nn.BatchNorm1d(batch),
            Permute(),
            nn.Dropout(p = 0.25),
            nn.Linear(nhid, nhid),
            nn.ReLU(),
            Permute(),
            nn.BatchNorm1d(batch),
            Permute(),
            nn.Dropout(0.25),
            nn.Linear(nhid, nclasses)
        )

    def forward(self, x):
        x = self.input_net(x.permute(1,0,2))
        x, attn = self.transformer_encoder(x)
        x = self.batch_norm(x)
        # x: seq_len x batch x emb_size

        # last row in the sequence:
        output = self.class_net(x[-1])

        return output, attn
    

# learning_rate = 0.0001
# dropout = 0.2 
# nclasses = len(np.unique(y_train))
# seq_len = X_train.shape[1]
# batch = 128 
# input_size = X_train.shape[2]
# emb_size, nhid, nhead, nlayers = 128, 1024, 4, 3 


# model = Transformer(device, nclasses, seq_len, batch, 
#                     input_size, emb_size, nhead, nhid, 
#                     nlayers, dropout).to(device)




# X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
# y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
# y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)


# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, drop_last=True)
# test_loader = DataLoader(test_dataset, batch_size= batch, shuffle=False, drop_last= True)


# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), 
#                              lr= learning_rate, weight_decay= 1e-5)


# print(model)


# num_epochs = 50

# for epoch in range(num_epochs):
#     model.train() # set the model into training mode
#     total_loss = 0 

#     for X_batch, y_batch in train_loader:
#         # zero the gradients:
#         optimizer.zero_grad()

#         # forward pass - prediction and loss computation:
#         outputs, _ = model(X_batch)
#         loss = criterion(outputs, y_batch)

#         # backward pass:
#         loss.backward()
#         # update weights
#         optimizer.step()

#         # aggregate total loss:
#         total_loss += loss.item()
    
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')


# def evaluate(model, dataloader):

#     model.eval() # set the model into eval mode
#     correct = 0
#     total = 0 

#     with torch.no_grad():
#         for X_batch, y_batch in dataloader:
#             outputs, _ = model(X_batch)
#             _, predicted = torch.max(outputs, 1)
#             correct += (predicted == y_batch).sum().item()
#             total += y_batch.size(0)

#     return correct/total 


# train_acc = evaluate(model, train_loader)
# test_acc = evaluate(model, test_loader)

# print(f'Training Accuracy: {train_acc:.4f}, Test Accuracy {test_acc:.4f}')

# torch.save({
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
# }, 'saved_models/model.pth')

# checkpoint = torch.load('model.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])