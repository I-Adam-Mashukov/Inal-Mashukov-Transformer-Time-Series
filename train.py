# MIT LICENSE
# Written by Dr. Inal Mashukov
# Affiliation: 
# University of Massachusetts Boston,
# Department of Computer Science,
# Artificial Intelligence Laboratory 

import torch 
import argparse
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from src.models.model import Transformer
from src.utils import utils


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type = str,
                        required= True, help='Model name to be saved.')
    parser.add_argument('--gpu', type = int, 
                        default= 0, required= True, help="GPU index to use.")
    args = parser.parse_args()
    gpu = args.gpu
    name = args.name

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

    print(f'\nSelected device: {gpu} \n')

    # loading model and training parameters:
    params = utils.get_parameters()
    training_params = params['train']
    model_params = params['model']
    print('********************** Model Parameters ******************** \n')
    print(f'{params} \n')

    # loading the data:
    X_train, y_train, X_test, y_test = utils.preprocess(*utils.data_loader(), 
                                                        batch_size= training_params['batch_size'])
    
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # getting remaining parameters:
    nclasses = int(torch.max(y_train).item()) + 1
    seq_len = X_train.size(1) #[1]
    input_size = X_train.size(2) #[2]

    print(f'\nSequence Length: {X_train.size(1)}\nNumber of classes: {nclasses}\nNumber of features: {input_size}\n')

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

    model = Transformer(device = device, 
                        nclasses= nclasses,
                        seq_len= seq_len,
                        batch = training_params['batch_size'],
                        input_size= input_size,
                        emb_size= model_params['emb_size'],
                        nhead = model_params['nhead'],
                        nhid = model_params['nhid'],
                        nlayers = model_params['nlayers'],
                        dropout= model_params['dropout']).to(device)
    
    # defining loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr = training_params['learning_rate'])
    
    y_train = y_train.long()
    y_test = y_test.long()


    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=training_params['batch_size'], shuffle=False, drop_last=True)



    # print("******************* THE MODEL ******************** \n")
    # print(model)

    num_epochs = training_params['epochs']

    print('********************** Training Start ********************** \n')

    for epoch in range(num_epochs):
        
        model.train() # set the model into training mode
        # start tracking the training loss:
        total_loss = 0

        for X_batch, y_batch in train_loader:
            # zero the gradients:
            optimizer.zero_grad()

            # forward pass - prediction and loss computation:
            outputs, _ = model(X_batch)
            loss = criterion(outputs, y_batch)

            # backward pass:
            loss.backward()
            # update weights
            optimizer.step()

            # aggregate total loss:
            total_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')


        # training and testing accuracy:
        train_acc = utils.evaluate(model, train_loader)
        test_acc = utils.evaluate(model, test_loader)

        print(f'Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}')

    # print('Saving the model ... \n')
    # torch.save({'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict()
    #             }, f'./experiments/model-{name}.pth')
    
    print('\n********************** Training Complete **********************\n')



# checkpoint = torch.load('model.pth')
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == '__main__':
    main()


# import torch 
# import argparse
# from torch.utils.data import DataLoader
# from torch.utils.data import TensorDataset
# import numpy as np

# from src.models.model import Transformer
# from src.utils import utils

# '''
# These were bad:
# // {
# //     "train": {
# //         "epochs": 10,
# //         "batch_size" : 32,
# //         "learning_rate": 0.001
# //     },

# //     "model": {
# //         "emb_size" : 2048,
# //         "nhead": 8,
# //         "nhid": 1024,
# //         "nlayers": 4,
# //         "dropout": 0.1
# //     }

# // }
# '''


# def main():

#     parser = argparse.ArgumentParser()

#     parser.add_argument('--name', type = str,
#                         required= True, help='Model name to be saved.')
#     parser.add_argument('--gpu', type = int, 
#                         default= 0, required= True, help="GPU index to use.")
#     args = parser.parse_args()
#     gpu = args.gpu
#     name = args.name

#     device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')

#     print(f'\nSelected device: {gpu} \n')

#     # loading model and training parameters:
#     params = utils.get_parameters()
#     training_params = params['train']
#     model_params = params['model']
#     print('********************** Model Parameters ******************** \n')
#     print(f'{params} \n')

#     # loading the data:
#     X_train, y_train, X_test, y_test = utils.data_loader()
    
#     # getting remaining parameters:
#     # nclasses = len(np.unique(y_train))
#     # seq_len = X_train.shape[1] 
#     # input_size = X_train.shape[2]

#     # X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
#     # y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
#     # X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
#     # y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

#     # print(f'\nSequence Length: {seq_len}\nNumber of classes: {nclasses}\nNumber of features: {input_size}\n')

#     '''
#     Encoder-only Transformer Model.
#     Args:
#         -device: device (cuda or cpu)
#         -nclasses: number of classes for the classification task
#         -seq_len: length of input sequence
#         -batch: batch size
#         -input_size: input dimension (#features)
#         -emb_size: embedding dimension
#         -nhead: number of attention heads
#         -nhid: dimension (#neurons) of the hidden layers
#         -nlayers: number of hidden layers and encoder stacks
#         -dropout: dropout rate
#     '''

#     model = Transformer(device = device, 
#                         nclasses= nclasses,
#                         seq_len= seq_len,
#                         batch = training_params['batch_size'],
#                         input_size= input_size,
#                         emb_size= model_params['emb_size'],
#                         nhead = model_params['nhead'],
#                         nhid = model_params['nhid'],
#                         nlayers = model_params['nlayers'],
#                         dropout= model_params['dropout']).to(device)
    
#     # defining loss and optimizer
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.parameters(), 
#                                  lr = training_params['learning_rate'], weight_decay= 1e-5)
    


#     # train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     # test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

#     # train_loader = DataLoader(train_dataset, batch_size= training_params['batch_size'], shuffle=True, drop_last=True)
#     # test_loader = DataLoader(test_dataset, batch_size= training_params['batch_size'], shuffle=False, drop_last= True)



#     # print("******************* THE MODEL ******************** \n")
#     # print(model)

#     num_epochs = training_params['epochs']

#     print('********************** Training Start ********************** \n')

#     for epoch in range(num_epochs):
        
#         model.train() # set the model into training mode
#         # start tracking the training loss:
#         total_loss = 0

#         for X_batch, y_batch in train_loader:
#             # zero the gradients:
#             optimizer.zero_grad()

#             # forward pass - prediction and loss computation:
#             outputs, _ = model(X_batch)
#             loss = criterion(outputs, y_batch)

#             # backward pass:
#             loss.backward()
#             # update weights
#             optimizer.step()

#             # aggregate total loss:
#             total_loss += loss.item()
        
#         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')


#     # training and testing accuracy:
#     train_acc = utils.evaluate(model, train_loader)
#     test_acc = utils.evaluate(model, test_loader)

#     print(f'Training accuracy: {train_acc:.4f}, Test accuracy: {test_acc:.4f}')

#     # print('Saving the model ... \n')
#     # torch.save({'model_state_dict': model.state_dict(),
#     #             'optimizer_state_dict': optimizer.state_dict()
#     #             }, f'./experiments/model-{name}.pth')
    
#     print('\n********************** Training Complete **********************\n')



# # checkpoint = torch.load('model.pth')
# # model.load_state_dict(checkpoint['model_state_dict'])
# # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# if __name__ == '__main__':
#     main()