"""
This module consists of the classes and functions required for the classifier model and the ensemble model, 
along with those needed to manage the question dataset
"""
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset

from copy import deepcopy


''' class for creating an iterator to the questions dataset '''
class CreateDataset(Dataset):

    # method to initialize the class variables
    def __init__(self, questions, limits, labels):
        self.questions = questions
        self.limits = limits
        self.labels = labels

    # method to calculate the total number of samples in the datset
    def __len__(self):
        return len(self.labels)

    # method to get a single sample from the dataset
    def __getitem__(self, index):
        return self.questions[index], self.limits[index], self.labels[index]



''' class for the Question classification model '''
class Classifier(nn.Module):

    # method to initialize the class variables
    def __init__(self, vocabSize, embDim, numOfLabels, hidLayers, bow, randomEmbed, pre_emb, freeze, lr=0.01, gamma=0.9, device='cpu'):

        super().__init__()
        
        self.trainLoss = []
        self.valLoss = []
        self.trainAcc = []
        self.valAcc = []
        self.lr = lr
        self.gamma = gamma
        self.device = device
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.vocabSize = vocabSize
        self.hidLayers = hidLayers
        self.randomEmbed = randomEmbed
        self.bow = bow
        
        # FOR EMBEDDING STAGE
        # in case of random embeddings selected for the model embedding layer
        if self.randomEmbed:
            self.embDim = embDim
        # in case of pretrained-embedding selected for the model embedding layer
        else:
            pre_emb = torch.tensor(pre_emb)
            # setting the output embedding dimension as the pretrained model's shape
            self.embDim = pre_emb.shape[-1]

        self.numOfLabels = numOfLabels
        self.bow = bow

        # SENTENCE REPRESENTATION STAGE
        # 1) in case of Bag of Words (BOW) being selected for the model embedding layer
        if self.bow:        
            # EMBEDDING STAGE
            # 1.1) in case of random embeddings selected for the model embedding layer
            if self.randomEmbed:
                # using random weights to create the embedding layer
                self.embedding = nn.EmbeddingBag(self.vocabSize, self.embDim, mode='mean', sparse=True)
            # 1.2) in case of pretrained-embedding selected for the model embedding layer
            else:
                # loading pretrained embedding to the embedding layer
                self.embedding = nn.EmbeddingBag.from_pretrained(pre_emb, mode='mean', sparse=True, freeze=freeze)
        
        # 2) n case of BILSTM being selected for the model
        else:
            # EMBEDDING STAGE
            # 2.1) in case of random embeddings selected for the model embedding layer
            if self.randomEmbed:
                # using random weights to create the embedding layer
                self.embedding = nn.Embedding(self.vocabSize, self.embDim)
            # 2.2) in case of pretrained-embedding selected for the model embedding layer
            else:
                 # loading pretrained embedding to the embedding layer
                self.embedding = nn.Embedding.from_pretrained(pre_emb, freeze=freeze)

            # adding the bilstm layer to the model
            self.lstm = nn.LSTM(self.embDim, self.hidLayers, bidirectional=True)

        self.embedding.padding_idx = 0
        # adding the linear dense layer, of the shape : sentence_vector x numOfLabels
        self.fc = nn.Linear(self.hidLayers*2 if not self.bow else self.embDim, self.numOfLabels)

        # setting the hyper parameters for the model
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=self.gamma)


    ''' method for a single forward pass for the neural network '''
    def forward(self, text, limits):

        # creating sentence vector in case of BOW selected for the model
        if self.bow:
            sentVector = self.embedding(text)

        # creating sentence vector in case of BILSTM selected for the model
        else:
            embedded = self.embedding(text)
            # creating packed sequence
            packedEmbed = nn.utils.rnn.pack_padded_sequence(
                embedded, limits.cpu(), batch_first=True, enforce_sorted=False)

            _output, (hidden, _c_n) = self.lstm(packedEmbed)
            # concatenating the final forward and backward hidden states
            sentVector = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # returning the output of the final activation : Softmax
        return self.fc(sentVector)


    ''' method to train the model '''
    def train(self, trainData):

        trainLoss = 0
        trainAcc = 0
        # looping through the training data in batches
        for batch, example in enumerate(trainData):
            self.optimizer.zero_grad()
            text, limits, cls = example[0].to(self.device), example[1].to(self.device), example[2].to(self.device)
            output = self(text, limits)
            loss = self.criterion(output, cls)
            # calculating the training loss
            trainLoss += loss.item()
            loss.backward()
            self.optimizer.step()
            # calculating the training accuracy
            trainAcc += (output.argmax(1) == cls).sum().item()/len(output)

        # using scheduler to adjust the learning rate
        self.scheduler.step()

        # adding training loss calculated to the collection
        self.trainLoss.append(trainLoss / len(trainData))
        # adding training accuracy calculated to the collection
        self.trainAcc.append(trainAcc / len(trainData))

        return trainLoss / len(trainData), trainAcc / len(trainData)


    '''  method to test the model '''
    def test(self, testData):

        loss = 0
        acc = 0
        for example in testData:
            text, limits, cls = example[0].to(self.device), example[1].to(self.device), example[2].to(self.device)
            with torch.no_grad():
                output = self(text, limits)

                loss = self.criterion(output, cls)
                # calculating the test/validation loss
                loss += loss.item()
                # calculating the test/validation accuracy
                acc += (output.argmax(1) == cls).sum().item() / len(output)

        # adding the test/validation loss to the collection
        self.valLoss.append(loss / len(testData))
        # adding the test/validation accuracy to the collection
        self.valAcc.append(acc / len(testData))

        return loss / len(testData), acc / len(testData)


    '''  method to predict the label tokens for the question given using the model '''
    def predict(self, sentences):
       
        preds = []
        for sentence in sentences:
            # converting sentence to tensor
            tensor = torch.LongTensor(sentence).to(self.device)  
            # reshaping the sentence tensor
            tensor = tensor.unsqueeze(1).T
            # converting the number of words in the sentence to tensor
            lt = torch.LongTensor([len(sentence)])  
            # predicting the label for the sentence
            pred = self(tensor, lt)  
            preds.append(pred.argmax().item())

        return np.array(preds)


    '''  method to fit the model using the training/validation data'''   
    def fit(self, train_data, validation, epochs, modelPath="../data/models/saved_weights.model"):

        maxAcc = 0
        # training the model in epochs
        for epoch in range(epochs):
            trainLoss, trainAcc = self.train(train_data)
            valLoss, valAcc = self.test(validation)
            if valAcc > maxAcc:
                maxAcc = valAcc
                # saving the model configuration file for the final model
                torch.save(self.state_dict(), modelPath)

            # printing the details for each epoch while training the model
            print(f'Epoch {epoch+0:03}: | Train Loss: {trainLoss:.5f} | Val Loss: {valLoss:.5f} | Train Acc: {trainAcc:.3f}| Val Acc: {valAcc:.3f}')
        
        return maxAcc


''' class for the Ensemble model made by combining the results of trained models and bagging it '''
class Ensemble():

    # method to initialize the class variables  
    def __init__(self, model_list, device='cpu'):
        self.models = deepcopy(model_list)
        self.device = device

    def fit(self, train_data, validation, epochs, modelPath="../data/models/saved_weights.model"):
        return np.mean([model.fit(train_data, validation, epochs, modelPath+"."+str(i)) for i, model in enumerate(self.models)])

    # method to predict the labels for the sentences using the ensemble model  
    def predict(self, sentences):
        predictions = np.array([model.predict(sentences)
                                for model in self.models])
        return np.array([Counter(predictions[:, i]).most_common(1)[0][0]
                         for i in range(predictions.shape[1])])
