"""
This module consists of the script, which helps in the overall training and testing of the different models
"""
import argparse
import configparser
import pickle
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data.sampler import SubsetRandomSampler

from model_builder import Classifier, CreateDataset, Ensemble
from preprocess_prepare import CreateVocab, LabelEncoder

import warnings
warnings.filterwarnings('ignore')

# fixing the random seed to ensure reproducibility of results
SEED = 1
random.seed(SEED)
torch.manual_seed(SEED)

# checking if cuda is available and setting device to cuda instead of cpu, for faster processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# setting default hyper-parameters for the model
lr = 0.2
gamma = 0.9
EPOCHS = 20
minCount = 3
BATCH_SIZE = 32
embDim = 200
hidNodes = 50
freezeTuning = False

# setting default configs for the model, which help determine the model configuration to be used
bow = False
randomEmbed = False

TEXT = None
LABEL = None
dataset = None

ensembleSize = 0

# setting the paths for files being used
modelPath = '../data/models/model'
testPath = "../data/dataset/test.txt"
trainPath = "../data/dataset/train.txt"
text_vocab_path = '../data/dataset/text.vocab'
embPath = '../data/models/glove.small.txt'
label_vocab_path = '../data/dataset/label.vocab'


# method to train the model
def train():
    


    # creating the data indices for training and validation splits
    dataSize = len(dataset)
    indices = list(range(dataSize))
    # creating a train-validation split of 90-10
    valSplit = .1
    split = int(np.floor(valSplit * dataSize))
    np.random.seed(SEED)
    np.random.shuffle(indices)
    trainIdx, valIdx = indices[split:], indices[:split]

    # creating data samplers and loaders using data indices
    trainSampler = SubsetRandomSampler(trainIdx)
    valSampler = SubsetRandomSampler(valIdx)

    # creating batch generator
    trainGenerator = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=trainSampler)
    valGenerator = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, sampler=valSampler)

    # print the model architecture
    print("######################################################################################")
    print(model)
    print("######################################################################################")
    print(">>>>>>>> Selected Learning Rate : ", lr, " with Batch-Size: ", BATCH_SIZE, " <<<<<<<<")
    # training the model
    max_acc = model.fit(trainGenerator, valGenerator, EPOCHS, modelPath)
    print(f"Validation Accuracy: {max_acc * 100:.1f}% was saved")


# method to test the model
def test():

    print('Results for Test Set')
    xTest = []
    yTest = []

    with open(testPath, errors='ignore') as fp:
        testQuestion = fp.readlines()

    for line in testQuestion:
        label, text = line.split(' ', 1)
        xTest.append(text)
        yTest.append(label)

    xTest = TEXT.sentenceEncoder(xTest)
    yTest = np.array(LABEL.labelEncoder(yTest))
    yPred = model.predict(xTest)

    test_acc = accuracy_score(yTest, yPred)
    score1 = precision_score(yTest, yPred, average='weighted')
    score2 = recall_score(yTest, yPred, average='weighted')
    # computing the performance metrics for the model
    performanceMetrics = [
        f'\tAccuracy: {test_acc * 100:.1f}%', f'\tPrecision-Score: {score1:.3f}', f'\tRecall-Score: {score2:.3f}']

    # writing the labels predicted to the output file
    with open("../data/output/output.txt", 'w') as fp:
        fp.writelines(
            map(lambda x: x+'\n', LABEL.labelDecoder(yPred)))
    with open("../data/output/performance.txt", 'w') as fp:
        fp.writelines(performanceMetrics)
    

    print(''.join(performanceMetrics))


# main method to run the code via command prompt
if __name__ == '__main__':

    # read the arguments given by the user
    parser = argparse.ArgumentParser()

    # get the config file specified by user
    parser.add_argument('--config', type=str, required=True,
                        help='Configuration file for the model')
    # if train parameter is specified by user
    parser.add_argument('--train', action='store_true',
                        help='Training the model')
    # if test parameter is specified by user
    parser.add_argument('--test', action='store_true',
                        help='Testing the model')
    
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.sections()
    config.read(args.config)

    # get the variables from the config file
    trainPath = config["PATH"]["training_path"]
    testPath = config["PATH"]["testing_path"]
    modelName = config["MODEL_SETTINGS"]["model"]
    modelPath = config["PATH"]["modelPath"]
    embPath = config["PATH"].get("embedPath")

    BATCH_SIZE = int(config["MODEL_SETTINGS"]["batch_size"])
    EPOCHS = int(config["MODEL_SETTINGS"]["epoch"])
    embDim = int(config["MODEL_SETTINGS"]["embedding_dimensions"])
    hidNodes = int(config["MODEL_SETTINGS"]["hidden_dimensions"])
    lr = float(config["MODEL_SETTINGS"]["learning_rate"])
    freezeTuning = config["MODEL_SETTINGS"].getboolean("freeze")
    minCount = int(config["MODEL_SETTINGS"]["minWordLimit"])
    bow = config["MODEL_SETTINGS"].getboolean("bow")
    randomEmbed = config["MODEL_SETTINGS"].getboolean("randomEmbed")
    ensembleSize = int(config["MODEL_SETTINGS"].get("ensemble_size"))

    # if train argument was passed by user
    if args.train:
        # calling the train function for the model
        with open(trainPath, errors='ignore') as fp:
            sents = fp.readlines()
        data = np.array(list(map(lambda x: x.split(' ', 1), sents)))
        # segregating the questions and their corresponding labels
        questions, labels = data[:, 1], data[:, 0].tolist()
        # building the vocabulary for the text
        TEXT = CreateVocab()
        # creating the categorical encoder for the labels
        LABEL = LabelEncoder()
        # creating tokens for the questions
        quesTokens, sequenceLen = TEXT.createVocab(
            questions,stopWords = [], minFreq = 3, embeddedFile = embPath)

        dataset = CreateDataset(quesTokens, sequenceLen,
                                  LABEL.createLabels(labels))

        # save the vocabulary in a file
        with open(text_vocab_path, 'wb') as fp:
            pickle.dump(TEXT, fp)
        # save the label encodings in a file
        with open(label_vocab_path, 'wb') as fp:
            pickle.dump(LABEL, fp)

        VOCAB_SIZE = len(TEXT.idxToSent)
        NUM_CLASS = len(set(LABEL.idxToLabel))

        # if ensemble size is specified 
        if ensembleSize:
            # create a list of of classifiers using the configurations
            models = [Classifier(VOCAB_SIZE, embDim, NUM_CLASS, hidNodes, bow=bow, randomEmbed=randomEmbed,
                                     pre_emb=TEXT.idxToVocab if not randomEmbed else None, freeze=freezeTuning, lr=lr, gamma=gamma, device=device).to(device) for i in range(ensembleSize)]
            # pass the classifier list to the ensemble model
            model = Ensemble(models, device)

        # if ensemble size is NOT specified 
        else:
            # create the classifier model using the configurations
            model = Classifier(VOCAB_SIZE, embDim, NUM_CLASS, hidNodes, bow=bow, randomEmbed=randomEmbed,
                               pre_emb=TEXT.idxToVocab if not randomEmbed else None, freeze=freezeTuning, lr=lr, gamma=gamma, device=device).to(device)
        # call the train function to train the model created
        train()

    # if test argument was passed by user
    elif args.test:
        # load the vocabulary and label encoding files created during training
        with open(text_vocab_path, 'rb') as fp:
            TEXT = pickle.load(fp)
        with open(label_vocab_path, 'rb') as fp:
            LABEL = pickle.load(fp)

        VOCAB_SIZE = len(TEXT.idxToSent)
        NUM_CLASS = len(set(LABEL.idxToLabel))
        # if ensemble size is specified
        if ensembleSize:
            models = [Classifier(VOCAB_SIZE, embDim, NUM_CLASS, hidNodes, bow=bow, randomEmbed=randomEmbed,
                                     pre_emb=TEXT.idxToVocab if not randomEmbed else None, freeze=freezeTuning, lr=lr, gamma=gamma, device=device).to(device) for i in range(ensembleSize)]
            # load all the classifier models 
            for i, m in enumerate(models):
                m.load_state_dict(torch.load(modelPath+"."+str(i)))
                m.to(device)
            # pass the models to the Ensemble model
            model = Ensemble(models, device)

        # if ensemble size is NOT specified 
        else:
            model = Classifier(VOCAB_SIZE, embDim, NUM_CLASS, hidNodes, bow=bow, randomEmbed=randomEmbed,
                               pre_emb=TEXT.idxToVocab if not randomEmbed else None, freeze=freezeTuning, lr=lr, gamma=gamma, device=device).to(device)
            # load the classifier model to be tested
            model.load_state_dict(torch.load(modelPath))
            model.to(device)

        # call the test function to test the model loaded    
        test()
