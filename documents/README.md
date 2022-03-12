# Documentation for all the functions and classes used

The following file contains the detailed descriptions for all the python modules being used, and the classes and functions they contain.


### 1) src/model_builder.py
The following python file contains the required modules for creating the Torch DataLoaders by using the Torch Dataset class, create a neural network model according to the provided specifications and ensembles several models together.

#### 1.1) Classifier
This class helps in defining the architecture of the model given the specifications and also defines the other functionalities required by a neural network model.

`__init__() -` Helps in initializing the model parameters and also construct the general architecture of the model based on the specifications. Takes into account the type of word embeddings to use(random/pre-trained), the type of sentence representations to use(BOW/BiLSTM) and whether or not to freeze/fine-tune model.

`forward() –` This function accepts a single tuple of data and passes it through the architecture of the neural network and returns the output of the final layer.

`train() -` This function accepts the batch-level data during training, computes all the losses and accuracies corresponding to one training-batch step and then performs backpropagation to update the required weights given the optimiser selected by us. It returns the loss and accuracies for the batch of data provided to it.

`test() –` This function computes and returns the validation accuracies and losses given the required batch of data.

`predict() -` This function helps in predicting the classes/labels of the given input sentences.

`fit() –` This function helps in controlling the entire training routine of the neural network. It runs for the required number of epochs, feeds the data into train() function. It also saves the best performing model based on the validation accuracy.

#### 1.2) CreateDataset
This class helps in creating and formatting the dataset in the required format. It classifies the question texts, answer labels and the lengths(limits) of each question text.

`__len\__() –` Helper function to return the number of observations in the sample.

`__getitem__() –` Helper function to return one observation of the dataset.

`1.3) Ensemble –` This class is used to create a combination of various models and make predictions based on it.

`__init__() –` Accepts the list of models which are to be a part of the ensemble.

`fit() –` This function  helps to fit all the different models together to the training data and thus create the combined ensemble model.

`predict() –` Makes class/label predictions for provided sentences by combining and bagging the results of the given list of models.


### 2) src/preprocess_prepare.py 
It consists of functions to perform pre-processing on the textual part of the data.

#### 2.1) CreateVocab()
This class contains modules to clean, tokenize, embed data and build vocabulary given a set of sentences.

`cleanText() –` Given a sentence, it converts it into lowercase, removes all the punctuations and returns the tokenized version of the sentence.

`getNgrams() –` Given a word, it returns the possible n-grams based on the word.

`getVector() –` Given a word has n-grams it returns the average of the n-grams to calculate the embedding of that particular word.

`createWordEmbedding() –` This function takes care of the overall embedding process. First of all it checks whether the token to be encoded is present in the provided embedding file, otherwise it finds the possible n-grams for the token and calculates the average to find its embedding. The last resort is to assign the token the embedding for unknown token, which is #UNK#.

`vocabIterator() –` The function responsible for creating the vocabulary. It builds a vocabulary according to the tokens being considered. It also helps in removing stopwords and removing the words which fail to satisfy the minimum occurrence in the dataset.

`createVocab() -` Function handling the overall vocabulary creation process. Creates the tokens from the list of sentences, contains the stopwords list, helps in loading the pre-trained embeddings if they are being used.

`sentenceEncoder() –` This function returns the encoded tokens given a sequence of input sentences.

`sentenceDecoder() –` This function returns the original sentence in list of words format given the encoded format.

#### 2.2) LabelEncoder()
This class contains functions which will take categorical data(labels of the questions) and encode them such that each label is denoted by a unique value.

`createLabels() –` This function returns the list of unique tokens present in a collection.

`labelEncoder() –` It returns the encoded form of a given label.

`labelDecoder() –` It returns the original label given a encoded label.

### 3) src/question_classifier.py
This is the main python script, which contains the main() function which helps in the overall training and testing of the different models. It loads all the functions defined in the classes above and call them accordingly to facilitate the smooth working of the designed pipeline.

`train() –` This function when called helps in training our model. This is the function where train-validation split occurs. We use torch in-built functions to design the required generators which perform the splitting and thus finally call the fit() function from _src/classifiers.py_ to allow our model to train on the data calling all the other required functions within itself. Finally, we get the maximum validation accuracy at the end.

`test() –` This function helps us to analyze our fitted model on the test set. First of all the sentences from the test-set are encoded and then the predict() function from _src/classifiers.py_ is called to get the outcomes. Finally, the performance of the model is measured in terms of accuracy, precision and recall.

`main() –` This is the segment of the code which handles the overall working of the segments. It initially parses the command line arguments and accordingly decides whether to train the model or test it. It also initializes all the parameters required for the model by parsing through the required configuration file. If train() is called then data is loaded then vocabulary is created, labels are encoded, data is converted into required format and finally selected model is fitted. If test() is called then saved model file is loaded (assuming training for the same model has already been performed), and then testing is done on the test set given the trained model weights and parameters.