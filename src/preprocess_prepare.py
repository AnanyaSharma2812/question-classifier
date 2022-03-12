"""
This module consists of the code required for preprocessing the dataset to create the vocabulary,
and create encoding for our output labels.
"""
from collections import Counter
import numpy as np
import torch
import torch.nn.utils as tut

'''  class for creating vocabulary from the question dataset '''
class CreateVocab():

    # method to initialize the class variables
    def __init__(self):

        self.sentToIdx = dict()
        self.idxToSent = []
        self.embedding = dict()
        self.idxToVocab = []
        self.tokens = []
        # setting a placeholder for unknown words that are not present in the vocabulary
        self.unknown = "#UNK#"
        # setting a placeholder for padding the sentences
        self.padding = "#PAD#"


    # method to clean the sentences in the dataset
    def cleanText(self, line):

        # convert the entire sentence into lower case
        line = line.lower()
        puncts = [',','(','?','"',';',':','`','-','_','|','!','@','#','$','%','^','&','*','<','>','+',')']
        # checking for punctuations in each sentence and removing them
        for punct in puncts:
            line = line.replace(punct,"")

        return line.split()


    # method to create n-grams for each word
    def getNgrams(self, word, n = 3):

        return [word[i:i+n] for i in range(len(word)-n+1)]


    # method to create the word vectors
    def getVector(self, word, n = 3):

        vectors = []

        if len(word) <= 3:
            ngrams = self.getNgrams(word, len(word)-1)
        else:
            ngrams = self.getNgrams(word, n)

        vectors = np.array(
            list(filter(lambda x: x != None, map(self.embedding.get, ngrams))))

        if vectors.shape[0]:
            return np.average(vectors, axis=0).tolist()
        else:
            return self.embedding.get(self.unknown)


    # method to create the word embeddings
    def createWordEmbedding(self, token):

        vector = self.embedding.get(token)

        if vector:
            return vector
        else:
            if len(token) > 1:
                return self.getVector(token)
            else:
                return self.embedding[self.unknown]


    # method to create an iterator for the vocabulary
    def vocabIterator(self, iterator, stopWords = [], minFreq = 1, embedding = False):

        counter = Counter()
        for i in iterator:
            counter.update(i)
        descRows = sorted(counter.items(), key = lambda x: x[1], reverse = True)

        if embedding:
            for token, f in descRows:
                if f >= minFreq and token not in stopWords:
                    self.idxToSent.append(token)
                    self.idxToVocab.append(self.createWordEmbedding(token))
        else:
            self.idxToSent.extend(
                [token for token, f in descRows
                 if f >= minFreq and token not in stopWords])

        if self.unknown not in self.idxToSent:
            self.idxToSent.insert(0, self.unknown)
            self.idxToVocab.insert(0, self.embedding.get(self.unknown))
            self.idxToSent.insert(0, self.padding)
            self.idxToVocab.insert(0, [0.0]*300)
        self.sentToIdx.update(zip(self.idxToSent, range(len(self.idxToSent))))


    # method to create the vocabulary
    def createVocab(self, sentences, stopWords = [], minFreq = 1,embeddedFile = None, unknown = "#UNK#", padding = "#PAD#"):
        
        # creating a list of stopwords
        stopWords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
                      "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
                      "she", "her", "hers", "herself", "it", "its", "itself", "they", "them",
                      "their", "theirs", "themselves", "this", "that", "these", "those",
                      "a", "an", "the", "and", "but", "if", "or", "because", "as", "until",
                      "while", "of", "at", "by", "for", "with", "about", "against",
                      "between", "into", "through", "during", "before", "after", "above",
                      "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
                      "under", "again", "further", "then", "once", "here", "there", "all", "any",
                      "both", "each", "few", "more", "most", "other", "some", "such", "no",
                      "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
                      "just", "don", "now"]
        
        
        self.unknown = unknown
        self.padding = padding
        senTokens = list(map(self.cleanText, sentences))

        # if a pre-embedded file is given to be used
        if embeddedFile:
            with open(embeddedFile) as fp:
                for line in fp.readlines():
                    values = line.split()
                    self.embedding[values[0]] = list(map(float, values[1:]))
            self.vocabIterator(senTokens, stopWords, minFreq, True)
        # if no pre-embedded file is given
        else:
            self.vocabIterator(senTokens, stopWords, minFreq, False)

        tokens = [torch.tensor(list(filter(lambda x: x != None, map(self.sentToIdx.get, sTokens)))) for sTokens in senTokens]
        limits = torch.LongTensor(list(map(len, tokens)))
        tokens = tut.rnn.pad_sequence(tokens, batch_first=True, padding_value=self.sentToIdx[self.padding])

        return tokens, limits


    # method to encode sentences
    def sentenceEncoder(self, sentences):
        
        encoded = []

        # for each question in the list of questions
        for sentence in sentences:
            # tokenizing the sentence
            tokenized = self.cleanText(sentence) 
            indexed = [self.sentToIdx.get(token) if self.sentToIdx.get(token) else self.sentToIdx.get(self.unknown) for token in tokenized]
            encoded.append(indexed)

        return encoded


    # method to decode the encoded sentence tokens
    def sentenceDecoder(self, encodings):
    
        return [' '.join(map(lambda x: self.idxToSent[x], encoding)) for encoding in encodings]


'''  class for creating labels for categorical output data '''
class LabelEncoder():

    # method to initialize the class variables
    def __init__(self):

        self.idxToLabel = None
        self.labelToIdx = None

    # method to create the labels 
    def createLabels(self, labels):

        self.idxToLabel = list(set(labels))
        self.labelToIdx = dict(zip(self.idxToLabel, range(len(self.idxToLabel))))

        return torch.tensor(list(map(self.labelToIdx.get, labels)))

    # method to create encoded tokens for the labels 
    def labelEncoder(self, labels):
        
        encoded = list(map(self.labelToIdx.get, labels))

        return encoded

    # method to decode the encoded labels
    def labelDecoder(self, encodings):

        return list(map(lambda x: self.idxToLabel[x], encodings))
