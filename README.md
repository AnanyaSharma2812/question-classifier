# Question Classifier

## Description

This repository is designed to perform question classification in which for a given question, the classifier will try to predict an output label that represents the type of answer expected.

This repository contains python scripts, the required data files, and the configuration files to aid with the training and testing of the classifier model.

## Contents

This repository comprises of three folders.

1. **data –** It contains the train dataset, the test dataset, the vocabulary built during the training process (for review purposes), and the configuration files which store the hyperparameter values and file paths required for the overall working of the various model combinations being used. It also contains the output text files, where the performance of the model during training and testing results are stored
2. **documents –** It contains the research paper written in accordance with this project along with a file that describes the purpose of each, and every function and class used in this project
3. **src –** It contains all the python scripts which are required to train and test the models.

**question_classifier.py -** It is the main python file that will be used to train and test the various combinations of models and analyse their performance

## Requirements

- To successfully run the question_classifiers.py, some basic libraries need to be included and installed before going forward.

To do that, from the command line at the current file location run the following command pip command :

```bash
$ pip install src/requirements.txt
```

(This should work regardless of the type of OS being used)

The above command will install all the required libraries used for this project.

- Also, for some of the models used in this project which use pre-trained word embeddings, it is required to add the embedding files (like Glove or Word2Vec) in the **models** folder inside the **data** folder, before the model is trained.

## Usage

In this section, a quick description will be given on how to run the code for both training and testing.

From the command line inside the **src** folder, the following two commands can be executed for training and testing the model respectively:

1. For training:
```bash
$ python question_classifier.py --train --config [configuration_file_path]
```

2. For testing:
```bash
$ python question_classifier.py --test --config ../[configuration_file_path]
```

### Description of Parameters being used:

- train : used to train and save the model
- test : used to load and test the model
- config : used to select the configuration file to be used (which in turn decides the type of model and its required hyperparameters) for training/testing

### Possible Model Configurations:

In place of "[configuration_file_path]" use one the following model configuration, all of which can be found under [/data/models/](data/models/):

- [bilstm_glove_freeze.ini](data/models/bilstm_glove_freeze.ini)
- [bilstm_random_finetune.ini](data/models/bilstm_random_finetune.ini)
- [bilstm_random_freeze.ini](data/models/bilstm_random_freeze.ini)
- [bow_glove_finetune.ini](data/models/bow_glove_finetune.ini)
- [bow_glove_freeze.ini](data/models/bow_glove_freeze.ini)
- [bow_random_finetune.ini](data/models/bow_random_finetune.ini)
- [bow_random_freeze.ini](data/models/bow_random_freeze.ini)
- [ensemble_bilstm_glove_finetune.ini](data/models/ensemble_bilstm_glove_finetune.ini)

### Example usage:

If we want to use the Bag of Words (BOW) model with pre-trained Glove embeddings and with fine-tuning, we need to execute the following commands for training and then testing the model:

```bash
$ python question_classifier.py --train --config data/models/bow_glove_finetune.ini
```

```bash
$ python question_classifier.py --test --config data/models/bow_glove_finetune.ini
```

**Note**  : Before running the test command for a particular model specification, ensure that the training command for the same configuration has already been executed.

## Documentation
Further documentation on how the code works can be found [here](document/README.md)