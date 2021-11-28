This project demonstates the workflow of building recurrent neural network model to predict name entity lables using tensor flow. With a simple structure of one embedding layer, one biderectional LSTM layer and one dense layer, prediction accuracy on held-out test data set can achieve above .95 on CONLL-2003 data set. a Below is a brief introduction of the workflow (tokenization, padding, model buildup, hyper parameter tuning and test).

Dataset: 

The CONLL-2003 dataset were already seperated into train, validation and test sets. All three datasets contain sentences and name-entity tags corresponding to each word in the sentences.

Step 1: Build Corpus & Tags dictionary 

To convert words to numbers, we first read in the sentences and name entities in the training dataset, break sentences into lists of words (tokenization) and tags and create a dictionary “Corpus” to map every word to a number, plus an “UNK” token for unknown words and another dictionary “Tags” to store mapping of all name entity tags and the corresponding number.

Step 2: Turn sentenses to list of tokens and name entity tags 

To convert words and name entity tags into numbers, We need to first break sentences in to lists of words (or tokens).

Step 3: Map words and tags to number 

With a map of token-number pair and tag-number pairs, we can convert the sentences in the dataset into numbers for model build-up later.

Step 4: Padding 

To make all sentences in the dataset the same length to put build models, we add zeros to the end of each sentece (called padding) to make them all the same length of the longest sentence.

Step 5: Model Buildup 

I use tensorflow to build a sequence model containing an embedding layer, a bi-directional Long-Short Term Memory (LSTM) layer and a dense layer. The structures (layers in the model) and other hyperparameters (e.g. learning rate, activation function, dropout rate, regularization) are all hyper parameters which can be tunned to optimize evaluation matrics using the training and validation data set.

Step 6: Model training 

After defining model structure, we can feed the preprocessed (tokenized, mapped and padded) training data to train the model. Epochs (times the model runs through all the training data) and batch size (during an epoch, training data are processed on batch at a time untill the whole training set is processed) are to be adjusted to maximize performance.

Step 7: Validation 

Hyperparameters can be tuned to maximize the model performance on the validation set. The performance on the training and validation set both serve as reference for hyperparameter tuning(eg. bias-variance issue).

Step 8: Test the Model with Unseen Test Data 

The final model and hyperparameters should be tested on unseen (held-out) dataset. This is to avoid overfitting due to hyperparameter tuning. The testing accuracy reached above .95 using bidirectional LSTM on CONLL-2003 test data.
