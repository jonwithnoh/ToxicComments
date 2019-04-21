# ToxicComments
Analytics work on Kaggle Toxic Comments Analysis

Bokeh EDA visualization can be run with 'bokeh serve --show bokeh_interactive.py'

tca.py contains data preparation (text preparation, tokenization, vectorization) and EDA tools. Note that the vectorize_text() function in the ToxicCommentsAnalysis object is not fully functional as support for doc2vec has been added and is not working quite right yet.

models.py contains Keras model (RNN, CNN, ELMo) building and training functions. 
embeddingtools.py contains tools to develop embeddings of various flavors (Word2Vec custom trained on the corpus, GLoVe of user-specified dimensionality, or spaCy's vectors).
batchtrain.py contains the grid search mechanisms. These models were trained on Google Colab notebooks to leverage GPU processing power.

Models are built using the build_1dcnn and build_rnn functions in models.py. Model architecture is passed as a dict, however unspecified settings will not cause the process to fail, as there are defaults contained within the function. The train_rnn and train_cnn functions will prepare data, initialize callbacks, fit the model, and record results to json in order to perform an effective parameter grid search.

Grid search is performed by passing a dict containing keys for the model parameters with the lists of possible values. 

Unfortunately, ELMo performed poorly relative to RNN/CNN. Currently working on preparing doc2vec representations as secondary input. Best performing model is a 128 unit GRU with GLoVe embeddings and 64 dense units with 15% dropout. This model was trained on 70% of the training data available from Kaggle and tested on the remaining 30% (split into test and holdout sets). With this scenario, the model achieved 98.4% mean accuracy, putting it in roughly the top third of competitors on Kaggle. Once doc2vec parameter tuning is complete the model will be retrained on the full training data with the doc2vec representation as an additional input.
