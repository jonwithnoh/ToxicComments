# ToxicComments
Analytics work on Kaggle Toxic Comments Analysis

Bokeh EDA visualization can be run with 'bokeh serve --show bokeh_interactive.py'

tca.py contains data preparation (text preparation, tokenization, vectorization) and EDA tools
models.py contains Keras model definitions (RNN, CNN, ELMo). Unfortunately, ELMo performed poorly relative to RNN/CNN. Currently working on preparing doc2vec representations as secondary input. Best performing model is a 128 unit GRU with GLoVe embeddings and 64 dense units with 15% dropout. This model was trained on 70% of the training data available from Kaggle and tested on the remaining 30% (split into test and holdout sets). With this scenario, the model achieved 98.4% mean accuracy, putting it in roughly the top third of competitors on Kaggle. Once doc2vec parameter tuning is complete the model will be retrained on the full training data with the doc2vec representation as an additional input.
