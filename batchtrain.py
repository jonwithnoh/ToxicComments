def batch_1dcnn(gridsearch, X_train, y_train, X_test, y_test, train_epochs=40, train=True, X_holdout=None, y_holdout=None):
	from models import results_cnn_exists, train_1dcnn
	for embed in gridsearch['embed']:
		if embed == 'word2vec':
			from embeddingtools import tokenize_docs
			X_train_tokens = tokenize_docs(X_train)

			for w2v_dims in gridsearch['w2v_dims']:
				for w2v_epochs in gridsearch['w2v_epochs']:

					from gensim.models import Word2Vec
					print('Training W2V model with {} dims for {} epochs...'.format(w2v_dims, w2v_epochs))
					w2v_model = Word2Vec(X_train_tokens, size=w2v_dims, iter=w2v_epochs, min_count=1)
					print('Done.')

					for spatial_dropout in gridsearch['spatial_dropout']:
						for conv_filters in gridsearch['conv_filters']:
							for conv_kernel in gridsearch['conv_kernel']:
								for conv_layers in gridsearch['conv_layers']:
									for dense_units in gridsearch['dense_units']:
										for dense_dropout in gridsearch['dense_dropout']:
											config = {
												'embed': embed,
												'w2v_dims': w2v_dims,
												'w2v_epochs': w2v_epochs,
												'w2v_model': w2v_model,
												'spatial_dropout': spatial_dropout,
												'conv_filters': conv_filters,
												'conv_kernel': conv_kernel,
												'conv_layers': conv_layers,
												'dense_units': dense_units,
												'dense_dropout': dense_dropout,
												'use_gpu': gridsearch['use_gpu'],
												'SEQ_LEN': 200,
												'train_epochs': train_epochs,
												'train': train,
											}
											if results_cnn_exists(config):
												pass
											else:
												train_1dcnn(config, X_train, y_train, X_test, y_test, 
													X_holdout=X_holdout, y_holdout=y_holdout)
		else:

			for spatial_dropout in gridsearch['spatial_dropout']:
				for conv_filters in gridsearch['conv_filters']:
					for conv_kernel in gridsearch['conv_kernel']:
						for conv_layers in gridsearch['conv_layers']:
							for dense_units in gridsearch['dense_units']:
								for dense_dropout in gridsearch['dense_dropout']:
									config = {
										'embed': embed,
										'spatial_dropout': spatial_dropout,
										'conv_filters': conv_filters,
										'conv_kernel': conv_kernel,
										'conv_layers': conv_layers,
										'dense_units': dense_units,
										'dense_dropout': dense_dropout,
										'use_gpu': gridsearch['use_gpu'],
										'SEQ_LEN': 200,
										'train_epochs': train_epochs,
										'train': train,
									}
									if not results_cnn_exists(config):
										train_1dcnn(config, X_train, y_train, X_test, y_test,
											X_holdout=X_holdout, y_holdout=y_holdout)

def batch_rnn(gridsearch, X_train, y_train, X_test, y_test, train_epochs=40, train=True, X_holdout=None, y_holdout=None):
	from models import results_rnn_exists, train_rnn
	for embed in gridsearch['embed']:
		if embed == 'word2vec':
			from embeddingtools import tokenize_docs
			X_train_tokens = tokenize_docs(X_train)

			for w2v_dims in gridsearch['w2v_dims']:
				for w2v_epochs in gridsearch['w2v_epochs']:

					from gensim.models import Word2Vec
					print('Training W2V model with {} dims for {} epochs...'.format(w2v_dims, w2v_epochs))
					w2v_model = Word2Vec(X_train_tokens, size=w2v_dims, iter=w2v_epochs, min_count=1)
					print('Done.')

					for spatial_dropout in gridsearch['spatial_dropout']:
						for rnn_type in gridsearch['rnn_type']:
							for rnn_units in gridsearch['rnn_units']:
								for bidirectional in gridsearch['bidirectional']:
									for dense_units in gridsearch['dense_units']:
										for dense_dropout in gridsearch['dense_dropout']:
											config = {
												'embed': embed,
												'w2v_dims': w2v_dims,
												'w2v_epochs': w2v_epochs,
												'w2v_model': w2v_model,
												'spatial_dropout': spatial_dropout,
												'rnn_type': rnn_type,
												'rnn_units': rnn_units,
												'bidirectional': bidirectional,
												'dense_units': dense_units,
												'dense_dropout': dense_dropout,
												'use_gpu': gridsearch['use_gpu'],
												'SEQ_LEN': 200,
												'train_epochs': train_epochs,
												'train': train
											}
											if results_rnn_exists(config):
												pass
											else:
												train_rnn(config, X_train, y_train, X_test, y_test,
													X_holdout=X_holdout, y_holdout=y_holdout)
		else:
			for glove_dim in gridsearch['glove_dim']:
				for spatial_dropout in gridsearch['spatial_dropout']:
					for rnn_type in gridsearch['rnn_type']:
						for rnn_units in gridsearch['rnn_units']:
							for bidirectional in gridsearch['bidirectional']:
								for dense_units in gridsearch['dense_units']:
									for dense_dropout in gridsearch['dense_dropout']:
										config = {
											'embed': embed,
											'glove_dim': glove_dim,
											'spatial_dropout': spatial_dropout,
											'rnn_type': rnn_type,
											'rnn_units': rnn_units,
											'bidirectional': bidirectional,
											'dense_units': dense_units,
											'dense_dropout': dense_dropout,
											'use_gpu': gridsearch['use_gpu'],
											'SEQ_LEN': 200,
											'train_epochs': train_epochs,
											'train': train
										}
										if not results_rnn_exists(config):
											train_rnn(config, X_train, y_train, X_test, y_test,
												X_holdout=X_holdout, y_holdout=y_holdout)