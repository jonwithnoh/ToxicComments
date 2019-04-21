from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPool1D, LSTM, Dropout, Flatten
from keras.layers import SpatialDropout1D, CuDNNLSTM, CuDNNGRU, GRU
from keras.layers import concatenate
from keras.optimizers import Adam
from tca import pkl_load, pkl_dump
import os
import glob
from keras import backend as K

MODELS_DIR = 'models'
target_flags = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def rocauc(y_true, y_pred):
	from sklearn.metrics import roc_auc_score
	return roc_auc_score(y_true, y_pred.ravel())

def f1(y_true, y_pred):
	from sklearn.metrics import f1_score
	return f1_score(y_true, y_pred.ravel().round(0))


def embedding_layer_w2v(w2v_model, SEQ_LEN, tokenizer='tokenizer.pickle'):
	from embeddingtools import load_gensim_wordvecs, gensim_embedding_matrix, make_embedding_layer
	tk = pkl_load(tokenizer)
	#wv = load_gensim_wordvecs(w2vfile)
	print('Preparing embedding matrix. {} word vectors, {} token ids'.format(len(w2v_model.wv.vocab), len(tk.word_index)))
	embedding_matrix = gensim_embedding_matrix(w2v_model, tk.word_index)
	return make_embedding_layer(embedding_matrix, SEQ_LEN)

def embedding_layer_glove(SEQ_LEN, glove_dim=100,tokenizer='tokenizer.pickle'):
	from embeddingtools import spacy_embedding_matrix, make_embedding_layer, glove_embedding_matrix
	#tk = pkl_load(tokenizer)
	#embedding_matrix = spacy_embedding_matrix(tk.word_index)
	embedding_matrix = glove_embedding_matrix(dims=glove_dim)
	return make_embedding_layer(embedding_matrix, SEQ_LEN)


def build_1dcnn(config={}):
	cfgdefaults = {
		'embed': ['glove', 'Embedding'],
		'glove_dim': [100, 'GLoVe dims'],
		'spatial_dropout': [0, 'Spatial dropout'],
		'use_gpu': [False, 'GPU support'],
		'conv_filters': [64, 'Conv. filters'],
		'conv_kernel': [5, 'Conv. kernel'],
		'conv_layers': [1, 'Conv. layers'],
		'dense_units': [64, 'Dense units'],
		'dense_dropout': [0, 'Dense dropout'],
		'adam_lr': [.0001, 'Adam optimizer learning rate'],
		'adam_lr_decay': [.000001, 'Adam optimizer LR decay'],
		'include_d2v': [False, 'Additional Doc2Vec representation'],
		'd2v_dim': [200, 'Doc2Vec dimensionality'],
		'd2v_dense_nodes': [64, 'Doc2Vec MLP nodes'],
		'train_epochs': [1, 'Training epochs'],
		'SEQ_LEN': [200, 'Sequence length'],
		'train': [True, 'Model training'],
		'target_field_names': [
			target_flags,
			'Target flag field names']
	}
	if type(config) is dict:
		for cfgkey in cfgdefaults.keys():
			if cfgkey not in config.keys():
				var = cfgdefaults.get(cfgkey)
				config[cfgkey] = var[0]
				print("{} not specified in config, defaulting to {}".format(var[1], var[0]))

	if config['embed'] == 'word2vec':
		from gensim.models import Word2Vec
		if not isinstance(config['w2v_model'], Word2Vec):
			print('Word2Vec model must be passed to config key w2vmodel')
			return

	from embeddingtools import make_embedding_layer

	seq_input = Input(shape=(config['SEQ_LEN'],), dtype='int32')
	if config['embed'] == 'word2vec':
		embedding = embedding_layer_w2v(config['w2v_model'], config['SEQ_LEN'])(seq_input)

	if config['embed'] == 'glove':
		embedding = embedding_layer_glove(config['SEQ_LEN'], glove_dim=config['glove_dim'])(seq_input)


	spatial_dropout = SpatialDropout1D(config['spatial_dropout'])(embedding)
	#spatial_drop = embedding

	for i in range(config['conv_layers']):
		if i == 0:
			conv = Conv1D(config['conv_filters'], 
				kernel_size=config['conv_kernel'], 
				activation='relu')(spatial_dropout)

			pool = MaxPool1D(pool_size=2)(conv)

		else:
			conv = Conv1D(config['conv_filters'], 
				kernel_size=config['conv_kernel'], 
				activation='relu')(pool)

			pool = MaxPool1D(pool_size=2)(conv)

	flatten = Flatten()(pool)



	inputs = [seq_input]

	if config['d2v_include']:
		d2v_input = Input(shape=(config['d2v_dim'],), name='D2VInput')
		d2v_dense = Dense(config['d2v_dense_nodes'], activation='relu', name='D2VDense')(d2v_input)

		flatten = concatenate([flatten, d2v_dense])

		inputs.append(d2v_input)

	dense = Dense(config['dense_units'], activation='relu')(flatten)
	dense_dropout = Dropout(config['dense_dropout'])(dense)

	output_nodes = [] # create output node for each flag
	for i in range(len(config['target_field_names'])):
		output_nodes.append(
			Dense(1, activation='sigmoid', name=config['target_field_names'][i])(dense_dropout)
			)

	adam = Adam(lr=config['adam_lr'], decay=config['adam_lr_decay'])
	model = Model(inputs=inputs, outputs=output_nodes)
	model.compile(loss='binary_crossentropy',
		optimizer=adam, 
		metrics=['accuracy'])
	return model, config

def build_rnn(config={}):

	cfgdefaults = {
		'embed': ['glove', 'Embedding'],
		'glove_dim': [100, 'GLoVe dims'],
		'spatial_dropout': [0, 'Spatial dropout'],
		'use_gpu': [False, 'GPU support'],
		'rnn_type': ['gru', 'RNN type'],
		'rnn_units': [32, 'RNN units'],
		'bidirectional': [False, 'Bi-directional RNN'],
		'dense_units': [64, 'Dense units'],
		'dense_dropout': [0, 'Dense dropout'],
		'd2v_include': [False, 'Additional Doc2Vec representation'],
		'd2v_dim': [200, 'Doc2Vec dimensionality'],
		'd2v_dense_nodes': [64, 'Doc2Vec MLP nodes'],
		'adam_lr': [.0001, 'Adam optimizer learning rate'],
		'adam_lr_decay': [.000001, 'Adam optimizer LR decay'],
		'SEQ_LEN': [200, 'Sequence length'],
		'train_epochs': [1, 'Training epochs'],
		'train': [True, 'Model training'],
		'target_field_names': [
			target_flags,
			'Target flag field names']
	}

	if type(config) is dict:
		for cfgkey in cfgdefaults.keys():
			if cfgkey not in config.keys():
				var = cfgdefaults.get(cfgkey)
				config[cfgkey] = var[0]
				print("{} not specified in config, defaulting to {}".format(var[1], var[0]))
	else:
		print('Must pass config as dict or leave empty.')
		return

	if config['embed'] == 'word2vec':
		from gensim.models import Word2Vec
		if not isinstance(config['w2v_model'], Word2Vec):
			print('Word2Vec model must be passed to config key w2vmodel')
			return

	from embeddingtools import make_embedding_layer

	seq_input = Input(shape=(config['SEQ_LEN'],), dtype='int32', name='seq_input')
	if config['embed'] == 'word2vec':
		embedding = embedding_layer_w2v(config['w2v_model'], config['SEQ_LEN'])(seq_input)

	if config['embed'] == 'glove':
		embedding = embedding_layer_glove(config['SEQ_LEN'], glove_dim=config['glove_dim'])(seq_input)

	spatial_dropout = SpatialDropout1D(config['spatial_dropout'])(embedding)

	if config['bidirectional']:
		from keras.layers import Bidirectional 

		if config['rnn_type'] == 'lstm':
			if config['use_gpu']:

				rnn = Bidirectional(CuDNNLSTM(config['rnn_units']))(spatial_dropout)
			else:

				rnn = Bidirectional(LSTM(config['rnn_units']))(spatial_dropout)

		elif config['rnn_type'] == 'gru':
			if config['use_gpu']:

				rnn = Bidirectional(CuDNNGRU(config['rnn_units']))(spatial_dropout)
			else:

				rnn = Bidirectional(GRU(config['rnn_units']))(spatial_dropout)
	else:
		if config['rnn_type'] == 'lstm':
			if config['use_gpu']:
				rnn = CuDNNLSTM(config['rnn_units'])(spatial_dropout)
			else:
				rnn = LSTM(config['rnn_units'])(spatial_dropout)
		elif config['rnn_type'] == 'gru':
			if config['use_gpu']:
				rnn = CuDNNGRU(config['rnn_units'])(spatial_dropout)
			else:
				rnn = GRU(config['rnn_units'])(spatial_dropout)

	inputs = [seq_input]

	if config['d2v_include']:
		d2v_input = Input(shape=(config['d2v_dim'],), name='D2VInput')
		d2v_dense = Dense(config['d2v_dense_nodes'], activation='relu', name='D2VDense')(d2v_input)

		rnn = concatenate([rnn, d2v_dense])

		inputs.append(d2v_input)


	dense = Dense(config['dense_units'], activation='relu')(rnn)
	dense_dropout = Dropout(config['dense_dropout'])(dense)

	output_nodes = []
	for i in range(len(config['target_field_names'])):
		output_nodes.append(
			Dense(1, activation='sigmoid', name=config['target_field_names'][i])(dense_dropout)
			)

	adam = Adam(lr=config['adam_lr'], decay=config['adam_lr_decay'])
	model = Model(inputs=inputs, outputs=output_nodes)
	model.compile(loss='binary_crossentropy',
		optimizer=adam, 
		metrics=['accuracy'])
	return model, config

def build_elmo_dnn(config={}):
	cfgdefaults = {
		'dense_units': [64, 'Dense units'],
		'dense_dropout': [0, 'Dense dropout'],
		'adam_lr': [.001, 'Adam optimizer learning rate'],
		'adam_lr_decay': [0, 'Adam optimizer LR decay'],
		'target_field_names':  [
			target_flags,
			'Target flag field names'],
		'SEQ_LEN': [200, 'Sequence length'],
	}

	if type(config) is dict:
		for cfgkey in cfgdefaults.keys():
			if cfgkey not in config.keys():
				var = cfgdefaults.get(cfgkey)
				config[cfgkey] = var[0]
				print("{} not specified in config, defaulting to {}".format(var[1], var[0]))
	
	from embeddingtools import ElmoEmbeddingLayer
	inputs = Input(shape=(1,), dtype='string')
	embedding = ElmoEmbeddingLayer()(inputs)
	dense = Dense(config['dense_units'])(embedding)
	dense_dropout = Dropout(config['dense_dropout'])(dense)

	
	output_nodes = []
	for i in range(len(config['target_field_names'])):
		output_nodes.append(
			Dense(1, activation='sigmoid', name=config['target_field_names'][i])(dense_dropout)
			)

	adam = Adam(lr=config['adam_lr'], decay=config['adam_lr_decay'])
	model = Model(inputs=[inputs], outputs=output_nodes)
	model.compile(loss='binary_crossentropy',
		optimizer=adam, 
		metrics=['accuracy'])
	return model, config


def load_json(file):
	import json
	with open(file, 'r') as f:
		js = json.load(f)
	return js

def dump_json(obj, file):
	import json
	with open(file, 'w') as f:
		json.dump(obj, f, indent=4)

def get_json_results(file):
	import glob
	files = glob.glob(file)
	if len(files) == 0:
		return None
	else:
		results = load_json(files[0])
		return results

def store_results_cnn(history, config, file=os.path.join(MODELS_DIR, 'results_cnn.json')): #'{}/results_cnn.json'.format(MODELS_DIR)
	if get_json_results(file) is None:
		results = {}
	else:
		results = get_json_results(file)

	
	if config['embed'] == 'word2vec':
		resultskey = '{}_dim{}_e{}_sd{}_cf{}_ck{}_cl{}_du{}_dd{}'.format(
			config['embed'],
			config['w2v_dims'],
			config['w2v_epochs'],
			config['spatial_dropout'],
			config['conv_filters'],
			config['conv_kernel'],
			config['conv_layers'],
			config['dense_units'],
			config['dense_dropout']
			)
		results[resultskey] = history

	elif config['embed'] == 'glove':
		resultskey = '{}_sd{}_cf{}_ck{}_cl{}_du{}_dd{}'.format(
			config['embed'],
			config['spatial_dropout'],
			config['conv_filters'],
			config['conv_kernel'],
			config['conv_layers'],
			config['dense_units'],
			config['dense_dropout']
			)
		results[resultskey] = history

	dump_json(results, file)
	

def store_results_rnn(history, config, file=os.path.join(MODELS_DIR, 'results_rnn.json')):
	if get_json_results(file) is None:
		results = {}
	else:
		results = get_json_results(file)

	if config['embed'] == 'word2vec':
		resultskey = '{}_dim{}_e{}_sd{}_rt{}_ru{}_bi{}_du{}_dd{}'.format(
			config['embed'],
			config['w2v_dims'],
			config['w2v_epochs'],
			config['spatial_dropout'],
			config['rnn_type'],
			config['rnn_units'],
			config['bidirectional'],
			config['dense_units'],
			config['dense_dropout']
			)
		results[resultskey] = history

	elif config['embed'] == 'glove':
		resultskey = '{}_sd{}_rt{}_ru{}_bi{}_du{}_dd{}'.format(
			config['embed'],
			config['spatial_dropout'],
			config['rnn_type'],
			config['rnn_units'],
			config['bidirectional'],
			config['dense_units'],
			config['dense_dropout']
			)
		results[resultskey] = history
		
	dump_json(results, file)


	

def store_results_elmo(history, config, file=os.path.join(MODELS_DIR, 'results_elmo.json')):
	if get_json_results(file) is None:
		results = {}
	else:
		results = get_json_results(file)

	try:
		results[config['dense_units']]
	except:
		results[config['dense_units']] = {}
	
	results[config['dense_units']][config['dense_dropout']] = history
	dump_json(results, file)

def results_cnn_exists(config, file=os.path.join(MODELS_DIR, 'results_cnn.json')):
	exists = False
	try:
		results = load_json(file)
		exists = True
	except:
		exists = False

	if exists:

		if config['embed'] == 'word2vec':
			resultskey = '{}_dim{}_e{}_sd{}_cf{}_ck{}_cl{}_du{}_dd{}'.format(
				config['embed'],
				config['w2v_dims'],
				config['w2v_epochs'],
				config['spatial_dropout'],
				config['conv_filters'],
				config['conv_kernel'],
				config['conv_layers'],
				config['dense_units'],
				config['dense_dropout']
				)
			try:
				results[resultskey]
				exists = True
				print('Training results for configuration found, skipping.')
			except:
				exists = False

		elif config['embed'] == 'glove':
			resultskey = '{}_sd{}_cf{}_ck{}_cl{}_du{}_dd{}'.format(
				config['embed'],
				config['spatial_dropout'],
				config['conv_filters'],
				config['conv_kernel'],
				config['conv_layers'],
				config['dense_units'],
				config['dense_dropout']
				)
			try:
				results[resultskey]
				exists = True
				print('Training results for configuration found, skipping.')
			except:
				exists = False

	return exists

	
	

def results_rnn_exists(config, file=os.path.join(MODELS_DIR, 'results_rnn.json')):
	exists = False
	try:
		results = load_json(file)

		exists = True
	except:
		exists = False
		print('Results file does not exist.')

	if exists:
		if config['embed'] == 'word2vec':
			resultskey = '{}_dim{}_e{}_sd{}_rt{}_ru{}_bi{}_du{}_dd{}'.format(
				config['embed'],
				config['w2v_dims'],
				config['w2v_epochs'],
				config['spatial_dropout'],
				config['rnn_type'],
				config['rnn_units'],
				config['bidirectional'],
				config['dense_units'],
				config['dense_dropout']
			)
			try:
				results[resultskey]
				exists = True
				print('Training results for configuration found, skipping.')
			except:
				exists = False

		elif config['embed'] == 'glove':
			resultskey = '{}_sd{}_rt{}_ru{}_bi{}_du{}_dd{}'.format(
				config['embed'],
				config['spatial_dropout'],
				config['rnn_type'],
				config['rnn_units'],
				config['bidirectional'],
				config['dense_units'],
				config['dense_dropout']
			)
			try:
				results[resultskey]
				exists = True
				print('Training results for configuration found, skipping.')
			except:
				exists = False

	return exists

	

def results_elmo_exists(config, file=os.path.join(MODELS_DIR, 'results_elmo.json')):
	exists = False
	try:
		results = load_json(file)
		exists = True
	except:
		exists = False
	
	if exists:
		try:
			results[config['dense_units']][config['dense_dropout']]
			print('Training results for configuration found, skipping.')
			exists = True
		except:
			exists = False

	return exists

def verify_inputs(X_train, X_test, X_holdout=None):
	if X_holdout is None:
		try:
			assert isinstance(X_train, list) & isinstance(X_test_seq, list)
		except AssertionError as e:
			e.args += ("All inputs must be of format 'list'",)
			raise
		try:
			assert len(X_train) == len(X_test)
		except:
			e.args += ('Input lists must be of equal length',)
			raise
	else:
		try:
			assert isinstance(X_train, list) & isinstance(X_test, list) & isinstance(X_holdout, list)
		except AssertionError as e:
			e.args += ("All inputs must be of format 'list'",)
			raise
		try:
			assert len(X_train) == len(X_test) == len(X_holdout)
		except:
			e.args += ('Input lists must be of equal length',)
	return True

def train_1dcnn(config, X_train, y_train, X_test, y_test, models_dir=MODELS_DIR, train_verbosity=0, X_holdout=None, y_holdout=None):

	verify_inputs(X_train, X_test, X_holdout)
	if config.get('SEQ_LEN') is None:
		config['SEQ_LEN'] = 200

	from embeddingtools import tokenize_docs, tokens_to_sequences


	X_train_tokens = tokenize_docs(X_train[0])
	X_test_tokens = tokenize_docs(X_test[0])

	X_train[0] = tokens_to_sequences(X_train_tokens, train=True, maxlen=config['SEQ_LEN'])
	X_test[0] = tokens_to_sequences(X_test_tokens, maxlen=config['SEQ_LEN'])

	if len(X_train) == 2:

		config['d2v_include'] = True
		config['d2v_dim'] = X_train[1].shape[1]
	else:
		config['d2v_include'] = False







	if config['embed'] == 'word2vec':

		from gensim.models import Word2Vec
		if not isinstance(config['w2v_model'], Word2Vec):
			print('W2V model not passed from config. Training W2V model with {} dims for {} epochs...'.format(
				config['w2v_dims'], config['w2v_epochs']))

			config['w2v_model'] = Word2Vec(X_train_tokens, size=config['w2v_dims'], iter=config['w2v_epochs'], min_count=1)

		model, config = build_1dcnn(config)

		filestr = '{0}_dim{1}_epoch{2}_sdrop{3}_cf{4}_ck{5}_cl{6}_du{7}_dd{8}'.format(
			config['embed'],
			config['w2v_dims'],
			config['w2v_epochs'],
			config['spatial_dropout'],
			config['conv_filters'],
			config['conv_kernel'],
			config['conv_layers'],
			config['dense_units'],
			config['dense_dropout']
			)
		


	else:
		model, config = build_1dcnn(config)
		filestr = '{0}_dim{1}_sdrop{2}_cf{3}_ck{4}_cl{5}_du{6}_dd{7}'.format(
			config['embed'],
			config['glove_dim'],
			config['spatial_dropout'],
			config['conv_filters'],
			config['conv_kernel'],
			config['conv_layers'],
			config['dense_units'],
			config['dense_dropout']
			)
		
		
	


	X_train_tokens, X_test_tokens = None, None

	filepath = os.path.join(models_dir, 'cnn', filestr)
	weightsfile = filepath + '.h5'
	modelfile = filepath + '.json'
	
	store_model_json(model, modelfile)



	if config['train']:
		from keras.callbacks import EarlyStopping, ModelCheckpoint
		callbacks = [
			EarlyStopping(monitor='val_loss',
				patience=3,
				restore_best_weights=True),
			ModelCheckpoint(weightsfile, monitor='val_loss', 
				save_best_only=True )
			]
		print('Training CNN model with hyperparameters:')
		for key in config.keys():
			print('Hyperparam: {}\tValue: {}'.format(key, config[key]))

		history = model.fit(X_train, y_train, 
			validation_data=(X_test, y_test), 
			epochs=config['train_epochs'], 
			callbacks=callbacks,
			#verbose=train_verbosity
			)

		hist = history.history

		test_preds = model.predict(X_test)

		for i, flag in enumerate(target_flags):
			hist['best_val_rocauc_{}'.format(flag)] = rocauc(y_test[i], test_preds[i])
			hist['best_val_f1_{}'.format(flag)] = f1(y_test[i], test_preds[i])


		if (X_holdout is not None) & (y_holdout is not None):
			X_holdout_tokens = tokenize_docs(X_holdout[0])
			X_holdout[0] = tokens_to_sequences(X_holdout_tokens, maxlen=config['SEQ_LEN'])


			holdout_preds = model.predict(X_holdout)
			for i, flag in enumerate(target_flags):
				hist['best_holdout_rocauc_{}'.format(flag)] = rocauc(y_holdout[i], holdout_preds[i])
				hist['best_holdout_f1_{}'.format(flag)] = f1(y_holdout[i], holdout_preds[i])

		

		store_results_cnn(hist, config)
	else:
		print('Model not trained due to config train = False')

def train_rnn(config, X_train, y_train, X_test, y_test, models_dir=MODELS_DIR, train_verbosity=0, X_holdout=None, y_holdout=None):

	verify_inputs(X_train, X_test, X_holdout)

	if config.get('SEQ_LEN') is None:
		config['SEQ_LEN'] = 200

	from embeddingtools import tokenize_docs, tokens_to_sequences
	X_train_tokens = tokenize_docs(X_train[0])
	X_test_tokens = tokenize_docs(X_test[0])


	X_train[0]= tokens_to_sequences(X_train_tokens, train=True, maxlen=config['SEQ_LEN'])
	X_test[0] = tokens_to_sequences(X_test_tokens, maxlen=config['SEQ_LEN'])


	if len(X_train) == 2:

		config['d2v_include'] = True
		config['d2v_dim'] = X_train[1].shape[1]
	else:
		config['d2v_include'] = False




	if config['embed'] == 'word2vec':

		from gensim.models import Word2Vec
		if not isinstance(config['w2v_model'], Word2Vec):
			print('W2V model not passed from config. Training W2V model with {} dims for {} epochs...'.format(
				config['w2v_dims'], config['w2v_epochs']))
			config['w2v_model'] = Word2Vec(X_train_tokens, size=config['w2v_dims'], iter=config['w2v_epochs'], min_count=1)
		

		model, config = build_rnn(config)

		filestr = '{0}_dim{1}_epoch{2}_sdrop{3}_{4}_U{5}_bi{6}_du{7}_dd{8}.h5'.format(
			config['embed'],
			config['w2v_dims'],
			config['w2v_epochs'],
			config['spatial_dropout'],
			config['rnn_type'],
			config['rnn_units'],
			config['bidirectional'],
			config['dense_units'],
			config['dense_dropout']
			)
		

	else:
		model, config = build_rnn(config)
		filestr = '{0}_dim{1}_sdrop{2}_{3}_U{4}_bi{5}_du{6}_dd{7}'.format(
			config['embed'],
			config['glove_dim'],
			config['spatial_dropout'],
			config['rnn_type'],
			config['rnn_units'],
			config['bidirectional'],
			config['dense_units'],
			config['dense_dropout']
			)
		



	X_train_tokens, X_test_tokens = None, None

	filepath = os.path.join(models_dir, 'rnn', filestr)
	weightsfile = filepath + '.h5'
	modelfile = filepath + '.json'
	
	store_model_json(model, modelfile)

	if config['train']:
		from keras.callbacks import EarlyStopping, ModelCheckpoint
		callbacks = [
			EarlyStopping(monitor='val_loss',
				patience=3,
				restore_best_weights=True),
			ModelCheckpoint(weightsfile, 
				monitor='val_loss', 
				save_best_only=True)
		]

		print('Training RNN model with hyperparameters:')
		for key in config.keys():
			print('Hyperparam: {}\tValue: {}'.format(key, config[key]))

		history = model.fit(X_train, y_train, 
			validation_data=(X_test, y_test), 
			epochs=config['train_epochs'], 
			callbacks=callbacks,
			#verbose=train_verbosity
			)


		hist = history.history
		test_preds = model.predict(X_test)

		for i, flag in enumerate(target_flags):
			hist['best_val_rocauc_{}'.format(flag)] = rocauc(y_test[i], test_preds[i])
			hist['best_val_f1_{}'.format(flag)] = f1(y_test[i], test_preds[i])


		if (X_holdout is not None) & (y_holdout is not None):
			X_holdout_tokens = tokenize_docs(X_holdout[0])
			X_holdout[0] = tokens_to_sequences(X_holdout_tokens, maxlen=config['SEQ_LEN'])


			holdout_preds = model.predict(X_holdout)
			for i, flag in enumerate(target_flags):
				hist['best_holdout_rocauc_{}'.format(flag)] = rocauc(y_holdout[i], holdout_preds[i])
				hist['best_holdout_f1_{}'.format(flag)] = f1(y_holdout[i], holdout_preds[i])


		store_results_rnn(hist, config)
	else:
		print('Model not trained due to config train = False')

def train_elmo(config, X_train, y_train, X_test, y_test, models_dir=MODELS_DIR, train_verbosity=0):

	model, config = build_elmo_dnn(config)


	from embeddingtools import tokenize_docs, prepare_elmo_seqs
	X_train_strings = prepare_elmo_seqs(tokenize_docs(X_train), maxlen=config['SEQ_LEN'])
	X_test_strings = prepare_elmo_seqs(tokenize_docs(X_test), maxlen=config['SEQ_LEN'])

	from keras.callbacks import EarlyStopping, ModelCheckpoint
	filestr = 'du{}_dd{}.h5'.format(
		config['dense_units'],
		config['dense_dropout']
		)
	filepath = os.path.join(models_dir, 'elmo', filestr)

	weightsfile = filepath + '.h5'
	modelfile = filepath + '.json'
	
	store_model_json(model, modelfile)

	callbacks = [
		EarlyStopping(monitor='val_loss',
			restore_best_weights=True,
			patience=2),
		ModelCheckpoint(weightsfile, monitor='val_loss', 
			save_best_only=True, 
			)
	]
	print('Training ELMo model with hyperparameters:')
	for key in config.keys():
		print('Hyperparam: {}\tValue: {}'.format(key, config[key]))
	history = model.fit(X_train_strings, y_train, 
		validation_data=(X_test_strings, y_test), 
		epochs=20, 
		callbacks=callbacks,
		#verbose=train_verbosity
		)

	store_results_elmo(history.history, config)

def store_model_json(model, file):
	import os
	js = model.to_json()
	with open(file, 'w') as f:
		f.write(js)