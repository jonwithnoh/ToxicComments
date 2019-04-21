from keras.engine import Layer
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K


EMBEDDING_FOLDER = 'embedding'
SEQ_LEN = 200
class ElmoEmbeddingLayer(Layer):
	# This was generously cribbed from:
	# https://github.com/strongio/keras-elmo/blob/master/Elmo%20Keras.ipynb
	# Other ELMo resources
	# https://github.com/PrashantRanjan09/Elmo-Tutorial/blob/master/Elmo_tutorial.ipynb
	# https://www.depends-on-the-definition.com/named-entity-recognition-with-residual-lstm-and-elmo/
	
	
	def __init__(self, **kwargs):
		self.dimensions = 1024
		self.trainable = True
		super(ElmoEmbeddingLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		url = 'https://tfhub.dev/google/elmo/2'
		self.elmo = hub.Module(url,
			trainable=self.trainable,
			name='{}_module'.format(self.name)
			)

		self.trainable_weights += K.tf.trainable_variables(
			scope='^{}_module/.*'.format(self.name))

		super(ElmoEmbeddingLayer, self).build(input_shape)

	def call(self, x, mask=None):
		result = self.elmo(
			K.squeeze(
				K.cast(
					x, tf.string
					), 
				axis=1), 
			as_dict=True,
			signature='default',)['default'] 
			#default denotes character-level embedding vs tokens for tokenized rep 

		return result

	def compute_mask(self, inputs, mask=None):
		return K.not_equal(inputs, '--PAD--')

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.dimensions)

def tokenize_docs(docs, verbose=False, n=50, return_token_counter=False):
	from collections import Counter

	token_counter = Counter()

	docs_tokenized = []

	for doc in docs:
		tokens = doc.split()
		token_counter.update(tokens)
		docs_tokenized.append(tokens)

	if verbose:
		print('Full vocab size: {}'.format(len(token_counter.keys())))
		mc = token_counter.most_common(n)
		for i in mc:
			print('{}: {}'.format(i[0], i[1]))
	if return_token_counter:
		return docs_tokenized, token_counter
	else:
		return docs_tokenized

def tokenize_all_docs(train, test, holdout=None, verbose=False):

	# tokenize all strings. Assumes all punctuation has been removed.
	# check if holdout set is included
	# for word embeddings based approaches


	train, train_tok_counter = tokenize_docs(train, verbose=verbose, n=10)
	test, _ = tokenize_docs(test, verbose=False)

	if holdout is not None:
		holdout, _ = tokenize_docs(holdout, verbose=False)
		return train, test, holdout

	else:
		return train, test

def prepare_elmo_seqs(docs, maxlen=SEQ_LEN):
	import numpy as np
	docs_trunc = []
	for doc in docs:
		docs_trunc.append(
			' '.join(doc[0:maxlen])
			)

	docs_trunc = np.array(docs_trunc, dtype='object')[:,np.newaxis]
	return docs_trunc


def tokens_to_sequences(docs, glove=False, train=False, tokenizer_pickle='tokenizer.pickle', maxlen=SEQ_LEN):
	from keras.preprocessing.text import Tokenizer
	from keras.preprocessing.sequence import pad_sequences
	from tca import pkl_dump, pkl_load

	if train and not glove:
		tk = Tokenizer()
		tk.fit_on_texts(docs)
		pkl_dump(tk, tokenizer_pickle) # store tokenizer for later use

	elif train and glove:
		_ = glove_embedding_matrix()
		tk = pkl_load(tokenizer_pickle)

	else:
		tk = pkl_load(tokenizer_pickle) #restore tokenizer

	seqs = tk.texts_to_sequences(docs)
	seqs = pad_sequences(seqs, maxlen=maxlen, padding='pre')

	return seqs

def train_gensim_w2v(train, config, save_increment=5, path=EMBEDDING_FOLDER+'/word2vec'):
	reqfields = {'w2v_epochs', 'w2v_dims'}
	if not reqfields <= set(config):
		print('W2V Config must be a dict with keys {}'.format(reqfields))

	else:
		from gensim.models import Word2Vec
		model = Word2Vec(train, size=config['w2v_dims'], min_count=1, iter=config['w2v_epochs'])
		return model
		'''
		for dim in config['dims']:
			print('Training Word2Vec model with {} dims over {} epochs'.format(dim, epochs))
			model = Word2Vec(train, size=dim, min_count=1, iter=0)

			for epoch in range(1, epochs + 1):
				model.train(train, total_examples=len(train), epochs=1)
				if epoch % save_increment == 0:
					print('Saving Word2Vec model with {} dims at epoch {}'.format(dim, epoch))
					savefile = '{}/dims{} epochs{}.w2v'.format(path, dim, epoch)
					model.save(savefile)
		'''


def load_gensim_wordvecs(file, path=EMBEDDING_FOLDER+'/word2vec'):
	from gensim.models import Word2Vec
	model = Word2Vec.load('{}/{}'.format(path, file))
	model_wv = model.wv
	return model_wv

def get_max_seq_len(docs_encoded):
	maxlen = 0
	for doc in docs_encoded:
		if len(doc) > maxlen:
			maxlen = len(doc)
	return maxlen

def stored_embedding_check(vocab, embedding, embeddingfolder=EMBEDDING_FOLDER):
	import glob
	import hashlib
	vocstr = ''.join(vocab.keys())
	md5str = hashlib.md5(vocstr.encode()).hexdigest()
	filestr = '{}/{}/{}.pickle'.format(embeddingfolder, embedding, md5str)
	files = glob.glob(filestr)
	return md5str, files

def gensim_embedding_matrix(wordvectors, vocab, embeddingfolder=EMBEDDING_FOLDER+'/word2vec'):
	import numpy as np
	from tca import pkl_dump, pkl_load
	md5str, files = stored_embedding_check(vocab, 'word2vec')
	
	if len(files) > 0:
		print('Loading {}'.format(files[0]))
		weight_matrix = pkl_load(files[0])
		print('Gensim Word2Vec weight matrix loaded from ID: {}'.format(md5str))
		return weight_matrix

	else:
		vocab_size = len(vocab) + 1
		
		key = next(iter(wordvectors.wv.vocab))
		dims = wordvectors[key].shape[0]
		
		weight_matrix = np.zeros((vocab_size, dims))
		
		for word, i in vocab.items():
			weight_matrix[i] = wordvectors[word]

		filestr = '{}.pickle'.format(md5str)
		path = '{}/{}'.format(embeddingfolder,filestr)
		#pkl_dump(weight_matrix, path) creates really fast anyway
		print('Gensim Word2Vec weight matrix created with dims: {}'.format(weight_matrix.shape))
		return weight_matrix


def glove_embedding_matrix(dims=100, embeddingfolder=EMBEDDING_FOLDER+'/glove', tokenizer_pickle='tokenizer.pickle'):

	'''
	use full GLoVe 6B vectors to populate embedding matrix with all words
	'''
	import zipfile
	import os
	import numpy as np
	from tca import pkl_dump
	from keras.preprocessing.text import Tokenizer

	# borrowed from https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
	file = 'glove.6B.{}d.txt'.format(dims)
	with zipfile.ZipFile(os.path.join(embeddingfolder, 'glove.6B.zip'), 'r') as zipobj:
		zipobj.extract(file, embeddingfolder)

	# borrowed from https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
	embeddings_index = {}
	filepath = os.path.join(embeddingfolder, file)
	file = open(filepath, 'r', encoding='utf8')
	for line in file:
		items = line.split()
		word = items[0]
		vector = np.asarray(items[1:], dtype='float32')
		embeddings_index[word] = vector
	file.close()

	allwords = list(embeddings_index.keys())
	
	tk = Tokenizer()
	tk.fit_on_texts(allwords)

	os.remove(filepath) # remove embedding text file
	pkl_dump(tk, tokenizer_pickle)

	weight_matrix = np.zeros((len(tk.word_index.keys())+1, dims))

	for word, idx in tk.word_index.items():
		vec = embeddings_index.get(word)
		weight_matrix[idx] = vec

	return weight_matrix


	

def spacy_embedding_matrix(vocab, lang_model='en_core_web_lg', embeddingfolder=EMBEDDING_FOLDER+'/glove'):
	'''
	Use spacy's built in word vectors to populate embedding matrix based on word tokens in corpus
	'''
	import spacy
	import numpy as np
	from tca import pkl_dump, pkl_load
	import glob
	import hashlib

	md5str, files = stored_embedding_check(vocab, 'glove')
	
	if len(files) > 0:
		print('Loading {}'.format(files[0]))
		weight_matrix = pkl_load(files[0])
		print('spaCy GLoVe weight matrix loaded from ID: {}'.format(md5str))
		return weight_matrix

	else:
		nlp = spacy.load(lang_model)
		nodim = True
		for word, i in vocab.items():
			token = nlp(word)
			for tkn in token:
				if tkn.has_vector:
					if nodim:
						dims = token.vector.shape[0]
						nodim = False

						vocab_size = len(vocab) + 1
						weight_matrix = np.zeros((vocab_size, dims))
						print('Empty spaCy GLoVe weight matrix created with dims: {}, populating embeddings from tokenizer...'.format(weight_matrix.shape))

					weight_matrix[i] = tkn.vector
		print('Complete. Saving...')
		filestr = '{}.pickle'.format(md5str)
		path = '{}/{}'.format(embeddingfolder,filestr)
		pkl_dump(weight_matrix, path)
		print('Complete.')
		return weight_matrix

def make_embedding_layer(weight_matrix, SEQ_LEN):
	from keras.layers import Embedding

	embeddng_layer = Embedding(weight_matrix.shape[0],
		weight_matrix.shape[1],	
		weights=[weight_matrix],
		input_length=SEQ_LEN,
		trainable=False
		)
	return embeddng_layer

def get_docs_with_missing_tokens(seqs, doctokens):
	# this function will accept zero padded token sequences (zeros IN FRONT)
	# and return any unrecognized (zero value) tokens after the first nonzero entry in the sequence
	import numpy as np
	assert len(seqs) == len(docs)


	for idx in range(len(seqs)):
		zeros = np.argwhere(seqs[idx][(seqs[idx] != 0).argmax():] == 0).ravel()
		if len(zeros) == len(seqs[idx]):

			print(idx)
			print('{}\t{}\n'.format(doctokens[idx]))
		
		elif len(zeros) > 0:
			for zero in range(zeros.shape[0]):
				print(idx)
				print('\t{}: {}'.format(zero, doctokens[idx][zero]))
