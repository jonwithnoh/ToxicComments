'''
ToxicClassAnalysis object contains all of
'''

class ToxicCommentsAnalysis:
	def __init__(self, file, DATA_DIR='data', imagesavefolder='eda_images'):
		''' initialize object, set some basic flags '''
		import pandas as pd

		self.DATA_DIR = DATA_DIR
		self.data_split_exists = False


		self.df = set_basic_flags(
			pd.read_csv('{}/{}'.format(self.DATA_DIR, file))
			)

		self.toxic_flags_ = self.df.columns.tolist()[2:8]
		
		flags = self.df.columns[2:-2]
		flagcounts = pd.get_dummies(self.df[flags].sum(axis=1))
		multicols = flagcounts.columns.tolist()[1:]

		for col in multicols:
			self.df[col] = flagcounts[col]

		self.toxic_flags_ = self.df.columns.tolist()[2:8]
		self.all_toxic_flags_ = self.df.columns.tolist()[2:]
		self.imgsavefolder_ = imagesavefolder

		print('Toxic comments object initialized')

	def create_sample(self, sample_size=1000, random_state=42):
		'''
		creates a sample of size n, useful for manifold transforms,
		full data is too computationally expensive
		'''

		self.sample_size = sample_size
		self.df_sample = self.df.sample(sample_size, random_state=random_state)

	def vectorize_text(self, mode='count', data='sample', ngram_range=(1,4), min_df=2,
		max_features=int(1e4), verbose=False, d2v_params={}):
		'''
		creates object .vectorizer which can be utilized later for transforming unseen documents
		<obj name>.vectorizer.transform(...)
		
		ngram_range = tuple of ngrams to identify (default 1-4)
		min_df = minimum frequency in corpus (default 2)

		'mode' option determines how to vectorize. allowed options are:
		count - simple count
		prop - proportional (i.e. entire row adds to 1)
		bin - binary (i.e. present or not)
		tfidf - TFIDF (I really don't think it'll be that useful here but whatevs)

		'data' option has the following options:
		all - entire train data from Kaggle
		sample - df_sample created by create_sample() function (for manifold viz)
		train - training data created by train_test_holdout_split()
		test - testing data created by train_test_holdout_split()
		holdout - holdout data created by train_test_holdout_split()

		if you attempt to use train, test, or holdout without splitting data, it'll error

		if you use test or holdout, it will use the existing vectorizer, instead of re-fitting.

		Test and holdout sets create .X_test_ variable. All others create .X_ variable

		'''

		allowed_data = ['all', 'sample', 'train', 'test', 'holdout']
		allowed_data_str = ', '.join(allowed_data)

		if data not in allowed_data:
			raise Exception('Invalid mode entered: {}, allowed modes: {}'.format(data, allowed_data_str))
		else:
			if data in allowed_data[-3:]:
				if not self.data_split_exists:
					raise Exception('Data must be split with .train_test_holdout_split() first')



		testholdout = allowed_data[2:]
		trainable = allowed_data[:2]
		novec = False

		if data in trainable:

			if mode == 'count':
				from sklearn.feature_extraction.text import CountVectorizer
				self.vectorizer = CountVectorizer(
					ngram_range=ngram_range,
					binary=False,
					min_df=min_df,
					max_features=max_features
					)

			elif mode == 'prop':
				from sklearn.feature_extraction.text import CountVectorizer
				self.vectorizer = CountVectorizer(
					ngram_range=ngram_range,
					binary=False,
					min_df=min_df,
					max_features=max_features
					)

			elif mode == 'bin':
				from sklearn.feature_extraction.text import CountVectorizer
				self.vectorizer = CountVectorizer(
					ngram_range=ngram_range,
					binary=True,
					min_df=min_df,
					max_features=max_features
					)

			elif mode == 'tfidf':
				from sklearn.feature_extraction.text import TfidfVectorizer
				self.vectorizer = TfidfVectorizer(
					ngram_range=ngram_range,
					min_df=min_df,
					max_features=max_features
					)
			elif mode == 'doc2vec':
				
				from gensim.models.doc2vec import TaggedDocument, Doc2Vec
				d2v_param_defaults = {
					'vector_size': 200,
					'min_count': 10,
					'window': 10,
					'dm': 1,
					'epochs': 30
				}
				

				try:
					assert isinstance(d2v_params, dict)
				except AssertionError as e:
					e.args += ('d2v_params must be dict',)
					raise

				for param in set(d2v_param_defaults) - set(d2v_params):
					d2v_params[param] = d2v_param_defaults[param]

				if data == 'all':
					df = self.df
				elif data == 'sample':
					df = self.df_sample
				elif data == 'train':
					df = self.df_train

				if verbose:
					print('Preparing tagged documents')
				tagged_docs = make_tagged_documents(df, 'comment_scrubbed', self.toxic_flags_)

				if verbose:
					print('Training doc2vec model')

				self.vectorizer = Doc2Vec(
					tagged_docs,
					size=d2v_params['vector_size'], 
					window=d2v_params['window'],
					dm=d2v_params['dm'],
					epochs=d2v_params['epochs'])

				
		else:
			try:
				novec = False
				self.vectorizer
			except:
				novec = True

		if novec:
			raise Exception("No vectorizer stored in object. Vectorize 'all', 'sample', or 'train' first")

		docs = {}

		if mode != 'doc2vec':
			if data == 'train-test-holdout':
				
				docs['train'] = self.df_train['comment_scrubbed'].tolist()
				docs['test'] = self.df_test['comment_scrubbed'].tolist()
				docs['holdout'] = self.df_holdout['comment_scrubbed'].tolist()


			if data == 'sample':		
				docs['sample'] = self.df_sample['comment_scrubbed'].tolist()

			elif data == 'train':
				docs['train'] = self.df_train['comment_scrubbed'].tolist()

			elif data == 'test':
				docs['test'] = self.df_test['comment_scrubbed'].tolist()

			elif data == 'holdout':
				docs['holdout'] = self.df_holdout['comment_scrubbed'].tolist()

			elif data == 'all':
				docs ['all'] = self.df['comment_scrubbed'].tolist()




			if data in ['sample']:
				if verbose:
					print('Vectorizing sample data...')
				self.X_sample_ = self.vectorizer.fit_transform(docs['sample'])
				self.X_sample_feature_names_ = self.vectorizer.get_feature_names()
				self.sample_vectorize_mode_ = mode
				if verbose:
					print('Done.')

			if data in ['train', 'all', 'train-test-holdout']:
				if verbose:
					print('Vectorizing training data...')
				if data == 'all':
					self.X_ = self.vectorizer.fit_transform(docs['all'])
				else:
					self.X_ = self.vectorizer.fit_transform(docs['train'])
				self.X_feature_names_ = self.vectorizer.get_feature_names()
				self.vectorize_mode_ = mode
				self.vectorize_data_ = data
				if verbose:
					print('Done.')

			if data in ['test', 'train-test-holdout']:
				if verbose:
					print('Vectorizing test data...')
				self.X_test_ = self.vectorizer.transform(docs['test'])
				if verbose:
					print('Done.')

			if data in ['holdout', 'train-test-holdout']:
				if verbose:
					print('Vectorizing holdout data...')
				self.X_holdout_ = self.vectorizer.transform(docs['holdout'])
				if verbose:
					print('Done.')

			if mode == 'prop':
				import numpy as np
				'''
				Had to switch to a function rather than broadcast divide row-wise sum
				due to running out of memory. Good times.
				'''
				if data in ['sample']:
					if verbose:
						print('Converting count to proportional for sample data...')
					self.X_sample_ = convert_matrix_count_to_prop(self.X_sample_)
					if verbose:
						print('Done.')

				elif data in ['train', 'all']:
					if verbose:
						print('Converting count to proportional for train data...')
					self.X_ = convert_matrix_count_to_prop(self.X_)
					self.df_corr_results = None
					if verbose:
						print('Done.')

				elif data in ['test']:
					if verbose:
						print('Converting count to proportional for test data...')
					self.X_test_ = convert_matrix_count_to_prop(self.X_test_)
					if verbose:
						print('Done.')
				
				elif data in ['holdout']:
					if verbose:
						print('Converting count to proportional for holdout data...')
					self.X_holdout_ = convert_matrix_count_to_prop(self.X_holdout_)
					if verbose:
						print('Done.')
		
		elif mode == 'doc2vec':
			tagged_docs = make_tagged_documents(self.df)
			model = Doc2Vec(tagged_docs, vector_size=200, window=3, min_count=10, workers=4)
			model.train(tagged_docs, epochs=30, total_examples=model.corpus_count)

			if data in ['sample']:


				docvecs = []
				sampledocs = spacy_tokenize(self.df_sample['comment_scrubbed'].tolist())
				for sampledoc in sampledocs:
					docvecs.append(model.infer_vector(doc_words=sampledoc))

				self.X_sample_doc2vec_ = np.asarray(docvecs)


		'''		
		else:
			if data in ['sample']:
				self.X_sample_ = self.X_sample_

			elif data in ['train']:
				self.X_ = self.X_
			elif data in ['test']:
				self.X_test_ = self.X_test_
			elif data in ['holdout']:
				self.X_test_ = self.X_test_
		'''


	def manifold(self, mode='isomap', use_pca=True, pca_dim=50, data='bow'):
		'''
		creates a manifold represenation of the high dimensional data,
		PCA preprocessing is optional, but should significantly accelerate 
		processing time (i.e. dropping from 10,000 dimensions down to whatever is
		specified.) 

		manifold will *ONLY WORK ON SAMPLE*
		'''
		from scipy.sparse import issparse

		self.manifold_dict_ = {
			'isomap': 'Isomap',
			'mds': 'Multidimensional Scaling',
			'tsne': 't-SNE'
		}

		if mode == 'isomap':
			from sklearn.manifold import Isomap
			mf = Isomap()


		if mode == 'mds':
			from sklearn.manifold import MDS
			mf = MDS()

		if mode == 'tsne':
			from sklearn.manifold import TSNE
			mf = TSNE()

		if issparse(self.X_sample_):
			self.X_sample_ = self.X_sample_.todense()

		if use_pca == True:
			from sklearn.decomposition import PCA
			pca = PCA(n_components=pca_dim)
			self.X_sample_ = pca.fit_transform(self.X_sample_)

		
		self.X_sample_mf_ = mf.fit_transform(self.X_sample_)
		self.df_sample['mf_bow_X0'] = self.X_sample_mf_[:,0]
		self.df_sample['mf_bow_X1'] = self.X_sample_mf_[:,1]
		self.manifold_ = mode
		self.manifold_pca_ = use_pca

	def prep_manifold_plotting(self, max_sim_age=30):
		import numpy as np
		self.df_sample.loc[self.df_sample[1] == 1, self.df_sample.columns[2:8]]
		self.df_sample['single_flag'] = np.nan
		self.df_sample['all_flags'] = ''
		

		for single_flag in self.toxic_flags_:
			self.df_sample.loc[(self.df_sample[single_flag] == 1) &
								(self.df_sample[1] == 1), 'single_flag'] = single_flag

			self.df_sample['tempstr'] = ''
			self.df_sample.loc[self.df_sample[single_flag] == 1, 'tempstr'] = single_flag + ', '
			self.df_sample['all_flags'] = self.df_sample['all_flags'] + self.df_sample['tempstr'] # keep working on this

		self.df_sample.drop(['tempstr'], axis=1, inplace=True)

		self.df_sample['all_flags'] = self.df_sample['all_flags'].str[:-2]
		for multi_flag in self.df.columns[9:16]:
			if multi_flag != 1:
				self.df_sample.loc[(self.df_sample[multi_flag] == 1), 
				'single_flag'] = '{} flags'.format(multi_flag)

		flags = {
			'toxic': (5, '#%02x%02x%02x' % (round(.5 * 255), round(.8*255), round(.15*255)), .8) ,
			'threat': (5,'#%02x%02x%02x' % (round(.8*255), round(.6*255), round(.15*255)), .5),
			'obscene': (5, '#%02x%02x%02x' % (round(.8*255), round(.07*255), round(.8*255)), .5),
			'identity_hate': (5,'#%02x%02x%02x' % (round(.4*255), round(0*255), round(.07*255)), .5),
			'insult': (5, '#%02x%02x%02x' % (round(.25*255), round(0*255), round(.45*255)), .7),
			#'severe_toxic': (20,.7),
			'none flags': (3, '#%02x%02x%02x' % (round(.14*255), round(.14*255), round(.8*255)), .3),
			'2 flags': (7, '#%02x%02x%02x' % (round(1*255), round(.95*255), round(.33*255)), .55),
			'3 flags': (8, '#%02x%02x%02x' % (round(1*255), round(.8*255), round(.33*255)), .55),
			'4 flags': (10, '#%02x%02x%02x' % (round(1*255), round(.6*255), round(.33*255)), .55),
			'5 flags': (12, '#%02x%02x%02x' % (round(1*255), round(.4*255), round(.4*255)), .5),
			'6 flags': (15, '#%02x%02x%02x' % (round(1*255), round(.2*255), round(.2*255)), .5)
		}
		for flag in flags.keys():
			self.df_sample.loc[self.df_sample['single_flag'] == flag, 'plot_size'] = flags[flag][0]
			self.df_sample.loc[self.df_sample['single_flag'] == flag, 'plot_color'] = flags[flag][1]
			self.df_sample.loc[self.df_sample['single_flag'] == flag, 'plot_alpha'] = flags[flag][2]

		self.df_sample['sim_age'] = np.random.uniform(0, max_sim_age, self.df_sample.shape[0])

	def plot_manifold(self, figsize=(12,8), save=True):

		import seaborn as sns
		import matplotlib.pyplot as plt


		flags = {'toxic': (20, (.5, .8, .15, .5) ),
				'threat': (20,(.8, .6, .15, .5)),
				'obscene': (20,(.8, .07, .8, .5)),
				'identity_hate': (20,(.4, 0, .07, .5)),
				'insult': (20, (.25, 0, .45, .7)),
				#'severe_toxic': (20,.7),
				'none flags': (1, (.14, .14, .8, .3)),
				'2 flags': (40, (1, .95, .33, .55)),
				'3 flags': (60, (1, .8, .33, .55)),
				'4 flags': (70, (1, .6, .33, .55)),
				'5 flags': (80, (1, .4, .4, .5)),
				'6 flags': (90, (1, .2, .2, .5))}

		figsize = (12,9)
		fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

		for flag in flags.keys():   
			x = self.df_sample.loc[self.df_sample['single_flag'] == flag, 'mf_X0'].values
			y = self.df_sample.loc[self.df_sample['single_flag'] == flag, 'mf_X1'].values
			
			ax.scatter(x, y, s=flags[flag][0], c=flags[flag][1], label=flag) #alpha=flags[flag][1], label=flag)

		for i in range(6,1,-1):
			if self.df_sample.loc[self.df_sample[i] == 1,].shape[0] >= 1:
				searching = True
				while searching:
		            
					extreme = self.df_sample.loc[self.df_sample[i] == 1,].sample(1)
					extreme = extreme.loc[extreme.index[0]]
		            
					if extreme['mf_X0'] < 0 and extreme['mf_X1'] > 0:
						searching = False
				break

		searching = True
		while searching:
			benign = self.df_sample.loc[self.df_sample['none'] == 1,].sample(1)
			benign = benign.loc[benign.index[0]]
			if benign['mf_X0'] > 0 and benign['mf_X1'] < 0:
				searching = False
		        
		exflags = []
		for flag in self.toxic_flags_:
			if extreme[flag] == 1:
				exflags.append(flag)

		s = extreme['comment_text'][0:75] + '...\nFlags: ' + ', '.join(exflags)
		ax.annotate(xy=(extreme['mf_X0'], extreme['mf_X1']), xytext=(-60,80), s=s,
					arrowprops=dict(facecolor='black', shrink=0.03))

		s = benign['comment_text'][0:75] + '...\n(benign)'
		ax.annotate(xy=(benign['mf_X0'], benign['mf_X1']), xytext=(0,-70), s=s,
					arrowprops=dict(facecolor='black', shrink=0.03))
		ax.set_ylim(bottom=-100, top=100)
		ax.set_xlim(left=-100, right=100)
		ax.legend()
		ax.axis('off')
		if save:
			fig.savefig('{}/t_sne annotated.png'.format(self.imgsavefolder_))

	def plot_manifold_bokeh(self, figsize=(800,600), output='notebook', interactive=False):
		from bokeh.layouts import column
		from bokeh.models import ColumnDataSource, Slider
		from bokeh.models.widgets import RangeSlider, CheckboxButtonGroup, Select
		from bokeh.plotting import figure
		from bokeh.themes import Theme
		from bokeh.io import show, output_notebook
		output_notebook()

		if (output == 'notebook') and interactive:
			
			def modify_doc(doc):
				df = self.df_sample
				source = ColumnDataSource(data={
					'x': df['mf_X0'],
					'y': df['mf_X1'],
					'color': df['plot_color'],
					'alpha': df['plot_alpha'],
					'size': df['plot_size'],
					'text': df['comment_text'].str[0:75] + '...' ,
					'flags': df['all_flags'],
					'single_flag': df['single_flag'],
					'sim_age': df['sim_age']
				})
				buttonlabels = self.toxic_flags_ + ['none']
				def callback(attr, old, new):
					if 0 in buttons.active:
						toxic = [0,1]
					else:
						toxic = [0]

					if 1 in buttons.active:
						severe_toxic = [0,1]
					else:
						severe_toxic = [0]

					if 2 in buttons.active:
						obscene = [0,1]
					else:
						obscene = [0]

					if 3 in buttons.active:
						threat = [0,1]
					else:
						threat = [0]

					if 4 in buttons.active:
						insult = [0,1]
					else:
						insult = [0]

					if 5 in buttons.active:
						identity = [0,1]
					else:
						identity = [0]

					if 6 in buttons.active:
						none = [0,1]
					else:
						none = [0]

					df_slice = df[
						(df.sim_age >= slider.value[0]) & 
						(df.sim_age <= slider.value[1]) &
						(df.toxic.isin(toxic)) &
						(df.severe_toxic.isin(severe_toxic)) &
						(df.obscene.isin(obscene)) &
						(df.threat.isin(threat)) &
						(df.insult.isin(insult)) &
						(df.identity_hate.isin(identity)) &
						(df.none.isin(none))
					]

					if select.value == 'Bag of Words':
						x = 'mf_X0'
						y = 'mf_X1'
					else:
						x = 'mf_d2v_X0'
						y = 'mf_d2v_X1'

					source.data = ColumnDataSource(data={
						'x': df_slice[x],
						'y': df_slice[y],
						'color': df_slice['plot_color'],
						'alpha': df_slice['plot_alpha'],
						'size': df_slice['plot_size'],
						'text': df_slice['comment_text'].str[0:75] + '...' ,
						'flags': df_slice['all_flags'],
						'single_flag': df_slice['single_flag'],
						'sim_age': df_slice['sim_age']
					}).data

				slider = RangeSlider(start=0, end=30, value=(0,30), step=1, title='Simulated Age')
				slider.on_change('value', callback)

				buttons = CheckboxButtonGroup(labels=buttonlabels, active=[0,1,2,3,4,5,6])
				buttons.on_change('active', callback)

				select = Select(title='Vectorization', value='Bag of Words', options=['Bag of Words', 'Doc2Vec'])
				select.on_change('value', callback)


				tooltips = [
					('Comment', '@text'),
					('Flags', '@flags'),
					('Age (d)', '@sim_age')
				]

				n_obs = self.df_sample.shape[0]
				manif = self.manifold_dict_[self.manifold_]
				title = '{} visualization of {} observations'.format(manif, n_obs)

				p = figure(plot_width=figsize[0], plot_height=figsize[1], title=title, tooltips=tooltips)
				p.circle(x='x', y='y', color='color', alpha='alpha', size='size', legend='single_flag',
				 source=source)

				p.xgrid.visible = False
				p.ygrid.visible = False

				p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
				p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks

				p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
				p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks

				p.xaxis.major_label_text_font_size = '0pt'  # turn off x-axis tick labels
				p.yaxis.major_label_text_font_size = '0pt'  # turn off y-axis tick labels

				p.title.text_color = 'black'
				p.title.text_font = 'calibri'

				p.legend.location = "top_left"

				doc.add_root(column(slider, select, buttons, p))

		else:
			source = ColumnDataSource(data={
				'x': self.df_sample['mf_X0'],
				'y': self.df_sample['mf_X1'],
				'color': self.df_sample['plot_color'],
				'alpha': self.df_sample['plot_alpha'],
				'size': self.df_sample['plot_size'],
				'text': self.df_sample['comment_text'].str[0:75] + '...' ,
				'flags': self.df_sample['all_flags'],
				'single_flag': self.df_sample['single_flag'],
				'sim_age': self.df_sample['sim_age']
			})

			tooltips = [
			    ('Comment', '@text'),
			    ('Flags', '@flags'),
			    ('Age (d)', '@sim_age')
			]

			n_obs = self.df_sample.shape[0]
			manif = self.manifold_dict_[self.manifold_]
			title = '{} visualization of {} observations'.format(manif, n_obs)

			p = figure(plot_width=figsize[0], plot_height=figsize[1], title=title, tooltips=tooltips)
			p.circle(x='x', y='y', color='color', alpha='alpha', size='size', legend='single_flag',
			     source=source)

			p.xgrid.visible = False
			p.ygrid.visible = False

			p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
			p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks

			p.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
			p.yaxis.minor_tick_line_color = None  # turn off y-axis minor ticks

			p.xaxis.major_label_text_font_size = '0pt'  # turn off x-axis tick labels
			p.yaxis.major_label_text_font_size = '0pt'  # turn off y-axis tick labels

			p.title.text_color = 'black'
			p.title.text_font = 'calibri'

			p.legend.location = "top_left"

		if output == 'notebook':
			if interactive:
				show(modify_doc)
			else:
				show(p)
		elif output == 'html':
			from bokeh.resources import CDN
			from bokeh.embed import file_html

			html = file_html(p, CDN, manif)

			filename = '{}, {} samples.html'.format(manif, n_obs)
			with open(filename, 'w') as f:
				f.write(html)		

			

	def scrub_comments(self, convert_ec=True, 
		drop_stopwords=False, 
		lemmatize=True,
		lemmatize_pronouns=False,
		equivclasses='toxic-ec.json'
		):
		'''
		Expands contractions, removes posessives, and removes non-alpha characters.
		Optionally: converts equivalence classes, lemmatizes text (optionally pronouns),
		removes stopwords 

		make sure you install spaCy and the 'en' language pack:
		python -m spacy download en

		Possibly deprecated:
		if return_tokens == True
	    Takes original df, extracts comments, cleans them (extend contractions, 
	    strip posessive, strip non-alpha), and dumps them into a 'tall' dataframe 
	    with the fields 'id' from the original comment and 'token' for the word
	    '''
		
		import spacy
		nlp = spacy.load('en')


		print('Scrubbing comments.\nEC conversion: {}\nDrop stopwords: {}\nLemmatize:{}'.format(
			convert_ec, drop_stopwords, lemmatize))

		if convert_ec:
			toxic_ec = load_json('{}/{}'.format(self.DATA_DIR, equivclasses))


		docs = self.df['comment_text'].tolist()
		ids = self.df['id'].tolist()


		df_prep = []

		docs_prepped = repeated_char_remover(
			strip_posessive(
				extend_contractions(
					replace_ip_in_docs(docs)
					)
				)
			)
		docs_clean = []



		for i, doc in enumerate(docs_prepped):

			doc_nlp = nlp(doc)
			doc_clean = []

			for token in doc_nlp:
				if lemmatize:
					if lemmatize_pronouns:
						tok = token.lemma_

					else:
						if token.lemma_ == '-PRON-':
							tok = token.text.lower()
						else:
							tok = token.lemma_

				else:
					tok = token.text

				if drop_stopwords:
					if not token.is_stop:
						doc_clean.append(tok)
				else:
					doc_clean.append(tok)

			docstr = ' ' + ' '.join(doc_clean) + ' '
			docstr = docstr.replace('\n', ' ').replace('\t', ' ')

			

			if convert_ec:
				for toxic_class in toxic_ec.keys():
					for toxic_word in toxic_ec[toxic_class].keys():
						for alias in toxic_ec[toxic_class][toxic_word]:
							if docstr.find(' '+ alias + ' ') > -1:
								docstr = docstr.replace(
									' '+ alias + ' ',
									' '+ toxic_word + ' '
									)
								proportion = i/len(docs)
								pct = round(proportion*100,3)
								print('{} ({}%): {} ===> {}'.format(ids[i], pct, alias, toxic_word))

			else:
				if i % 1000 == 0 and i > 0:
					proportion = i/len(docs)
					pct = round(proportion*100,3)
					print('{}%...'.format(pct), end=' ')

			docs_clean.append(docstr)
	        
			
		docs_clean = strip_nonalpha(docs_clean)
		self.df['comment_scrubbed'] = docs_clean

		print('Comment scrubbing complete.')

	def add_upper_lower_punc(self, verbose=False):
		'''
		calculates the proportion of characters in comment that are upper, lower, and punctuation,
		writes results to .df
		'''

		import string
		import pandas as pd
    
		docs = self.df['comment_text'].tolist()

		numbers = string.printable[0:9]
		letters = string.printable[10:62] #a-Z
		punc = string.printable[62:94] # !'"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
		chars = []

		for doc in docs:
			char = {}
			char['upper'] = 0
			char['lower'] = 0
			char['punc'] = 0
			char['number'] = 0

			char['len'] = len(doc)

			for c in doc:
				if c in letters:
					if c.isupper():
						char['upper'] += 1

					if c.islower():
						char['lower'] += 1
	                
				if c in punc:
					char['punc'] += 1

				if c in numbers:
					char['number'] += 1

			for key in char.keys():
				if key != 'len':
					char[key] = char[key] / char['len']

			chars.append(char)

		df_chars = pd.DataFrame(chars)

		self.character_analysis_columns_ = []
		for col in df_chars.columns.tolist():
			if verbose:
				ct = df_chars[col].describe()['count'].round(0)
				mean = df_chars[col].describe()['mean'].round(2)
				std = df_chars[col].describe()['std'].round(2)
				
				repstr = 'Character analysis: {}\t'.format(col) + \
				'Count: {}\tMean: {}\tStd: {}'.format(ct,mean,std)
				print(repstr)

			self.df['char_'+col] = df_chars[col].values
			self.character_analysis_columns_.append('char_'+col)

	def ck_spelling_toxicity(self, wordsdict='words_dictionary.json', 
		toxicwordsdict='toxic-ec.json', verbose=False):
		# iterates through each word token and records whether or not the word is spelled correctly (per big word list)
		# and identifies toxic words by category
		# also returns total number of tokens
		import json
		from nltk.tokenize import word_tokenize
		import spacy
		nlp = spacy.load('en')
		import pandas as pd
		
		allwords = load_json('{}/{}'.format(self.DATA_DIR, wordsdict))
		toxicwords = load_json('{}/{}'.format(self.DATA_DIR, toxicwordsdict))

		doc_scores = []
		docs = self.df['comment_text'].tolist()
		for doc in docs:
			doc_nlp = nlp(doc)
			
			
			# initialize scoring object
			doc_score = {}
			fields = [
				'spell_correct',
				'spell_incorrect',
				'curse',
				'offensive',
				'identity'
			]
			
			toxfields = fields[2:]
			
			for field in fields:
				doc_score[field] = 0
			
			doc_score['total'] = len(doc_nlp)
			
			for token in doc_nlp:
				if token.text in allwords.keys():
					doc_score['spell_correct'] += 1
				else:
					doc_score['spell_incorrect'] += 1
				
				for field in toxfields:
					if token.text in toxicwords[field]:
						doc_score[field] += 1
					
			for key in doc_score.keys(): # make proportional to total token counts
				if key != 'total':
					doc_score[key] = doc_score[key] / doc_score['total']

			doc_scores.append(doc_score)

		self.token_analysis_columns_ = []
		df_docscores = pd.DataFrame(doc_scores)
		for col in df_docscores.columns.tolist():
			if verbose:
				ct = df_docscores[col].describe()['count'].round(0)
				mean = df_docscores[col].describe()['mean'].round(2)
				std = df_docscores[col].describe()['std'].round(2)

				repstr = 'Token analysis: {}\t'.format(col) + \
				'Count: {}\tMean: {}\tStd: {}'.format(ct,mean,std)
				print(repstr)
			self.df['token_'+col] = df_docscores[col].values
			self.token_analysis_columns_.append('token_'+col)

	def create_token_correlations(self, verbose=False, force=False):
		'''
		Creates correlation data for heatmap
		'''
		
		if self.vectorize_mode_ not in ['count', 'bin', 'prop', 'tfidf']:
			print("Must be vectorized in mode 'count', 'prop', or 'bin'")
			return
		if self.vectorize_data_ != 'all':
			print("Only configured to run on 'all' data.")
			return

		if not force:
			try:
				self.df_corr_results.shape
				print('Correlation results already exist. Force creation with force=True')
				# object exists, break
				return
			except:
				pass

		import pandas as pd
		from scipy.sparse import issparse
		
		if issparse(self.X_):
			df_vocabmatrix = pd.SparseDataFrame(self.X_, columns=self.X_feature_names_).fillna(0)
		else:
			df_vocabmatrix = pd.DataFrame(self.X_, columns=self.X_feature_names_).fillna(0)

		flagdata = self.df[self.all_toxic_flags_].values

		from scipy.stats import pearsonr
		corr_results = []

		for i, ngram in enumerate(self.X_feature_names_):
			for j, flag in enumerate(self.all_toxic_flags_):
				corr, pval = pearsonr(df_vocabmatrix[ngram].values, flagdata[:,j])
				corr_results.append(
					{
						'ngram': ngram, 
						'flag': flag, 
						'corr': corr, 
						'pval': pval
					}
				)
				if verbose:
					if i % 1e2 == 0:
						print('{}:\t{}\tCorrelation: {}, p-val: {}'.format(
							ngram,flag,
							round(corr,2),
							round(pval,2))
						)
		self.df_corr_results = pd.DataFrame(corr_results)

	def make_corr_heatmap(self, n=25, ascending=False, figsize=(12,6), sortflag='any', save=True):
		'''
		Creates correlation heatmap for tokens, df_wordbag must be created
		'''
		try:
			self.df_corr_results.shape
			exists = True
		except:
			exists = False

		if exists:

			import seaborn as sns
			import matplotlib.pyplot as plt

			df = self.df_corr_results

			topcorr = df.loc[df['flag'] == sortflag,].sort_values(
				by='corr', axis=0, ascending=ascending).head(n)
			topcorr_ngrams = topcorr['ngram'].tolist()

			# make 2d matrix of correlations between ngram and flag
			res = df.loc[df['ngram'].isin(topcorr_ngrams),].pivot(
				'ngram','flag', 'corr').sort_values(
					by=sortflag, 
					axis=0, 
					ascending=ascending)
			
			fig, ax = plt.subplots(1,1, figsize=figsize)
			cbar_kws = dict(use_gridspec=False,location="top")
			sns.heatmap(res.transpose(), 
						cbar_kws=cbar_kws, 
						center=0, ax=ax)
			ax.set_ylabel('Flag or number of flags')

			if self.vectorize_mode_ == 'prop':
				vecmode = 'Proprtional'
			if self.vectorize_mode_ == 'count':
				vecmode = 'Count'
			if self.vectorize_mode_ == 'bin':
				vecmode = 'Binary'
			if self.vectorize_mode_ == 'tfidf':
				vecmode = 'TF-IDF'

			fig.suptitle('Correlation heatmap (Vecotrize mode: {})\nSort flag:{}'.format(vecmode, sortflag))
			#ax.set_xlabel('ngram token')
			if save:
				filestr = '{}/correlation heatmap vec_{} sortflag_{} asc_{}.png'.format(
					self.imgsavefolder_, self.vectorize_mode_, sortflag, ascending)
				fig.savefig(filestr)

			return fig, ax
		else:
			print('Must generate token dataframe with .create_token_correlations()')

	def prepare_stripplot(self, variable='token_total', figsize=(12,16), 
		alpha=.25, jitter=.4, linewidth=0, save=True, display=True, palette='xkcd'):



		allowedvars = self.character_analysis_columns_ + self.token_analysis_columns_
		if variable not in allowedvars:
			print('Variable must be in: {}'.format(', '.join(allowedvars)))
			return
		import matplotlib.pyplot as plt
		import seaborn as sns

		if palette == 'xkcd':
			xkcd_colors = ['denim blue', 'pale red']
			pal = sns.xkcd_palette(xkcd_colors)

		fig, ax = plt.subplots(nrows=len(self.toxic_flags_), ncols=1, figsize=figsize)

		
		for i, flag in enumerate(self.toxic_flags_):
			sns.stripplot(data=self.df, y=flag, x=variable, orient='h', 
				jitter=jitter, alpha=alpha, ax=ax[i], palette=pal)
			ax[i].xaxis.label.set_visible(False)

			if i == 0:
				ax[i].set_title('Comment character/token analysis for variable: {}'.format(variable))

		if save:
			fig.savefig('{}/eda_strip {}.png'.format(self.imgsavefolder_,variable))

		if display:

			return fig, ax

	def train_test_holdout_split(self, test=.15, holdout=.15, random_state=42):
		'''
		splits dataframe into train, test, holdout sets. Uses default random state for
		reproducability
		'''
		import numpy as np

		np.random.seed(random_state)
		random = np.random.rand(self.df.shape[0])

		train = 1 - (test + holdout)
		trainmask = random <= train
		testmask =  (random > train) & (random <= train + test)
		holdoutmask = (random > train + test) & (random <= train + test + holdout)

		self.df_train = self.df[trainmask]
		self.df_test = self.df[testmask]
		self.df_holdout = self.df[holdoutmask]
		self.data_split_exists = True



def pkl_dump(obj, file):
    import pickle
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

def pkl_load(file):
    import pickle
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    return obj

def set_basic_flags(df):
    # creates 'any' and 'none' flags
    # 'any' = 1 when any of the toxic flags are set
    # 'none = 1 when none of the toxic flags are set
    
    flags = df.columns.tolist()[2:]
    df['any'] = 0
    df['none'] = 0
    for flag in flags:
        df.loc[df[flag] == 1, 'any'] = 1
    
    df.loc[df['any'] == 0, 'none'] = 1
    return df

def load_json(file):
    # imports a JSON file into a dict
    import json
    with open(file, 'r') as f:
        obj = json.load(f)
    return obj
    
def dump_json(obj, file):
    import json
    with open(file, 'w') as f:
        json.dump(obj, f, indent=1)




def extend_contractions(docs, contrfile='data/contractions.xlsx'):
    # this function takes the list of contractions obtained from wikipedia 
    # https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions
    # and returns the extended words (ie. don't -> do not)

    import pandas as pd
    df_contractions = pd.read_excel(contrfile)
    contractions = dict(zip(df_contractions['Contraction'], df_contractions['Full']))
    docs_contraction_extended = []
    for doc in docs:
        for contr in contractions.keys():
            doc = doc.lower().replace(contr, contractions[contr])
        docs_contraction_extended.append(doc)
    return docs_contraction_extended

def repeated_char_remover(docs, num_chars=2):
	cleandocs = []
	for doc in docs:
		cleandoc = []
		for i, c in enumerate(doc):
			if i >= num_chars:
				same = True
				for n in range(1,num_chars+1):
					if not (doc[i] == doc[i-n]):
						same = False
						break
				if not same:
					cleandoc.append(c)
			else:
				cleandoc.append(c)
		cleandocs.append(''.join(cleandoc))
	return cleandocs

def replace_ip_in_docs(docs, rep='IP ADDRESS'):
	cleandocs = []
	for doc in docs:
		cleandocs.append(
			replace_ip(doc)
			)
	return cleandocs

def replace_ip(string, rep='IP ADDRESS'):
	import regex as re
	pat = re.compile("(([2][5][0-5]\.)|([2][0-4][0-9]\.)|([0-1]?[0-9]?[0-9]\.)){3}(([2][5][0-5])|([2][0-4][0-9])|([0-1]?[0-9]?[0-9]))")
	string = re.sub(pat, rep, string)
	return string

def strip_posessive(docs):
    # removes posessives from words
    cleandocs = []
    for doc in docs:
        cleandocs.append(doc.replace("'s", ''))
    return cleandocs

def strip_nonalpha(docs):
    # strips non alphabet characters from the string
    import string
    alpha = string.printable[10:52] + ' '

    cleandocs = []
    for doc in docs:
        docnew = []
        docstr = doc
        docstr = docstr.replace('\n', ' ')

        for c in docstr:
            if c in alpha:
                docnew.append(c)
        cleandocs.append(''.join(docnew))
    return cleandocs



def get_ids_by_flags(df, flags):
    # takes original df and a list of flags to extract, and returns the ids associated with those comments. 
    # Useful for filtering the word bag dataframe by comment type.
    
    cols = df.columns.tolist()
    tmp_flag_field = 'any_tmp_flag'
    
    df[tmp_flag_field] = 0
    
    for flag in flags:
        if flag in cols:
            df.loc[df[flag] == 1, tmp_flag_field] = 1
    
    ids = df.loc[df[tmp_flag_field] == 1, 'id'].tolist()
    return ids

def make_wordcloud(words, groupcolors=True, flag='severe_toxic',
	toxicwordsdict='data/toxic-ec.json', figsize=(8,6)):
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    from misc.colored_by_group import GroupedColorFunc
    
    toxicwords = load_json(toxicwordsdict)
    
    colors_key = {
        'red': 'offensive',
        'orange': 'identity',
        'yellow': 'curse'
    } 
    
    colors = {}
    default_color = 'grey'
    for col in colors_key.keys():
        colors[col] = [word for word in toxicwords[colors_key[col]].keys()]

    grouped_color_func = GroupedColorFunc(colors, default_color)
    words = ' '.join(words)
    wc = WordCloud(collocations=False).generate(words.lower())
    wc.recolor(color_func=grouped_color_func)
    
    #fig, ax = plt.figure(figsize=figsize)
    #ax.imshow(wc, interpolation="bilinear")
    plt.figure(figsize=figsize)
    plt.imshow(wc,interpolation='bilinear')
    plt.axis('off')
    plt.title('WordCloud for comments with flag: {}'.format(flag))
    plt.savefig('eda_images/wordcloud_{}.png'.format(flag))
    return plt

def convert_matrix_count_to_prop(matrix):
	import numpy as np
	matrix = matrix.astype(np.float32)
	for i in range(matrix.shape[0]):
		row = matrix[i,:]
		matrix[i,:] = matrix[i,:]/row.sum()
	return np.nan_to_num(matrix)



def spacy_tokenize(docs):
	import spacy
	nlp = spacy.load('en')
	

	docs_tokenized = []
	for doc in docs:
		doc_nlp = nlp(doc)
		doc_tokenized = []
		for token in doc_nlp:
			tk = token.text.lower().strip()
			if len(tk) > 0:
				doc_tokenized.append(token.text.lower().strip())

		docs_tokenized.append(doc_tokenized)

	return docs_tokenized

def ngram_phrases(docs, n=2):
	from gensim.models.phrases import Phrases, Phraser
	for i in range(n-1):
		phrases = Phrases(docs)
		ngrams = Phraser(phrases)
		docs = [ngrams[doc] for doc in docs]

	return docs

def make_tagged_documents(df, text_field, flags, phrase_length=2):
	from gensim.models.doc2vec import TaggedDocument
	doclabels = []
	ids = df['id'].tolist()
	docs = spacy_tokenize(df['comment_text'].tolist())
	docs = ngram_phrases(docs, phrase_length)

	for comment_id in ids:
		df_tmp = df[df.id == comment_id]
		doc = df_tmp[text_field].values[0]
		doclabel = []

		doclabel.append(comment_id)
		for flag in flags:
			if df_tmp[flag].values[0] == 1:
				doclabel.append(flag)

		doclabels.append(doclabel)

	tagged_docs = [TaggedDocument(doc, doclabels[i]) for i, doc in enumerate(docs)]

	return tagged_docs

def d2v_infer_vecs(model, docs):
	from gensim.models.doc2vec import Doc2Vec
	import numpy as np
	docs = spacy_tokenize(docs)

	docvecs = np.asarry([model.infer_vector(docs[i]) for i in len(docs)])
	return docvecs




