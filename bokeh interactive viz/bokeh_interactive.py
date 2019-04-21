from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider
from bokeh.models.widgets import RangeSlider, CheckboxButtonGroup, Select
from bokeh.plotting import figure
from bokeh.themes import Theme
from bokeh.io import curdoc
import pandas as pd

df = pd.read_csv('toxic_sample.csv')

figsize = (1200,800)

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

buttonlabels = [
	'toxic',
	 'severe_toxic',
	 'obscene',
	 'threat',
	 'insult',
	 'identity_hate',
	 'none'
 ]

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
		p.title.text = 't-SNE visualization of {} observations'.format(df.shape[0])
	else:
		x = 'mf_d2v_X0'
		y = 'mf_d2v_X1'
		p.title.text = 'Doc2Vec visualization of {} observations'.format(df.shape[0])
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

n_obs = df.shape[0]
manif = 't-SNE'
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

layout = column(
	slider,
	select,
	buttons,
	p
)
curdoc().add_root(layout)
curdoc().title = 'Toxic comment EDA'

#doc.add_root(column(slider, select, buttons, p))