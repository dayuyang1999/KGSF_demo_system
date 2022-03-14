from html import entities
import numpy as np
from tqdm import tqdm
from math import exp
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import signal
import json
import argparse
from nltk import word_tokenize
import pickle as pkl
from dataset_test import dataset,CRSdataset
from model import CrossModel
import torch.nn as nn
from torch import optim
import torch
try:
	import torch.version
	import torch.distributed as dist
	TORCH_AVAILABLE = True
except ImportError:
	TORCH_AVAILABLE = False
from nltk.translate.bleu_score import sentence_bleu

def is_distributed():
	"""
	Returns True if we are in distributed mode.
	"""
	return TORCH_AVAILABLE and dist.is_available() and dist.is_initialized()

def setup_args():
	train = argparse.ArgumentParser()
	train.add_argument("-test","--test",type=bool,default=False)
	train.add_argument("-max_c_length","--max_c_length",type=int,default=256)
	train.add_argument("-max_r_length","--max_r_length",type=int,default=30)
	train.add_argument("-batch_size","--batch_size",type=int,default=1)
	train.add_argument("-max_count","--max_count",type=int,default=5)
	train.add_argument("-use_cuda","--use_cuda",type=bool,default=True)
	train.add_argument("-load_dict","--load_dict",type=str,default=None)
	train.add_argument("-learningrate","--learningrate",type=float,default=1e-3)
	train.add_argument("-optimizer","--optimizer",type=str,default='adam')
	train.add_argument("-momentum","--momentum",type=float,default=0)
	train.add_argument("-is_finetune","--is_finetune",type=bool,default=False)
	train.add_argument("-embedding_type","--embedding_type",type=str,default='random')
	train.add_argument("-epoch","--epoch",type=int,default=30)
	train.add_argument("-gpu","--gpu",type=str,default='0,1')
	train.add_argument("-gradient_clip","--gradient_clip",type=float,default=0.1)
	train.add_argument("-embedding_size","--embedding_size",type=int,default=300)

	train.add_argument("-n_heads","--n_heads",type=int,default=2)
	train.add_argument("-n_layers","--n_layers",type=int,default=2)
	train.add_argument("-ffn_size","--ffn_size",type=int,default=300)

	train.add_argument("-dropout","--dropout",type=float,default=0.1)
	train.add_argument("-attention_dropout","--attention_dropout",type=float,default=0.0)
	train.add_argument("-relu_dropout","--relu_dropout",type=float,default=0.1)

	train.add_argument("-learn_positional_embeddings","--learn_positional_embeddings",type=bool,default=False)
	train.add_argument("-embeddings_scale","--embeddings_scale",type=bool,default=True)

	train.add_argument("-n_entity","--n_entity",type=int,default=64368)
	train.add_argument("-n_relation","--n_relation",type=int,default=214)
	train.add_argument("-n_concept","--n_concept",type=int,default=29308)
	train.add_argument("-n_con_relation","--n_con_relation",type=int,default=48)
	train.add_argument("-dim","--dim",type=int,default=128)
	train.add_argument("-n_hop","--n_hop",type=int,default=2)
	train.add_argument("-kge_weight","--kge_weight",type=float,default=1)
	train.add_argument("-l2_weight","--l2_weight",type=float,default=2.5e-6)
	train.add_argument("-n_memory","--n_memory",type=float,default=32)
	train.add_argument("-item_update_mode","--item_update_mode",type=str,default='0,1')
	train.add_argument("-using_all_hops","--using_all_hops",type=bool,default=True)
	train.add_argument("-num_bases", "--num_bases", type=int, default=8)

	return train

class TrainLoop_fusion_gen():
	def __init__(self, opt, is_finetune):
		self.opt=opt
		self.entity2entityId=pkl.load(open('data/entity2entityId.pkl','rb'))
		self.id2entity=pkl.load(open('data/id2entity.pkl','rb'))

		self.dict = json.load(open('word2index_redial.json', encoding='utf-8'))
		self.index2word={self.dict[key]:key for key in self.dict}

		self.batch_size=self.opt['batch_size']
		self.epoch=self.opt['epoch']

		self.use_cuda=opt['use_cuda']
		if opt['load_dict']!=None:
			self.load_data=True
		else:
			self.load_data=False

		self.movie_ids = pkl.load(open("data/movie_ids.pkl", "rb"))
		# Note: we cannot change the type of metrics ahead of time, so you
		# should correctly initialize to floats or ints here

		self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"count":0}
		self.metrics_gen={"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}

		self.build_model(is_finetune=True)

		if opt['load_dict'] is not None:
			# load model parameters if available
			print('[ Loading existing model params from {} ]'
				  ''.format(opt['load_dict']))
			states = self.model.load(opt['load_dict'])
		else:
			states = {}

		self.init_optim(
			[p for p in self.model.parameters() if p.requires_grad],
			optim_states=states.get('optimizer'),
			saved_optim_type=states.get('optimizer_type')
		)

	def build_model(self,is_finetune):
		self.model = CrossModel(self.opt, self.dict, is_finetune)
		if self.opt['embedding_type'] != 'random':
			pass
		if self.use_cuda:
			self.model.cuda()



	def val(self, user_input):
		self.metrics_gen={"ppl":0,"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}
		self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"gate":0,"count":0,'gate_count':0}
		self.model.eval()
		self.model.load_model(True)
		val_dataset = dataset(user_input, self.opt)
		val_set=CRSdataset(val_dataset.data_process(True),self.opt['n_entity'],self.opt['n_concept'])
		val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
														   batch_size=self.batch_size,
														   shuffle=False)
		inference_sum=[]
		golden_sum=[]
		context_sum=[]
		losses=[]
		recs=[]
		for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec in val_dataset_loader:
			with torch.no_grad():
				seed_sets = []
				batch_size = context.shape[0]
				for b in range(batch_size):
					seed_set = entity[b].nonzero().view(-1).tolist()
					seed_sets.append(seed_set)
				_, _, _, _, gen_loss, mask_loss, info_db_loss, info_con_loss = self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie, concept_vec, db_vec, entity_vector.cuda(), rec, test=False)
				scores, preds, rec_scores, rec_loss, _, mask_loss, info_db_loss, info_con_loss = self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie, concept_vec, db_vec, entity_vector.cuda(), rec, test=True, maxlen=20, bsz=batch_size)

			golden_sum.extend(self.vector2sentence(response.cpu()))
			inference_sum.extend(self.vector2sentence(preds.cpu()))
			context_sum.extend(self.vector2sentence(context.cpu()))
			recs.extend(rec.cpu())
			losses.append(torch.mean(gen_loss))
			#exit()
			c= ' '.join(self.vector2sentence(context.cpu())[0])+ "\n"
			r= ' '.join(self.vector2sentence(preds.cpu())[0])+ "\n"
			token_text = word_tokenize(r)
			num=0
			movies_id=[]
			while num<len(token_text):
				if token_text[num]=='@' and num+1<len(token_text):
					movie= token_text[num+1]
					movies_id.append(movie)
					token_text[num]= ''
					entity = self.id2entity[int(movie.strip())]
					print("entity in recommender's turn: ",  entity)
					try:
						a= self.entity2entityId[entity]
						f3=open('entity.txt','a',encoding='utf-8')
						f3.write(str(a)+"\n")
						f3.close()
					except:
						pass
					if entity== None:
						pass
					else:
						token_text[num+1]= entity
					num+=2
				else:
					num+=1
			print("recommender: "+ ' '.join(token_text))
			f=open('output_test2.txt','a',encoding='utf-8')
			f.write(' '.join(token_text))
			f.close()
		

	def vector2sentence(self,batch_sen):
		sentences=[]
		for sen in batch_sen.numpy().tolist():
			sentence=[]
			for word in sen:
				if word>3:
					sentence.append(self.index2word[word])
				elif word==3:
					sentence.append('_UNK_')
			sentences.append(sentence)
		return sentences

	@classmethod
	def optim_opts(self):
		"""
		Fetch optimizer selection.

		By default, collects everything in torch.optim, as well as importing:
		- qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

		Override this (and probably call super()) to add your own optimizers.
		"""
		# first pull torch.optim in
		optims = {k.lower(): v for k, v in optim.__dict__.items()
				  if not k.startswith('__') and k[0].isupper()}
		try:
			import apex.optimizers.fused_adam as fused_adam
			optims['fused_adam'] = fused_adam.FusedAdam
		except ImportError:
			pass

		try:
			# https://openreview.net/pdf?id=S1fUpoR5FQ
			from qhoptim.pyt import QHM, QHAdam
			optims['qhm'] = QHM
			optims['qhadam'] = QHAdam
		except ImportError:
			# no QHM installed
			pass

		return optims

	def init_optim(self, params, optim_states=None, saved_optim_type=None):
		"""
		Initialize optimizer with model parameters.

		:param params:
			parameters from the model

		:param optim_states:
			optional argument providing states of optimizer to load

		:param saved_optim_type:
			type of optimizer being loaded, if changed will skip loading
			optimizer states
		"""

		opt = self.opt

		# set up optimizer args
		lr = opt['learningrate']
		kwargs = {'lr': lr}
		kwargs['amsgrad'] = True
		kwargs['betas'] = (0.9, 0.999)

		optim_class = self.optim_opts()[opt['optimizer']]
		self.optimizer = optim_class(params, **kwargs)

	def backward(self, loss):
		"""
		Perform a backward pass. It is recommended you use this instead of
		loss.backward(), for integration with distributed training and FP16
		training.
		"""
		loss.backward()

	def update_params(self):
		"""
		Perform step of optimization, clipping gradients and adjusting LR
		schedule if needed. Gradient accumulation is also performed if agent
		is called with --update-freq.

		It is recommended (but not forced) that you call this in train_step.
		"""
		update_freq = 1
		if update_freq > 1:
			# we're doing gradient accumulation, so we don't only want to step
			# every N updates instead
			self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
			if self._number_grad_accum != 0:
				return

		if self.opt['gradient_clip'] > 0:
			torch.nn.utils.clip_grad_norm_(
				self.model.parameters(), self.opt['gradient_clip']
			)

		self.optimizer.step()

	def zero_grad(self):
		"""
		Zero out optimizer.

		It is recommended you call this in train_step. It automatically handles
		gradient accumulation if agent is called with --update-freq.
		"""
		self.optimizer.zero_grad()

if __name__ == '__main__':
	args=setup_args().parse_args()
	#print(vars(args))
	i= input("user: ")
	contexts= []
	entities= []
	file = open("context_test2.txt","r+")
	file.truncate(0)
	file.close()
	file = open("output_test2.txt","r+")
	file.truncate(0)
	file.close()
	file = open("entity.txt","r+")
	file.truncate(0)
	file.close()
	while i!= "quit":
		loop=TrainLoop_fusion_gen(vars(args), is_finetune=True)
		met=loop.val(i)
		i= input("user: ")



	#print(met)
