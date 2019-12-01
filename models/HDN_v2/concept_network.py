import torch
import torch.nn as nn
import pdb
import json
import os
import os.path as osp
import re
import numpy as np

import requests
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .AGRU import AttnGRU

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors


class Concept_network(nn.Module):
	def __init__(self, opts=None, args=None):
		super(Concept_network, self).__init__()
		self.opts = opts
		self.args = args
		annotation_dir = self.opts['dir']
		obj_cats = json.load(open(osp.join(annotation_dir, 'objects.json')))
		self._object_classes = tuple(['__background__'] + obj_cats)
		self._object_class_to_ind = dict(
			zip(self.object_classes, range(self.num_object_classes)))

		# pdb.set_trace()
		if not os.path.exists(self.args.CON_base_dir):
			self.set_up_glove()
		self.get_concept_base()

		self.bi_gru = nn.GRU(input_size=300, hidden_size=300,
							 batch_first=True, bidirectional=True)

		# kg fusion
		self.W_q = nn.Linear(512, 300)
		self.W_1_2 = nn.Sequential(
			nn.Linear(300*4, 512),
			nn.Tanh(),
			nn.Linear(512, 1)
		)

		self.agru = AttnGRU(input_size=300, hidden_size=300)
		self.next_mt = nn.Linear(300*3, 300)

		self.update_obj = nn.Linear(300+512, 512)

	def forward(self, pooled_object_features, cls_score_object):
		pdb.set_trace()
		input_embeddings, valid_obj_idx = self.generate_embeddings(
			cls_score_object)
		pack_embeds = self.packed_embeddings(input_embeddings)

		fact_embeds = self.get_fact_embeds(pack_embeds)

		pre_m = None
		for _ in range(self.args.CON_T_m):
			pre_m = self.attention_based_knowledge_fusion(
				pooled_object_features[valid_obj_idx], fact_embeds, pre_m)
		obj_update = nn.functional.relu(self.update_obj(
			torch.cat([pooled_object_features[valid_obj_idx], pre_m], -1)))

		pooled_object_features_clone = pooled_object_features.clone()
		pooled_object_features_clone[valid_obj_idx] = obj_update

		return pooled_object_features_clone

	def attention_based_knowledge_fusion(self, pooled_object_features, fact_embeds, pre_m):
		q = torch.tanh(self.W_q(pooled_object_features))
		if pre_m is None:
			pre_m = q

		q_tile = q.unsqueeze(1)
		pre_m_tile = pre_m.unsqueeze(1)
		z = torch.cat([fact_embeds*q_tile, fact_embeds*pre_m_tile,
					   torch.abs(fact_embeds-q_tile), torch.abs(fact_embeds-pre_m_tile)], -1)

		g = nn.functional.softmax(self.W_1_2(z).squeeze(), -1)

		attention_res = self.dmn_attention(g, fact_embeds)

		mt = nn.functional.relu(self.next_mt(
			torch.cat([pre_m, attention_res, q], -1)))

		return mt

	def dmn_attention(self, g, fact_embeds, mode='agru'):
		if mode == 'sum':
			return torch.sum(fact_embeds*g.unsqueeze(-1), 1)
		elif mode == 'agru':
			return self.agru(facts=fact_embeds, G=g)

		return None

	def get_fact_embeds(self, pack_embeds):
		res = []
		for pack in pack_embeds:
			f_fact = self.bi_gru(pack)
			h_n = f_fact[1].view(2, self.args.CON_top_k, 300)
			res.append(torch.sum(h_n, 0))

		res = torch.stack(res)
		return res

	def packed_embeddings(self, input_embeddings):
		pack = []
		for e in input_embeddings:
			seq_lengths = torch.LongTensor([len(seq) for seq in e]).cuda()
			seq_tensor = torch.zeros(
				(len(e), seq_lengths.max(), 300)).cuda()
			for idx in range(len(e)):
				seq_tensor[idx, :seq_lengths[idx].item(), :] = torch.Tensor(
					e[idx]).cuda()

			packed_input = pack_padded_sequence(
				seq_tensor, seq_lengths.cpu().numpy(), batch_first=True, enforce_sorted=False)

			pack.append(packed_input)

		return pack

	def generate_embeddings(self, obj_outputs):
		input_objs = torch.argmax(obj_outputs, dim=-1)
		obj_with_rels5 = np.where(np.array([len(self.concept_base[k]) for k in sorted(
			self.concept_base.keys())]) >= self.args.CON_top_k)[0]+1
		valid_obj_idx = [i for i in range(input_objs.size(
			0)) if input_objs[i].item() in obj_with_rels5]
		input_valid_objs = input_objs[valid_obj_idx]

		valid_embeddings = [[t[0] for t in self.concept_base[v.item(
		)][:self.args.CON_top_k]] for v in input_valid_objs]

		# max_length = max([len(seq) for v in valid_embeddings for seq in v])
		return valid_embeddings, valid_obj_idx

	def get_concept_base(self):
		# load concept from web
		if not os.path.exists(self.args.CON_base_dir):
			print('get concept from web...')
			base_dict = {}
			for idx, c in enumerate(self.object_classes):
				if self.args and c == '__background__':
					continue

				# pdb.set_trace()
				obj = requests.get(
					'http://api.conceptnet.io/c/en/{}'.format(c)).json()
				edges = obj['edges']
				condidates = []
				for e in edges:
					start = e['start']
					rel = e['rel']
					end = e['end']

					if 'language' in start and start['language'] != 'en':
						continue
					if 'language' in rel and rel['language'] != 'en':
						continue
					if 'language' in end and end['language'] != 'en':
						continue

					try:
						rel['label'] = ' '.join(
							re.findall('[A-Z][^A-Z]*', rel['label']))

						rel_words = '{} {} {}'.format(
							start['label'], rel['label'], end['label'])

						rel_embed = []
						for word in rel_words.split(' '):
							rel_embed.append(self.glove_model[word])
						rel_embed = np.stack(rel_embed)

						condidates.append((rel_embed, e['weight']))
					except Exception as e:
						print(rel_words)
						continue

				base_dict[self._object_class_to_ind[c]] = sorted(
					condidates, key=lambda x: x[-1], reverse=True)

			np.save(self.args.CON_base_dir, base_dict)

		# pdb.set_trace()
		self.concept_base = np.load(self.args.CON_base_dir).item()

	def set_up_glove(self):
		if not os.path.exists(self.args.CON_glove_word2vec_root):
			print('generate glove word2vec...')
			glove_input_file = os.path.join(
				'/'.join(self.args.CON_glove_word2vec_root.split('/')[:-1]), 'glove.840B.300d.txt')
			word2vec_output_file = self.args.CON_glove_word2vec_root
			(count, dimensions) = glove2word2vec(
				glove_input_file, word2vec_output_file)

		glove_model = KeyedVectors.load_word2vec_format(
			self.args.CON_glove_word2vec_root, binary=False)

		self.glove_model = glove_model

	@property
	def object_classes(self):
		return self._object_classes

	@property
	def num_object_classes(self):
		return len(self._object_classes)
