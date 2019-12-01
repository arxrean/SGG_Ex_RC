import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import pdb

class AttnGRUCell(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(AttnGRUCell, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.Wr = nn.Linear(input_size, hidden_size)
		self.Ur = nn.Linear(hidden_size, hidden_size)
		self.W = nn.Linear(input_size, hidden_size)
		self.U = nn.Linear(hidden_size, hidden_size)

		init.xavier_normal(self.Wr.state_dict()['weight'])
		init.xavier_normal(self.Ur.state_dict()['weight'])
		init.xavier_normal(self.W.state_dict()['weight'])
		init.xavier_normal(self.U.state_dict()['weight'])

	def forward(self, fact, hi_1, g):
		r_i = torch.sigmoid(self.Wr(fact) + self.Ur(hi_1))
		h_tilda = torch.tanh(self.W(fact) + r_i*self.U(hi_1))
		hi = g.unsqueeze(-1)*h_tilda + (1 - g.unsqueeze(-1))*hi_1

		return hi

class AttnGRU(nn.Module):
	def __init__(self, input_size, hidden_size):
		super(AttnGRU, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.AttnGRUCell = AttnGRUCell(input_size, hidden_size)

	def forward(self, facts, G):
		# facts.size() = (batch_size, num_sentences, embedding_length)
		# fact.size() = (batch_size, embedding_length=hidden_size)
		# G.size() = (batch_size, num_sentences)
		# g.size() = (batch_size, )

		h_0 = Variable(torch.zeros(self.hidden_size)).cuda()

		for sen in range(facts.size()[1]):
			fact = facts[:, sen, :]
			g = G[:, sen]
			if sen == 0: # Initialization for first sentence only
				hi_1 = h_0.unsqueeze(0).expand_as(fact)
			hi_1 = self.AttnGRUCell(fact, hi_1, g)
		C = hi_1 # Final hidden vector as the contextual vector used for updating memory

		return C