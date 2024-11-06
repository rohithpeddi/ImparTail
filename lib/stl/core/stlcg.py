# -*- coding: utf-8 -*-
import numpy as np
import torch

from lib.stl.core.functions import Maxish, Minish
from lib.stl.core.stl_formula import STLFormula

'''
Important information:
- This has the option to use an arithmetic-geometric mean robustness metric: https://arxiv.org/pdf/1903.05186.pdf.
    The default is not to use it. But this is still being tested.
- Assume inputs are already reversed, but user does not need to worry about the indexing.
- "pscale" stands for "predicate scale" (not the scale used in maxish and minish)
- "scale" is the scale used in maxish and minish which Always, Eventually, Until, and Then uses.
- "time" variable when computing robustness: time=0 means the current time, t=1 means next time step.
    The reversal of the trace is accounted for inside the function, the user does not need to worry about this
- must specify subformula (no default string value)
'''


# TODO:
# - Run tests to ensure that "Expression" correctly overrides operators
# - Make a test for each temporal operator, and make sure that they all produce the expected output for at least one example trace
# - Implement log-barrier
# - option to choose type of padding

class TemporalOperator(STLFormula):
	"""
    Class to compute Eventually and Always. This builds a recurrent cell to perform dynamic programming
    subformula: The formula that the temporal operator is applied to.
    interval: either None (defaults to [0, np.inf]), [a, b] ( b < np.inf), [a, np.inf] (a > 0)
    NOTE: Assume that the interval is describing the INDICES of the desired time interval. The user is responsible for converting the time interval (in time units) into indices (integers) using knowledge of the time step size.
    """
	
	def __init__(self, subformula, interval=None):
		super(TemporalOperator, self).__init__()
		self.subformula = subformula
		self.interval = interval
		self._interval = [0, np.inf] if self.interval is None else self.interval
		self.rnn_dim = 1 if not self.interval else self.interval[
			-1]  # rnn_dim=1 if interval is [0, ∞) otherwise rnn_dim=end of interval
		if self.rnn_dim == np.inf:
			self.rnn_dim = self.interval[0]
		self.steps = 1 if not self.interval else self.interval[-1] - self.interval[
			0] + 1  # steps=1 if interval is [0, ∞) otherwise steps=length of interval
		self.operation = None
		# Matrices that shift a vector and add a new entry at the end.
		self.M = torch.tensor(np.diag(np.ones(self.rnn_dim - 1), k=1)).requires_grad_(False).float()
		self.b = torch.zeros(self.rnn_dim).unsqueeze(-1).requires_grad_(False).float()
		self.b[-1] = 1.0
	
	def _initialize_rnn_cell(self, x):
		"""
        x is [batch_size, time_dim, x_dim]
        initial rnn state is [batch_size, rnn_dim, x_dim]
        This requires padding on the signal. Currently, the default is to extend the last value.
        TODO: have option on this padding

        The initial hidden state is of the form (hidden_state, count). count is needed just for the case with self.interval=[0, np.inf) and distributed=True. Since we are scanning through the sigal and outputing the min/max values incrementally, the distributed min function doesn't apply. If there are multiple min values along the signal, the gradient will be distributed equally across them. Otherwise it will only apply to the value that occurs earliest as we scan through the signal (i.e., practically, the last value in the trace as we process the signal backwards).
        """
		raise NotImplementedError("_initialize_rnn_cell is not implemented")
	
	def _rnn_cell(self, x, hc, scale=-1, agm=False, distributed=False, **kwargs):
		"""
        x: rnn input [batch_size, 1, ...]
        h0: input rnn hidden state. The hidden state is either a tensor, or a tuple of tensors, depending on the interval chosen. Generally, the hidden state is of size [batch_size, rnn_dim,...]
        """
		raise NotImplementedError("_initialize_rnn_cell is not implemented")
	
	def _run_cell(self, x, scale, agm=False, distributed=False):
		"""
        Run the cell through the trace.
        """
		
		outputs = []
		states = []
		hc = self._initialize_rnn_cell(x)  # [batch_size, rnn_dim, x_dim]
		xs = torch.split(x, 1, dim=1)  # time_dim tuple
		time_dim = len(xs)
		for i in range(time_dim):
			o, hc = self._rnn_cell(xs[i], hc, scale, agm=agm, distributed=distributed)
			outputs.append(o)
			states.append(hc)
		return outputs, states
	
	def robustness_trace(self, inputs, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
		# Compute the robustness trace of the subformula and that is the input to the temporal operator graph.
		trace = self.subformula(inputs, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed,
		                        **kwargs)
		outputs, states = self._run_cell(trace, scale=scale, agm=agm, distributed=distributed)
		return torch.cat(outputs, dim=1)  # [batch_size, time_dim, ...]
	
	def _next_function(self):
		# next function is the input subformula
		return [self.subformula]


class Always(TemporalOperator):
	def __init__(self, subformula, interval=None):
		super(Always, self).__init__(subformula=subformula, interval=interval)
		self.operation = Minish()
		self.oper = "min"
	
	def _initialize_rnn_cell(self, x):
		"""
        Padding is with the last value of the trace
        """
		if x.is_cuda:
			self.M = self.M.cuda()
			self.b = self.b.cuda()
		h0 = torch.ones([x.shape[0], self.rnn_dim, x.shape[2]], device=x.device) * x[:, :1, :]
		count = 0.0
		# if self.interval is [a, np.inf), then the hidden state is a tuple (like in an LSTM)
		if (self._interval[1] == np.inf) & (self._interval[0] > 0):
			d0 = x[:, :1, :]
			return (d0, h0.to(x.device)), count
		
		return h0.to(x.device), count
	
	def _rnn_cell(self, x, hc, scale=-1, agm=False, distributed=False, **kwargs):
		"""
        x: rnn input [batch_size, 1, ...]
        hc=(h0, c) h0 is the input rnn hidden state  [batch_size, rnn_dim, ...]. c is the count. Initialized by self._initialize_rnn_cell
        """
		h0, c = hc
		if self.operation is None:
			raise Exception()
		# keeping track of all values that share the min value so the gradients can be distributed equally.
		if self.interval is None:
			if distributed:
				if x == h0:
					new_h = (h0 * c + x) / (c + 1)
					new_c = c + 1.0
				elif x < h0:
					new_h = x
					new_c = 1.0
				else:
					new_h = h0
					new_c = c
				state = (new_h, new_c)
				output = new_h
			else:
				input_ = torch.cat([h0, x], dim=1)  # [batch_size, rnn_dim+1, x_dim]
				output = self.operation(input_, scale, dim=1, keepdim=True, agm=agm)  # [batch_size, 1, x_dim]
				state = (output, None)
		else:  # self.interval is [a, np.inf)
			if (self._interval[1] == np.inf) & (self._interval[0] > 0):
				d0, h0 = h0
				dh = torch.cat([d0, h0[:, :1, :]], dim=1)  # [batch_size, 2, x_dim]
				output = self.operation(dh, scale, dim=1, keepdim=True, agm=agm,
				                        distributed=distributed)  # [batch_size, 1, x_dim]
				state = ((output, torch.matmul(self.M, h0) + self.b * x), None)
			else:  # self.interval is [a, b]
				state = (torch.matmul(self.M, h0) + self.b * x, None)
				h0x = torch.cat([h0, x], dim=1)  # [batch_size, rnn_dim+1, x_dim]
				input_ = h0x[:, :self.steps, :]  # [batch_size, self.steps, x_dim]
				output = self.operation(input_, scale, dim=1, keepdim=True, agm=agm,
				                        distributed=distributed)  # [batch_size, 1, x_dim]
		return output, state
	
	def __str__(self):
		return "◻ " + str(self._interval) + "( " + str(self.subformula) + " )"


class Eventually(TemporalOperator):
	def __init__(self, subformula='Eventually input', interval=None):
		super(Eventually, self).__init__(subformula=subformula, interval=interval)
		self.operation = Maxish()
		self.oper = "max"
	
	def _initialize_rnn_cell(self, x):
		"""
        Padding is with the last value of the trace
        """
		if x.is_cuda:
			self.M = self.M.cuda()
			self.b = self.b.cuda()
		h0 = torch.ones([x.shape[0], self.rnn_dim, x.shape[2]], device=x.device) * x[:, :1, :]
		count = 0.0
		if (self._interval[1] == np.inf) & (self._interval[0] > 0):
			d0 = x[:, :1, :]
			return (d0, h0.to(x.device)), count
		return h0.to(x.device), count
	
	def _rnn_cell(self, x, hc, scale=-1, agm=False, distributed=False, **kwargs):
		"""
        x: rnn input [batch_size, 1, ...]
        hc=(h0, c) h0 is the input rnn hidden state  [batch_size, rnn_dim, ...].
        c is the count. Initialized by self._initialize_rnn_cell
        """
		h0, c = hc
		if self.operation is None:
			raise Exception()
		
		if self.interval is None:
			if distributed:
				if x == h0:
					new_h = (h0 * c + x) / (c + 1)
					new_c = c + 1.0
				elif x > h0:
					new_h = x
					new_c = 1.0
				else:
					new_h = h0
					new_c = c
				state = (new_h, new_c)
				output = new_h
			else:
				input_ = torch.cat([h0, x], dim=1)  # [batch_size, rnn_dim+1, x_dim]
				output = self.operation(input_, scale, dim=1, keepdim=True, agm=agm)  # [batch_size, 1, x_dim]
				state = (output, None)
		else:  # self.interval is [a, np.inf)
			if (self._interval[1] == np.inf) & (self._interval[0] > 0):
				d0, h0 = h0
				dh = torch.cat([d0, h0[:, :1, :]], dim=1)  # [batch_size, 2, x_dim]
				output = self.operation(dh, scale, dim=1, keepdim=True, agm=agm,
				                        distributed=distributed)  # [batch_size, 1, x_dim]
				state = ((output, torch.matmul(self.M, h0) + self.b * x), None)
			else:  # self.interval is [a, b]
				state = (torch.matmul(self.M, h0) + self.b * x, None)
				h0x = torch.cat([h0, x], dim=1)  # [batch_size, rnn_dim+1, x_dim]
				input_ = h0x[:, :self.steps, :]  # [batch_size, self.steps, x_dim]
				output = self.operation(input_, scale, dim=1, keepdim=True, agm=agm,
				                        distributed=distributed)  # [batch_size, 1, x_dim]
		return output, state
	
	def __str__(self):
		return "♢ " + str(self._interval) + "( " + str(self.subformula) + " )"
