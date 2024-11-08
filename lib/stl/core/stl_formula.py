import numpy as np
import torch

from lib.stl.core.expression import Expression
from lib.stl.core.functions import Maxish, Minish
from lib.stl.core.temporal_operator import Eventually, Always
from lib.stl.core.utils import convert_to_input_values, tensor_to_str


class STLFormula(torch.nn.Module):
	"""
    NOTE: All the inputs are assumed to be TIME REVERSED. The outputs are also TIME REVERSED
    All STL formulas have the following functions:
    robustness_trace: Computes the robustness trace.
    robustness: Computes the robustness value of the trace
    eval_trace: Computes the robustness trace and returns True in each entry if the robustness value is > 0
    eval: Computes the robustness value and returns True if the robustness value is > 0
    forward: The forward function of this STL_formula PyTorch module (default to the robustness_trace function)

    Inputs to these functions:
    trace: the input signal assumed to be TIME REVERSED. If the formula has two subformulas (e.g., And), then it is a tuple of the two inputs. An input can be a tensor of size [batch_size, time_dim,...], or an Expression with a .value (Tensor) associated with the expression.
    pscale: predicate scale. Default: 1
    scale: scale for the max/min function.  Default: -1
    keepdim: Output shape is the same as the input tensor shapes. Default: True
    agm: Use arithmetic-geometric mean. (In progress.) Default: False
    distributed: Use the distributed mean. Default: False
    """
	
	def __init__(self):
		super(STLFormula, self).__init__()
	
	def robustness_trace(self, trace, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
		raise NotImplementedError("robustness_trace not yet implemented")
	
	def robustness(self, inputs, time=0, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
		"""
        Extracts the robustness_trace value at the given time.
        Default: time=0 assuming this is the index for the NON-REVERSED trace. But the code will take it from the end since the input signal is TIME REVERSED.

        """
		return self.forward(inputs, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed,
		                    **kwargs)[:, -(time + 1), :].unsqueeze(1)
	
	def eval_trace(self, inputs, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
		"""
        The values in eval_trace are 0 or 1 (False or True)
        """
		return self.forward(inputs, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed,
		                    **kwargs) > 0
	
	def eval(self, inputs, time=0, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
		"""
        Extracts the eval_trace value at the given time.
        Default: time=0 assuming this is the index for the NON-REVERSED trace. But the code will take it from the end since the input signal is TIME REVERSED.
        """
		return self.eval_trace(inputs, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed,
		                       **kwargs)[:, -(time + 1), :].unsqueeze(1)  # [batch_size, time_dim, x_dim]
	
	def forward(formula, inputs, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
		"""
        Evaluates the robustness_trace given the input. The input is converted to the numerical value first.
        """
		if isinstance(inputs, Expression):
			assert inputs.value is not None, "Input Expression does not have numerical values"
			return formula.robustness_trace(inputs.value, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm,
			                                distributed=distributed, **kwargs)
		elif isinstance(inputs, torch.Tensor):
			return formula.robustness_trace(inputs, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm,
			                                distributed=distributed, **kwargs)
		elif isinstance(inputs, tuple):
			return formula.robustness_trace(convert_to_input_values(inputs), pscale=pscale, scale=scale,
			                                keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)
		else:
			raise ValueError("Not a invalid input trace")
	
	def __str__(self):
		raise NotImplementedError("__str__ not yet implemented")
	
	def __and__(phi, psi):
		return And(phi, psi)
	
	def __or__(phi, psi):
		return Or(phi, psi)
	
	def __invert__(phi):
		return Negation(phi)


class LessThan(STLFormula):
	"""
    lhs <= val where lhs is the signal, and val is the constant.
    lhs can be a string or an Expression
    val can be a float, int, Expression, or tensor. It cannot be a string.
    """
	
	def __init__(self, lhs='x', val='c'):
		super(LessThan, self).__init__()
		assert isinstance(lhs, str) | isinstance(lhs,
		                                         Expression), "LHS of expression needs to be a string (input name) or Expression"
		assert not isinstance(val, str), "value on the rhs cannot be a string"
		self.lhs = lhs
		self.val = val
		self.subformula = None
	
	def robustness_trace(self, trace, pscale=1.0, **kwargs):
		"""
        Computing robustness trace.
        pscale scales the robustness by a constant. Default pscale=1.
        """
		if isinstance(trace, Expression):
			trace = trace.value
		if isinstance(self.val, Expression):
			return (self.val.value - trace) * pscale
		else:
			return (self.val - trace) * pscale
	
	def _next_function(self):
		# expects self.lhs to be a string (used for visualizing the graph)
		# if isinstance(self.lhs, Expression):
		#     return [self.lhs.name, self.val]
		return [self.lhs, self.val]
	
	def __str__(self):
		lhs_str = self.lhs
		if isinstance(self.lhs, Expression):
			lhs_str = self.lhs.name
		
		if isinstance(self.val, str):  # could be a string if robustness_trace is never called
			return lhs_str + " <= " + self.val
		if isinstance(self.val, Expression):
			return lhs_str + " <= " + self.val.name
		if isinstance(self.val, torch.Tensor):
			return lhs_str + " <= " + tensor_to_str(self.val)
		# if self.value is a single number (e.g., int, or float)
		return lhs_str + " <= " + str(self.val)


class GreaterThan(STLFormula):
	"""
    lhs >= val where lhs is the signal, and val is the constant.
    lhs can be a string or an Expression
    val can be a float, int, Expression, or tensor. It cannot be a string.
    """
	
	def __init__(self, lhs='x', val='c'):
		super(GreaterThan, self).__init__()
		assert isinstance(lhs, str) | isinstance(lhs,
		                                         Expression), "LHS of expression needs to be a string (input name) or Expression"
		assert not isinstance(val, str), "value on the rhs cannot be a string"
		self.lhs = lhs
		self.val = val
		self.subformula = None
	
	def robustness_trace(self, trace, pscale=1.0, **kwargs):
		"""
        Computing robustness trace.
        pscale scales the robustness by a constant. Default pscale=1.
        """
		if isinstance(trace, Expression):
			trace = trace.value
		if isinstance(self.val, Expression):
			return (trace - self.val.value) * pscale
		else:
			return (trace - self.val) * pscale
	
	def _next_function(self):
		# expects self.lhs to be a string (used for visualizing the graph)
		# if isinstance(self.lhs, Expression):
		#     return [self.lhs.name, self.val]
		return [self.lhs, self.val]
	
	def __str__(self):
		lhs_str = self.lhs
		if isinstance(self.lhs, Expression):
			lhs_str = self.lhs.name
		
		if isinstance(self.val, str):  # could be a string if robustness_trace is never called
			return lhs_str + " >= " + self.val
		if isinstance(self.val, Expression):
			return lhs_str + " >= " + self.val.name
		if isinstance(self.val, torch.Tensor):
			return lhs_str + " >= " + tensor_to_str(self.val)
		# if self.value is a single number (e.g., int, or float)
		return lhs_str + " >= " + str(self.val)


class Equal(STLFormula):
	"""
    lhs == val where lhs is the signal, and val is the constant.
    lhs can be a string or an Expression
    val can be a float, int, Expression, or tensor. It cannot be a string.
    """
	
	def __init__(self, lhs='x', val='c'):
		super(Equal, self).__init__()
		assert isinstance(lhs, str) | isinstance(lhs,
		                                         Expression), "LHS of expression needs to be a string (input name) or Expression"
		assert not isinstance(val, str), "value on the rhs cannot be a string"
		self.lhs = lhs
		self.val = val
		self.subformula = None
	
	def robustness_trace(self, trace, pscale=1.0, **kwargs):
		"""
        Computing robustness trace.
        pscale scales the robustness by a constant. Default pscale=1.
        """
		if isinstance(trace, Expression):
			trace = trace.value
		if isinstance(self.val, Expression):
			return -torch.abs(trace - self.val.value) * pscale
		
		return -torch.abs(trace - self.val) * pscale
	
	def _next_function(self):
		# if isinstance(self.lhs, Expression):
		#     return [self.lhs.name, self.val]
		return [self.lhs, self.val]
	
	def __str__(self):
		lhs_str = self.lhs
		if isinstance(self.lhs, Expression):
			lhs_str = self.lhs.name
		
		if isinstance(self.val, str):  # could be a string if robustness_trace is never called
			return lhs_str + " == " + self.val
		if isinstance(self.val, Expression):
			return lhs_str + " == " + self.val.name
		if isinstance(self.val, torch.Tensor):
			return lhs_str + " == " + tensor_to_str(self.val)
		# if self.value is a single number (e.g., int, or float)
		return lhs_str + " == " + str(self.val)


class Negation(STLFormula):
	"""
    not Subformula
    """
	
	def __init__(self, subformula):
		super(Negation, self).__init__()
		self.subformula = subformula
	
	def robustness_trace(self, inputs, pscale=1, scale=-1, keepdim=True, distributed=False, **kwargs):
		return -self.subformula(inputs, pscale=pscale, scale=scale, keepdim=keepdim, distributed=distributed, **kwargs)
	
	def _next_function(self):
		# next function is actually input (traverses the graph backwards)
		return [self.subformula]
	
	def __str__(self):
		return "¬(" + str(self.subformula) + ")"


class Implies(STLFormula):
	"""
    Implies
    """
	
	def __init__(self, subformula1, subformula2):
		super(Implies, self).__init__()
		self.subformula1 = subformula1
		self.subformula2 = subformula2
		self.operation = Maxish()
	
	def robustness_trace(self, inputs, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
		x, y = inputs
		trace1 = self.subformula1(x, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed,
		                          **kwargs)
		trace2 = self.subformula2(y, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed,
		                          **kwargs)
		xx = torch.stack([-trace1, trace2], dim=-1)  # [batch_size, time_dim, ..., 2]
		return self.operation(xx, scale, dim=-1, keepdim=False, agm=agm,
		                      distributed=distributed)  # [batch_size, time_dim, ...]
	
	def _next_function(self):
		# next function is actually input (traverses the graph backwards)
		return [self.subformula1, self.subformula2]
	
	def __str__(self):
		return "(" + str(self.subformula1) + ") => (" + str(self.subformula2) + ")"


class And(STLFormula):
	"""
    inputs: tuple (x,y) where x and y are the inputs to each subformula respectively. x or y can also be a tuple if the subformula requires multiple inputs (e.g, ϕ₁(x) ∧ (ϕ₂(y) ∧ ϕ₃(z)) would have inputs=(x, (y,z)))
    trace1 and trace2 are size [batch_size, time_dim, x_dim]
    """
	
	def __init__(self, subformula1, subformula2):
		super(And, self).__init__()
		self.subformula1 = subformula1
		self.subformula2 = subformula2
		self.operation = Minish()
	
	@staticmethod
	def separate_and(formula, input_, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
		if formula.__class__.__name__ != "And":
			return formula(input_, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed,
			               **kwargs).unsqueeze(-1)
		else:
			return torch.cat([And.separate_and(formula.subformula1, input_[0], pscale=pscale, scale=scale,
			                                   keepdim=keepdim, agm=agm, distributed=distributed, **kwargs),
			                  And.separate_and(formula.subformula2, input_[1], pscale=pscale, scale=scale,
			                                   keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)], axis=-1)
	
	def robustness_trace(self, inputs, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
		xx = torch.cat([And.separate_and(self.subformula1, inputs[0], pscale=pscale, scale=scale, keepdim=keepdim,
		                                 agm=agm, distributed=distributed, **kwargs),
		                And.separate_and(self.subformula2, inputs[1], pscale=pscale, scale=scale, keepdim=keepdim,
		                                 agm=agm, distributed=distributed, **kwargs)], axis=-1)
		return self.operation(xx, scale, dim=-1, keepdim=False, agm=agm,
		                      distributed=distributed)  # [batch_size, time_dim, ...]
	
	def _next_function(self):
		# next function is actually input (traverses the graph backwards)
		return [self.subformula1, self.subformula2]
	
	def __str__(self):
		return "(" + str(self.subformula1) + ") ∧ (" + str(self.subformula2) + ")"


class Or(STLFormula):
	"""
    inputs: tuple (x,y) where x and y are the inputs to each subformula respectively. x or y can also be a tuple if the subformula requires multiple inputs (e.g, ϕ₁(x) ∨ (ϕ₂(y) ∨ ϕ₃(z)) would have inputs=(x, (y,z)))
    trace1 and trace2 are size [batch_size, time_dim, x_dim]
    """
	
	def __init__(self, subformula1, subformula2):
		super(Or, self).__init__()
		self.subformula1 = subformula1
		self.subformula2 = subformula2
		self.operation = Maxish()
	
	@staticmethod
	def separate_or(formula, input_, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
		if formula.__class__.__name__ != "Or":
			return formula(input_, pscale=pscale, scale=scale, keepdim=keepdim, agm=agm, distributed=distributed,
			               **kwargs).unsqueeze(-1)
		else:
			return torch.cat([Or.separate_or(formula.subformula1, input_[0], pscale=pscale, scale=scale,
			                                 keepdim=keepdim, agm=agm, distributed=distributed, **kwargs),
			                  Or.separate_or(formula.subformula2, input_[1], pscale=pscale, scale=scale,
			                                 keepdim=keepdim, agm=agm, distributed=distributed, **kwargs)], axis=-1)
	
	def robustness_trace(self, inputs, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
		xx = torch.cat([Or.separate_or(self.subformula1, inputs[0], pscale=pscale, scale=scale, keepdim=keepdim,
		                               agm=agm, distributed=distributed, **kwargs),
		                Or.separate_or(self.subformula2, inputs[1], pscale=pscale, scale=scale, keepdim=keepdim,
		                               agm=agm, distributed=distributed, **kwargs)], axis=-1)
		return self.operation(xx, scale, dim=-1, keepdim=False, agm=agm,
		                      distributed=distributed)  # [batch_size, time_dim, ...]
	
	def _next_function(self):
		# next function is actually input (traverses the graph backwards)
		return [self.subformula1, self.subformula2]
	
	def __str__(self):
		return "(" + str(self.subformula1) + ") ∨ (" + str(self.subformula2) + ")"


class Until(STLFormula):
	def __init__(self, subformula1="Until subformula1", subformula2="Until subformula2", interval=None, overlap=True):
		"""
        subformula1 U subformula2 (ϕ U ψ)
        This assumes that ϕ is always true before ψ becomes true.
        If overlap=True, then the last time step that ϕ is true, ψ starts being true. That is, sₜ ⊧ ϕ and sₜ ⊧ ψ.
        If overlap=False, when ϕ stops being true, ψ starts being true. That is sₜ ⊧ ϕ and sₜ+₁ ⊧ ψ, but sₜ ¬⊧ ψ
        """
		super(Until, self).__init__()
		self.subformula1 = subformula1
		self.subformula2 = subformula2
		self.interval = interval
		if overlap == False:
			self.subformula2 = Eventually(subformula=subformula2, interval=[0, 1])
	
	def robustness_trace(self, inputs, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
		"""
        trace1 is the robustness trace of ϕ
        trace2 is the robustness trace of ψ
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        """
		assert isinstance(self.subformula1, STLFormula), "Subformula1 needs to be an stl formula"
		assert isinstance(self.subformula2, STLFormula), "Subformula2 needs to be an stl formula"
		LARGE_NUMBER = 1E6
		interval = self.interval
		trace1 = self.subformula1(inputs[0], pscale=pscale, scale=scale, keepdim=keepdim, agm=agm,
		                          distributed=distributed, **kwargs)
		trace2 = self.subformula2(inputs[1], pscale=pscale, scale=scale, keepdim=keepdim, agm=agm,
		                          distributed=distributed, **kwargs)
		Alw = Always(subformula=Identity(name=str(self.subformula1)))
		minish = Minish()
		maxish = Maxish()
		LHS = trace2.unsqueeze(-1).repeat([1, 1, 1, trace2.shape[1]]).permute(0, 3, 2, 1)
		if interval == None:
			RHS = torch.ones_like(LHS) * -LARGE_NUMBER
			for i in range(trace2.shape[1]):
				RHS[:, i:, :, i] = Alw(trace1[:, i:, :])
			return maxish(
				minish(torch.stack([LHS, RHS], dim=-1), scale=scale, dim=-1, keepdim=False, agm=agm,
				       distributed=distributed),
				scale=scale, dim=-1, keepdim=False, agm=agm, distributed=distributed)
		elif interval[1] < np.Inf:  # [a, b] where b < ∞
			a = int(interval[0])
			b = int(interval[1])
			RHS = [torch.ones_like(trace1)[:, :b, :] * -LARGE_NUMBER]
			for i in range(b, trace2.shape[1]):
				A = trace2[:, i - b:i - a + 1, :].unsqueeze(-1)
				relevant = trace1[:, :i + 1, :]
				B = Alw(relevant.flip(1), scale=scale, keepdim=keepdim, distributed=distributed)[:, a:b + 1, :].flip(
					1).unsqueeze(-1)
				RHS.append(maxish(
					minish(torch.cat([A, B], dim=-1), dim=-1, scale=scale, keepdim=False, distributed=distributed),
					dim=1, scale=scale, keepdim=keepdim, distributed=distributed))
			return torch.cat(RHS, dim=1);
		else:
			a = int(interval[0])  # [a, ∞] where a < ∞
			RHS = [torch.ones_like(trace1)[:, :a, :] * -LARGE_NUMBER]
			for i in range(a, trace2.shape[1]):
				A = trace2[:, :i - a + 1, :].unsqueeze(-1)
				relevant = trace1[:, :i + 1, :]
				B = Alw(relevant.flip(1), scale=scale, keepdim=keepdim, distributed=distributed)[:, a:, :].flip(
					1).unsqueeze(-1)
				RHS.append(maxish(
					minish(torch.cat([A, B], dim=-1), dim=-1, scale=scale, keepdim=False, distributed=distributed),
					dim=1, scale=scale, keepdim=keepdim, distributed=distributed))
			return torch.cat(RHS, dim=1);
	
	def _next_function(self):
		# next function is actually input (traverses the graph backwards)
		return [self.subformula1, self.subformula2]
	
	def __str__(self):
		return "(" + str(self.subformula1) + ")" + " U " + "(" + str(self.subformula2) + ")"


class Then(STLFormula):
	
	def __init__(self, subformula1, subformula2, interval=None, overlap=True):
		"""
        subformula1 T subformula2 (ϕ U ψ)
        This assumes that ϕ is eventually true before ψ becomes true.
        If overlap=True, then the last time step that ϕ is true, ψ starts being true. That is, sₜ ⊧ ϕ and sₜ ⊧ ψ.
        If overlap=False, when ϕ stops being true, ψ starts being true. That is sₜ ⊧ ϕ and sₜ+₁ ⊧ ψ, but sₜ ¬⊧ ψ
        """
		super(Then, self).__init__()
		self.subformula1 = subformula1
		self.subformula2 = subformula2
		self.interval = interval
		if overlap == False:
			self.subformula2 = Eventually(subformula=subformula2, interval=[0, 1])
	
	def robustness_trace(self, inputs, pscale=1, scale=-1, keepdim=True, agm=False, distributed=False, **kwargs):
		"""
        trace1 is the robustness trace of ϕ
        trace2 is the robustness trace of ψ
        trace1 and trace2 are size [batch_size, time_dim, x_dim]
        """
		assert isinstance(self.subformula1, STLFormula), "Subformula1 needs to be an stl formula"
		assert isinstance(self.subformula2, STLFormula), "Subformula2 needs to be an stl formula"
		LARGE_NUMBER = 1E6
		interval = self.interval
		trace1 = self.subformula1(inputs[0])
		trace2 = self.subformula2(inputs[1])
		Ev = Eventually(subformula=Identity(name=str(self.subformula1)))
		minish = Minish()
		maxish = Maxish()
		LHS = trace2.unsqueeze(-1).repeat([1, 1, 1, trace2.shape[1]]).permute(0, 3, 2, 1)
		RHS = torch.ones_like(LHS) * -LARGE_NUMBER
		if interval == None:
			for i in range(trace2.shape[1]):
				RHS[:, i:, :, i] = Ev(trace1[:, i:, :])
			return maxish(
				minish(torch.stack([LHS, RHS], dim=-1), scale=scale, dim=-1, keepdim=False, agm=agm,
				       distributed=distributed),
				scale=scale, dim=-1, keepdim=False, agm=agm, distributed=distributed)
		elif interval[1] < np.Inf:  # [a, b] where b < ∞
			a = int(interval[0])
			b = int(interval[1])
			RHS = [torch.ones_like(trace1)[:, :b, :] * -LARGE_NUMBER]
			for i in range(b, trace2.shape[1]):
				A = trace2[:, i - b:i - a + 1, :].unsqueeze(-1)
				relevant = trace1[:, :i + 1, :]
				B = Ev(relevant.flip(1), scale=scale, keepdim=keepdim, distributed=distributed)[:, a:b + 1, :].flip(
					1).unsqueeze(-1)
				RHS.append(maxish(
					minish(torch.cat([A, B], dim=-1), dim=-1, scale=scale, keepdim=False, distributed=distributed),
					dim=1, scale=scale, keepdim=keepdim, distributed=distributed))
			return torch.cat(RHS, dim=1);
		else:
			a = int(interval[0])  # [a, ∞] where a < ∞
			RHS = [torch.ones_like(trace1)[:, :a, :] * -LARGE_NUMBER]
			for i in range(a, trace2.shape[1]):
				A = trace2[:, :i - a + 1, :].unsqueeze(-1)
				relevant = trace1[:, :i + 1, :]
				B = Ev(relevant.flip(1), scale=scale, keepdim=keepdim, distributed=distributed)[:, a:, :].flip(
					1).unsqueeze(-1)
				RHS.append(maxish(
					minish(torch.cat([A, B], dim=-1), dim=-1, scale=scale, keepdim=False, distributed=distributed),
					dim=1, scale=scale, keepdim=keepdim, distributed=distributed))
			return torch.cat(RHS, dim=1);  # [batch_size, time_dim, x_dim]
	
	def _next_function(self):
		# next function is actually input (traverses the graph backwards)
		return [self.subformula1, self.subformula2]
	
	def __str__(self):
		return "(" + str(self.subformula1) + ")" + " T " + "(" + str(self.subformula2) + ")"


class Integral1d(STLFormula):
	def __init__(self, subformula, interval=None):
		super(Integral1d, self).__init__()
		self.subformula = subformula
		self.padding_size = None if interval is None else interval[1]
		self.interval = interval
		if interval is not None:
			kernel = interval[1] - interval[0] + 1
			self.conv = torch.nn.Conv1d(1, 1, kernel, padding=0, bias=False)
			for param in self.conv.parameters():
				param.requires_grad = False
			self.conv.weight /= self.conv.weight
	
	def construct_padding(self, padding_type, custom_number=100):
		if self.padding_size is not None:
			if padding_type == "zero":
				return torch.zeros([1, self.padding_size, 1])
			elif padding_type == "custom":
				return torch.ones([1, self.padding_size, 1]) * custom_number
			elif padding_type == "same":
				return torch.ones([1, self.padding_size, 1])
		else:
			return None
	
	def robustness_trace(self, inputs, pscale=1, scale=-1, keepdim=False, use_relu=False, padding_type="same",
	                     custom_number=100, integration_scheme="riemann", **kwargs):
		
		subformula_trace = self.subformula(inputs, pscale=pscale, scale=scale, keepdim=keepdim, **kwargs)
		
		if self.interval is not None:
			if integration_scheme == "trapz":
				self.conv.weight[:, :, 0] /= 2
				self.conv.weight[:, :, -1] /= 2
			padding = self.construct_padding(padding_type, custom_number)
			if subformula_trace.is_cuda:
				padding = padding.cuda()
			if padding_type == "same":
				padding = padding * subformula_trace[:, 0, :]
			signal = torch.cat([padding, subformula_trace], dim=1).transpose(1, 2)
			if use_relu:
				return self.conv(torch.relu(signal)).transpose(1, 2)[:, :subformula_trace.shape[1], :]
			else:
				return self.conv(signal).transpose(1, 2)[:, :subformula_trace.shape[1], :]
		else:
			if integration_scheme == "trapz":
				pad = torch.zeros_like(subformula_trace[:, :1, :])
				if subformula_trace.is_cuda:
					pad = padding.cuda()
				signal = torch.cat([pad, (subformula_trace[:, :-1, :] + subformula_trace[:, 1:, :]) / 2], dim=1)
			else:
				signal = subformula_trace
			if use_relu == True:
				return torch.cumsum(torch.relu(signal), dim=1)
			else:
				return torch.cumsum(signal, dim=1)
	
	def _next_function(self):
		# next function is actually input (traverses the graph backwards)
		return [self.subformula]
	
	def __str__(self):
		return "I" + str(self.interval) + "(" + str(self.subformula) + ")"


class Identity(STLFormula):
	
	def __init__(self, name='x'):
		super(Identity, self).__init__()
		self.name = name
	
	def robustness_trace(self, trace, pscale=1, **kwargs):
		return trace * pscale
	
	def _next_function(self):
		return []
	
	def __str__(self):
		return "%s" % self.name