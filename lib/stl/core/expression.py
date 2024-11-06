import torch

from lib.stl.core.stlcg import LessThan, GreaterThan, Equal


class Expression(torch.nn.Module):
	"""
    Wraps a pytorch arithmetic operation, so that we can intercept and overload comparison operators.
    Expression allows us to express tensors using their names to make it easier to code up and read,
    but also keep track of their numeric values.
    """
	
	def __init__(self, name, value):
		super(Expression, self).__init__()
		self.name = name
		self.value = value
	
	def set_name(self, new_name):
		self.name = new_name
	
	def set_value(self, new_value):
		self.value = new_value
	
	def __neg__(self):
		return Expression(self.name, -self.value)
	
	def __add__(self, other):
		if isinstance(other, Expression):
			return Expression(self.name + '+' + other.name, self.value + other.value)
		else:
			return Expression(self.name + "+ other", self.value + other)
	
	def __radd__(self, other):
		return self.__add__(other)
	
	# No need for the case when "other" is an Expression, since that case will be handled by the regular add
	def __sub__(self, other):
		if isinstance(other, Expression):
			return Expression(self.name + '-' + other.name, self.value - other.value)
		else:
			return Expression(self.name + "-other", self.value - other)
	
	def __rsub__(self, other):
		return Expression(other - self.value)
	
	# No need for the case when "other" is an Expression, since that case will be handled by the regular sub
	def __mul__(self, other):
		if isinstance(other, Expression):
			return Expression(self.name + '*' + other.name, self.value * other.value)
		else:
			return Expression(self.name + "*other", self.value * other)
	
	def __rmul__(self, other):
		return self.__mul__(other)
	
	def __truediv__(self, a, b):
		# This is the new form required by Python 3
		numerator = a
		denominator = b
		numerator_name = 'num'
		denominator_name = 'denom'
		if isinstance(numerator, Expression):
			numerator_name = numerator.name
			numerator = numerator.value
		if isinstance(denominator, Expression):
			denominator_name = denominator.name
			denominator = denominator.value
		return Expression(numerator_name + '/' + denominator_name, numerator / denominator)
	
	# Comparators
	def __lt__(self, lhs, rhs):
		assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of LessThan needs to be a string or Expression"
		assert not isinstance(rhs, str), "RHS cannot be a string"
		return LessThan(lhs, rhs)
	
	def __le__(self, lhs, rhs):
		assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of LessThan needs to be a string or Expression"
		assert not isinstance(rhs, str), "RHS cannot be a string"
		return LessThan(lhs, rhs)
	
	def __gt__(self, lhs, rhs):
		assert isinstance(lhs, str) | isinstance(lhs,
		                                         Expression), "LHS of GreaterThan needs to be a string or Expression"
		assert not isinstance(rhs, str), "RHS cannot be a string"
		return GreaterThan(lhs, rhs)
	
	def __ge__(self, lhs, rhs):
		assert isinstance(lhs, str) | isinstance(lhs,
		                                         Expression), "LHS of GreaterThan needs to be a string or Expression"
		assert not isinstance(rhs, str), "RHS cannot be a string"
		return GreaterThan(lhs, rhs)
	
	@staticmethod
	def equal(lhs, rhs):
		assert isinstance(lhs, str) | isinstance(lhs, Expression), "LHS of Equal needs to be a string or Expression"
		assert not isinstance(rhs, str), "RHS cannot be a string"
		return Equal(lhs, rhs)
	
	@staticmethod
	def not_equal(lhs, rhs):
		raise NotImplementedError("Not supported yet")
	
	def __str__(self):
		return str(self.name)