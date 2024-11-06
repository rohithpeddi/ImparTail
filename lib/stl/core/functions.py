import torch

from lib.stl.core.expression import Expression


class Maxish(torch.nn.Module):
	"""
    Function to compute the max, or softmax, or other variants of the max function.
    """
	
	def __init__(self, name="Maxish input"):
		super(Maxish, self).__init__()
		self.input_name = name
	
	def forward(self, x, scale, dim=1, keepdim=True, agm=False, distributed=False):
		"""
        x is of size [batch_size, T, ...] where T is typically the trace length.

        if scale <= 0, then the true max is used, otherwise, the softmax is used.

        dim is the dimension which the max is applied over. Default: 1

        keepdim keeps the dimension of the input tensor. Default: True

        agm is the arithmetic-geometric mean. Currently in progress. If some elements are >0, output is the average of those elements. If all the elements <= 0, output is -ᵐ√(Πᵢ (1 - ηᵢ)) + 1. scale doesn't play a role here except to switch between the using the AGM or true robustness value (scale <=0). Default: False

        distributed addresses the case when there are multiple max values. As max is poorly defined in these cases, PyTorch (randomly?) selects one of the max values only. If distributed=True and scale <=0 then it will average over the max values and split the gradients equally. Default: False
        """
		
		if isinstance(x, Expression):
			assert x.value is not None, "Input Expression does not have numerical values"
			x = x.value
		if scale > 0:
			if agm == True:
				if torch.gt(x, 0).any():
					return x[torch.gt(x, 0)].reshape(*x.shape[:-1], -1).mean(dim=dim, keepdim=keepdim)
				else:
					return -torch.log(1 - x).mean(dim=dim, keepdim=keepdim).exp() + 1
			else:
				# return torch.log(torch.exp(x*scale).sum(dim=dim, keepdim=keepdim))/scale
				return torch.logsumexp(x * scale, dim=dim, keepdim=keepdim) / scale
		else:
			if distributed:
				return self.distributed_true_max(x, dim=dim, keepdim=keepdim)
			else:
				return x.max(dim, keepdim=keepdim)[0]
	
	@staticmethod
	def distributed_true_max(xx, dim=1, keepdim=True):
		"""
        If there are multiple values that share the same max value, the max value is computed by taking the mean of all those values.
        This will ensure that the gradients of max(xx, dim=dim) will be distributed evenly across all values, rather than just the one value.
        """
		m, mi = torch.max(xx, dim, keepdim=True)
		inds = xx == m
		return torch.where(inds, xx, xx * 0).sum(dim, keepdim=keepdim) / inds.sum(dim, keepdim=keepdim)
	
	def _next_function(self):
		"""
        This is used for the graph visualization to keep track of the parent node.
        """
		return [str(self.input_name)]


class Minish(torch.nn.Module):
	"""
    Function to compute the min, or softmin, or other variants of the min function.
    """
	
	def __init__(self, name="Minish input"):
		super(Minish, self).__init__()
		self.input_name = name
	
	def forward(self, x, scale, dim=1, keepdim=True, agm=False, distributed=False):
		"""
        x is of size [batch_size, T, ...] where T is typically the trace length.

        if scale <= 0, then the true max is used, otherwise, the softmax is used.

        dim is the dimension which the max is applied over. Default: 1

        keepdim keeps the dimension of the input tensor. Default: True

        agm is the arithmetic-geometric mean. Currently in progress. If all elements are >0, output is ᵐ√(Πᵢ (1 + ηᵢ)) - 1.If some the elements <= 0, output is the average of those negative values. scale doesn't play a role here except to switch between the using the AGM or true robustness value (scale <=0).

        distributed addresses the case when there are multiple max values. As max is poorly defined in these cases, PyTorch (randomly?) selects one of the max values only. If distributed=True and scale <=0 then it will average over the max values and split the gradients equally. Default: False
        """
		
		if isinstance(x, Expression):
			assert x.value is not None, "Input Expression does not have numerical values"
			x = x.value
		
		if scale > 0:
			if agm == True:
				if torch.gt(x, 0).all():
					return torch.log(1 + x).mean(dim=dim, keepdim=keepdim).exp() - 1
				else:
					# return x[torch.lt(x, 0)].reshape(*x.shape[:-1], -1).mean(dim=dim, keepdim=keepdim)
					return (torch.lt(x, 0) * x).sum(dim, keepdim=keepdim) / torch.lt(x, 0).sum(dim, keepdim=keepdim)
			else:
				# return -torch.log(torch.exp(-x*scale).sum(dim=dim, keepdim=keepdim))/scale
				return -torch.logsumexp(-x * scale, dim=dim, keepdim=keepdim) / scale
		
		else:
			if distributed:
				return self.distributed_true_min(x, dim=dim, keepdim=keepdim)
			else:
				return x.min(dim, keepdim=keepdim)[0]
	
	@staticmethod
	def distributed_true_min(xx, dim=1, keepdim=True):
		"""
        If there are multiple values that share the same max value, the min value is computed by taking the mean of all those values.
        This will ensure that the gradients of min(xx, dim=dim) will be distributed evenly across all values, rather than just the one value.
        """
		m, mi = torch.min(xx, dim, keepdim=True)
		inds = xx == m
		return torch.where(inds, xx, xx * 0).sum(dim, keepdim=keepdim) / inds.sum(dim, keepdim=keepdim)
	
	def _next_function(self):
		"""
        This is used for the graph visualization to keep track of the parent node.
        """
		return [str(self.input_name)]