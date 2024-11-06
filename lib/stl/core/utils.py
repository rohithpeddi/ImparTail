import torch

from lib.stl.core.expression import Expression

LARGE_NUMBER = 1E4


def tensor_to_str(tensor):
	"""
    turn tensor into a string for printing
    """
	device = tensor.device.type
	req_grad = tensor.requires_grad
	if req_grad == False:
		return "input"
	tensor = tensor.detach()
	if device == "cuda":
		tensor = tensor.cpu()
	return str(tensor.numpy())


def convert_to_input_values(inputs):
	x_, y_ = inputs
	if isinstance(x_, Expression):
		assert x_.value is not None, "Input Expression does not have numerical values"
		x_ret = x_.value
	elif isinstance(x_, torch.Tensor):
		x_ret = x_
	elif isinstance(x_, tuple):
		x_ret = convert_to_input_values(x_)
	else:
		raise ValueError("First argument is an invalid input trace")
	
	if isinstance(y_, Expression):
		assert y_.value is not None, "Input Expression does not have numerical values"
		y_ret = y_.value
	elif isinstance(y_, torch.Tensor):
		y_ret = y_
	elif isinstance(y_, tuple):
		y_ret = convert_to_input_values(y_)
	else:
		raise ValueError("Second argument is an invalid input trace")
	
	return x_ret, y_ret