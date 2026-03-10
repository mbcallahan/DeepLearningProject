# Gradient Reversal layer implementation

import torch
from torch import nn
from torch.autograd import Function

class GradReversalFunc(Function):
    @staticmethod
    def forward(ctx, x, hp_lambda):
        ctx.hp_lambda = hp_lambda
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        hp_lambda = grad.new_tensor(ctx.hp_lambda)
        grad_x = -hp_lambda * grad
        grad_lam = None
        return grad_x, grad_lam
    
class GradRevLayer(nn.Module):
    def __init__(self, hp_lambda = 1.0):
        super(GradRevLayer, self).__init__()
        self.hp_lambda = hp_lambda

    def forward(self, x):
        return GradReversalFunc.apply(x, self.hp_lambda)