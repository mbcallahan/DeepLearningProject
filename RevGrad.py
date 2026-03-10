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
    

layer = GradRevLayer(hp_lambda = 0.5)
x = torch.tensor([1.0,2.0,3.0], requires_grad=True)
y = layer(x)
loss = y.sum()
loss.backward()
print(x.grad)