require 'torch'
require 'nn'

input = {torch.randn(5,3), torch.randn(5,2)}
target = {}

criterion = nn.ParallelCriterion()
criterion:add()
