require 'nn'
require 'torch'

mlp = nn.ConcatTable()

mlp:add(nn.Linear(5,2))
mlp:add(nn.Linear(5,3))

pred = mlp:forward(torch.randn(5))
print(pred)

for i,k in ipairs(pred) do print(i,k) end
