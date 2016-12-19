--[[
require 'torch'
require 'nn'

input = {torch.randn(5,3), torch.randn(5,2)}
target = {torch.IntTensor{1,8}, torch.randn(2,10)}

criterion = nn.ParallelCriterion()
criterion:add(nn.MSECriterion()):add()
--]]

require 'nn'
require 'torch'


i1 = torch.rand(2,10)
i2 = (torch.randn(2,10))

t1 = torch.IntTensor{1,8}
t2 = i2:clone()

input = { i1, i2 }
target = { t1, t2 }
cec = nn.CrossEntropyCriterion()
abs = nn.AbsCriterion()
pc = nn.ParallelCriterion():add(cec, 0.0):add(abs, 1)
output = pc:forward(input, target)
print(output)

--[[ criterion = nn.AbsCriterion()
t1 = torch.rand(10, 10)
t2 = t1:clone()
t3 = t1 + 0.01

output1 = criterion:forward(t2, t1)
print(output1)

output2 = criterion:forward(t3, t1)
print(output2)

--]]
