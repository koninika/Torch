require 'nn'
require 'torch'

criterion = nn.AbsCriterion()
t1 = torch.rand(10, 10)
t2 = t1:clone()
t3 = t1 + 0.01

output1 = criterion:forward(t2, t1)
print(output1)

output2 = criterion:forward(t3, t1)
print(output2)
