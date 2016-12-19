-- Include
-- print('Included:')

require 'torch'
require 'nn'

-- Load Data
local mnist = require 'mnist'
local train_dataset = mnist.traindataset()
local test_dataset = mnist.testdataset()

trainset = {
	data = train_dataset.data:double(),
	labels = train_dataset.label
}

-- labels can be from 1 to 10 because we have 10 classes and 0 is not allowed
-- so add one to each class label ( 0-9 becomes 1-10)
i = 0 
trainset.labels:apply(function()
	i = i+1
	trainset.labels[i] = trainset.labels[i] + 1
	return trainset.labels[i]

end)

testset = {
	data = test_dataset.data:double(),
	labels = test_dataset.label
}

-- print('trainset size:')
-- print(trainset.size)
-- print('testset size:')
-- print(trainset.size)


-- Define model
-- classes = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

model = nn.Sequential()

model:add(nn.Reshape(28*28))
model:add(nn.Linear(28*28, 30))
model:add(nn.Tanh())
model:add(nn.Linear(30, 10))
model:add(nn.LogSoftMax())

print(model)

-- Define loss function (criterion)
-- model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

print('Training the trainset:')
-- print(trainset.data[3])
-- print(trainset.labels)

-- Forward pass
output = model:forward(trainset.data)
loss = criterion:forward(output, trainset.labels)

-- print(output)
print('First Loss: ')
print (loss)

-- Optimization
-- Gradient Descent paramaters

sgd_params = {
	learning_rate = 1e-2,
	learningrate_decay = 1e-4,
	weight_decay = 1e-3,
	momentum = 1e-4
}

w, dw = model:getParameters()

-- print(w)
-- print(dl_dx)

 
