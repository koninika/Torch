-- Include
-- print('Included:')

require 'torch'
require 'nn'

-- Load Data
mnist = require 'mnist'
train_dataset = mnist.traindataset()
test_dataset = mnist.testdataset()

trainset = {
	data = train_dataset.data:double(),
	label = train_dataset.label
}

function trainset:size()
	return self.data:size(1)
end

-- labels can be from 1 to 10 because we have 10 classes and 0 is not allowed
-- so add one to each class label ( 0-9 becomes 1-10)
i = 0
trainset.label:apply(function()
	i = i+1
	trainset.label[i] = trainset.label[i] + 1
	return trainset.label[i]

end)

testset = {
	data = test_dataset.data:double(),
	label = test_dataset.label
}

setmetatable(trainset,
	{ __index = function(t,i)
		return {t.data[i], t.label[i]}
	end}
);

print('trainset:')
print(trainset)
print('trainset size:')
print(trainset:size())


-- Define model
classes = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

model = nn.Sequential()

--[[
model:add(nn.Reshape(28*28))
model:add(nn.Linear(28*28, 30))
model:add(nn.Tanh())
model:add(nn.Linear(30, 10))
model:add(nn.LogSoftMax())
]]
--[[]]


model:add(nn.SpatialConvolution(1, 6, 5, 5))

-- A max-pooling operation that looks at 2x2 windows and finds the max.
model:add(nn.SpatialMaxPooling(2,2,2,2))

-- non-linearity
model:add(nn.Tanh())

-- additional layers
model:add(nn.SpatialConvolution(6, 16, 5, 5))
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.Tanh())

-- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5
model:add(nn.View(16*5*5))

-- fully connected layers (matrix multiplication between input and weights)
model:add(nn.Linear(16*5*5, 120))
model:add(nn.Tanh())
model:add(nn.Linear(120, 84))
model:add(nn.Tanh())

-- 10 is the number of outputs of the network (10 classes)
model:add(nn.Linear(84, 10))

print(model)

-- Define loss function (criterion)
-- model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

print('Training the trainset:')
-- print(trainset.data[3])
-- print(trainset.labels)

-- Forward pass
-- output = model:forward(trainset.data)
-- loss = criterion:forward(output, trainset.label)

-- print(output)
-- print('First Loss: ')
-- print (loss)

-- Optimization
-- Gradient Descent paramaters

--[[
sgd_params = {
	learning_rate = 1e-2,
	learningrate_decay = 1e-4,
	weight_decay = 1e-3,
	momentum = 1e-4
}

parameters, grad_parameters = model:getParameters()

-- print(w)

trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.001
trainer.maxIteration = 1
trainer:train(trainset)
]]

-- Training the Model
batch_size = 100
number_iterations = 10

for i = 1, number_iterations do
--	for index = 1, trainset:size(), batch_size  do
	for index = 1, trainset:size()  do
		model:zeroGradParameters()

--	    data = trainset.data:narrow(1, index, batch_size)
--	    labels = trainset.label:narrow(1, index, batch_size)
		
		data = trainset.data:double()
		labels = trainset.label

		output = model:forward(data[index])
		loss = criterion:forward(output, labels[index])
		gradient = criterion:backward(output, labels[index])
	    model:backward(data, gradient)

		print("Loss: " ..loss)
	end
end

output = model:forward(trainset.data)
loss = criterion:forward(output, trainset.label)
print('Second Loss: ')
print (loss)

print(testset.data:size(1))
-- print(testset.label[1])

-- Testing the model
for i=1,testset.data:size(1) do
    local y = testset.label[i]
    local prediction = model:forward(testset.data[i])
    -- print (prediction)
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if y == indices[1] then
        class_performance[y] = class_performance[y] + 1
        correct = correct + 1
    end
end
print('Accuracy: ')
print(correct, 100*correct/10000 .. ' % ')


for i=1,#classes do
    print(classes[i], 100*class_performance[i]/1000 .. ' %')
end