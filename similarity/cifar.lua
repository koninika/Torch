-- Include
-- print('Included:')

require 'torch'
require 'nn'

-- Load Data CIFAR
train_dataset = torch.load('cifar10-train.t7')
test_dataset = torch.load('cifar10-test.t7')

trainset = {
	data = train_dataset.data:double(),
	label = train_dataset.label
}

function trainset:size()
	return self.data:size(1)
end

testset = {
	data = test_dataset.data:double(),
	label = test_dataset.label
}

--[[
setmetatable(trainset,
	{ __index = function(t,i)
		return {t.data[i], t.label[i]}
	end}
);
]]

print('trainset:')
print(trainset)
print('trainset size:' .. trainset:size())
print('testset:')
print(testset)
--print(trainset:size())


-- Define model
classes = {'airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}

model = nn.Sequential()

model:add(nn.SpatialConvolution(3, 6, 5, 5,1,1,2,2))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.SpatialConvolution(6, 16, 5, 5,1,1,2,2))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.SpatialConvolution(16,32,5,5,1,1,2,2))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2,2,2,2))
model:add(nn.View(32*4*4))
model:add(nn.Linear(32*4*4, 120))
model:add(nn.ReLU())
model:add(nn.Linear(120, 84))
model:add(nn.ReLU())
model:add(nn.Linear(84, 10))
model:add(nn.LogSoftMax())

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
	for index = 1, trainset:size(), batch_size  do
	-- for index = 1, trainset:size()  do
		model:zeroGradParameters()

	        data = trainset.data:narrow(1, index, batch_size)
                labels = trainset.label:narrow(1, index, batch_size)
		output = model:forward(data)
		loss = criterion:forward(output, labels)
		gradient = criterion:backward(output, labels)
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
