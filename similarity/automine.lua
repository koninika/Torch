-- Include
-- print('Included:')

require 'torch'
require 'nn'
require 'image'

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

function testset:size()
	return self.data:size(1)
end

--[[
-- Stuff to do before scaling
print ('Scaling images to 256x256..')
data_scaled = torch.Tensor(1000, 3, 256, 256)

-- Image Scaling
for index = 1, 1000 do -- trainset:size() do
	data_scaled[index] = image.scale(trainset.data[index], 256, 256)
end

trainset.data = data_scaled

testdata_scaled = torch.Tensor(500, 3, 256, 256)

for index = 1, 500 do -- testset:size() do
	testdata_scaled[index] = image.scale(testset.data[index], 256, 256)
end 

testset.data = testdata_scaled
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
print(testset:size())


-- Define model

model = nn.Sequential()
model:add(nn.Reshape(32*32*3))
model:add(nn.Linear(32*32*3, 49))
model:add(nn.ReLU())
model:add(nn.Linear(49, 32*32*3))
model:add(nn.Reshape(32*32*3))

--AlexNet
--[[ 
model = nn.Sequential()
model:add(nn.SpatialConvolution(3, 12, 3, 3, 1, 1, 0, 0))
model:add(nn.ReLU())
model:add(nn.SpatialConvolution(12, 12, 3, 3, 1, 1, 0, 0))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
model:add(nn.SpatialConvolution(12, 24, 3, 3, 1, 1, 0, 0))
model:add(nn.ReLU())
model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
model:add(nn.Reshape(24 * 14 * 14))
model:add(nn.Linear(24 * 14 * 14, 1568))
model:add(nn.Linear(1568, 24 * 14 * 14))
model:add(nn.Reshape(24, 14, 14))
model:add(nn.SpatialConvolution(24, 12, 3, 3, 1, 1, 0, 0))
model:add(nn.ReLU())
model:add(nn.SpatialMaxUnpooling(nn.SpatialMaxPooling(2, 2, 2, 2)))
model:add(nn.SpatialConvolution(12, 12, 3, 3, 1, 1, 0, 0))
model:add(nn.ReLU())
model:add(nn.SpatialMaxUnpooling(nn.SpatialMaxPooling(2, 2, 2, 2)))
model:add(nn.SpatialConvolution(12, 3, 3, 3, 1, 1, 0, 0))
--]]
print(model)

-- Define loss function (criterion)
-- model:add(nn.LogSoftMax())
criterion = nn.MSECriterion()

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
number_iterations = 2

for i = 1, number_iterations do
	print("Iteration: " ..i)
	for index = 1, trainset:size(), batch_size  do
	-- for index = 1, trainset:size()  do
		model:zeroGradParameters()

	        data = trainset.data:narrow(1, index, batch_size)
                --labels = trainset.label:narrow(1, index, batch_size)
		output = model:forward(data)
		loss = criterion:forward(output, data)
		gradient = criterion:backward(output, data)
	   	model:backward(data, gradient)

		print("Loss: " ..loss, index)
	end
end

-- print(output)


op = model.modules[4]

vec = torch.zeros(49)
vec[1] = 1

translate = nn.Sequential()
translate:add(op)
translate:add(nn.Reshape(32*32*3))
x=translate:forward(vec)
print(x)
-- image.save("image1.jpg", x)

--print(testset.data:size(1))
-- print(testset.label[1])

--[[
correct = 0
class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
classes = {'airplane', 'automobile', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
-- Testing the model
for i=1,500 do
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
print(correct, 100*correct/testset:size() .. ' % ')


for i=1,#classes do
    print(classes[i],  class_performance[i], 1000*class_performance[i]/testset:size() .. ' %')
end
]]
