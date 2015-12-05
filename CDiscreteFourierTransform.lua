local CDiscreteFourierTransform, parent = torch.class('nn.CDiscreteFourierTransform', 'nn.Module')

function CDiscreteFourierTransform:__init(inputSize)
    parent.__init(self)
    self.inputSize = inputSize
    self.real = nn.Sequential()
    self.real:add(nn.Linear(inputSize, inputSize))
    self.imag = nn.Sequential()
    self.imag:add(nn.Linear(inputSize, inputSize))

    for k=1,inputSize do
        for m=1,inputSize do
            self.real.modules[1].weight[{k,m}] = math.cos(2 * math.pi * (m-1) * (k-1) * (1/inputSize))
            self.imag.modules[1].weight[{k,m}] = math.sin(2 * math.pi * (m-1) * (k-1) * (1/inputSize))
        end
    end
    self.real.modules[1].bias:fill(0)
    self.imag.modules[1].bias:fill(0)
end

function CDiscreteFourierTransform:updateOutput(input)
    self.output = torch.Tensor(self.inputSize, 2)
    self.output[{{},1}] = self.real:updateOutput(input)
    self.output[{{},2}] = self.imag:updateOutput(input)
    return self.output
end

function CDiscreteFourierTransform:updateGradInput(input, gradOutput)
    self.gradInput = self.real:updateGradInput(input, gradOutput[{{},1}]) +
                     self.imag:updateGradInput(input, gradOutput[{{},2}])
    return self.gradInput
end

