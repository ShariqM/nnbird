local DiscreteFourierTransform, parent = torch.class('nn.DiscreteFourierTransform', 'nn.Module')

function DiscreteFourierTransform:__init(inputSize)
    parent.__init(self)
    self.net = nn.Sequential()
    self.net:add(nn.Linear(inputSize, inputSize))

    for k=1,inputSize do
        for m=1,inputSize do
            self.net.modules[1].weight[{k,m}] = math.sin(2 * math.pi * (m-1) * (k-1) * (1/inputSize))
        end
    end
    self.net.modules[1].bias:fill(0)
end

function DiscreteFourierTransform:updateOutput(input)
    self.output = self.net:updateOutput(input)
    return self.output
end

function DiscreteFourierTransform:updateGradInput(input, gradOutput)
    self.gradInput = self.net:updateGradInput(input, gradOutput)
    return self.gradInput
end
