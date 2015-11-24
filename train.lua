require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'gnuplot'


dt = 1 / 16000
length = 30 * 0.001 -- 30 ms)
seq_length = length / dt

protos = {}
protos.ernn = ERNN.ernn()
protos.criterion = nn.MSECriterion()


for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_T_times(proto, seq_length)
end
