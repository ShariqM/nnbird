require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'gnuplot'

local model_utils=require 'model_utils'
local ENN = require 'models.enn'

dt = 1 / 16000
length = 30 * 0.001 -- 30 ms
seq_length = length / dt

x_0 = 1.0
v_0 = 0.0
k = -50
dt = 0.01
seq_length = 200 -- FIXME

protos = {}
protos.enn = ENN.enn()
protos.criterion = nn.MSECriterion()

params, grad_params = model_utils.combine_all_parameters(protos.enn)
params[1] = k
params[2] = 0 -- 0 bias
print('number of parameters in the model: ' .. params:nElement())

clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_T_times(proto, seq_length)
end

dt_tensor = torch.DoubleTensor(1,1):fill(dt)
x  = torch.DoubleTensor(1,1):fill(x_0)
v  = torch.DoubleTensor(1,1):fill(v_0)
xv = torch.DoubleTensor(seq_length)

function sleep(n)
    os.execute("sleep " .. tonumber(n))
end

for t=1, seq_length do
    x, v, vdot = unpack(clones.enn[t]:forward{x, v, dt_tensor})
    print (string.format("x=%.3f,v=%.3f, vdot=%.3f", x[{1,1}], v[{1,1}], vdot[{1,1}]))
    xv[t] = x[{1,1}]
end

gnuplot.plot(xv)
sleep(100)

print('number of parameters in the model: ' .. params:nElement())
