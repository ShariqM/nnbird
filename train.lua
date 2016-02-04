require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'gnuplot'
require 'helpers'

matio = require 'matio'
matio.use_lua_strings = true

require 'DiscreteFourierTransform'
require 'CDiscreteFourierTransform'

local model_utils=require 'model_utils'
local ENN = require 'models.enn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Bird Network')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_gen',false,'generate data')
-- optimization
cmd:option('-iters',100,'iterations per epoch')
cmd:option('-learning_rate',5e-1,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')

-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
cmd:option('-save_every',200,'Save every $1 iterations')
cmd:option('-checkpoint_dir', 'save', 'output directory where checkpoints get written')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

torch.manualSeed(os.time())

dt = 1e-6
length = 500 * dt
seq_length = length / dt
print ('seq_length', seq_length)

alpha = -0.41769
beta = 0.346251775
hf = 1/2
x_0 = 0.0
v_0 = 0.0
gam = 23500.0

hf_tensor = torch.DoubleTensor(1,1):fill(hf)
dt_tensor = torch.DoubleTensor(1,1):fill(dt)
gam_tensor = torch.DoubleTensor(1,1):fill(gam)

dft = nn.Sequential()
dft:add(nn.CDiscreteFourierTransform(seq_length))
if string.len(opt.init_from) > 0 then
    print('loading an Network from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
else
    protos = {}
    protos.enn = ENN.hedi_rk2e()
    protos.criterion = nn.MSECriterion()
    dft_criterion = nn.MSECriterion()
    dft_criterion.sizeAverage = false
end

params, grad_params = model_utils.combine_all_parameters(protos.enn)
params[2] = 0 -- 0 bias
params[4] = 0 -- 0 bias
params[1] = alpha
params[3] = beta

print('number of parameters in the model: ' .. params:nElement())
print(string.format('Alpha: %f; Beta: %f', params[1], params[3]))

clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_T_times(proto, seq_length)
    print ('done')
end

checkpoint_now = true
if checkpoint_now then
    local savefile = string.format('%s/enn_epoch%.2f.t7', opt.checkpoint_dir, epoch)
    print('saving checkpoint to ' .. savefile)
    local checkpoint = {}
    checkpoint.protos = protos
    checkpoint.opt = opt
    torch.save(savefile, checkpoint)
end

if opt.data_gen then data_gen(x_0, v_0, k, dt_tensor, seq_length) end

graph = true
piter = 0
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ forward pass -------------------
    local loss = 0
    tgt  = torch.load('data/k=50.t7')
    -- tgt_f = matio.load('x_rk.mat')
    -- tgt = tgt_f['x_rk']
    tgt_dft  = torch.load('data/k=50_dft.t7')
    x  = torch.DoubleTensor(1,1):fill(x_0)
    v  = torch.DoubleTensor(1,1):fill(v_0)
    init_state = {x,v}
    enn_state = {[0] = init_state}
    xv_graph = torch.Tensor(seq_length)
    tgt_graph = torch.Tensor(seq_length)

    pcutoff = 4
    for t=1,seq_length do
        print (t)
        xp, vp = unpack(enn_state[t-1])
        enn_state[t] = clones.enn[t]:forward{hf_tensor, dt_tensor, gam_tensor, gam_tensor, xp, xp, vp}
        xv = enn_state[t][1]

        -- loss = loss + clones.criterion[t]:forward(xv, tgt[t])
        xv_graph[t] = xv
        -- tgt_graph[t] = tgt[t][{1,1}]
        -- tgt_graph[t] = tgt[{1,t}]
    end

    -- x_dft = dft:forward(xv_graph)
    -- loss = dft_criterion:forward(x_dft, tgt_dft)

    if graph and piter % 20 == 0 then graph_data(piter, seq_length, xv_graph, tgt, x_dft, tgt_dft) end
    piter = piter + 1
    debug.debug()

    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local denn_state = {[seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    local doutput = dft_criterion:backward(x_dft, tgt_dft)
    -- doutput = weight_doutput(tgt_dft, doutput)
    doutput = dft:backward(xv_graph, doutput)

    for t=seq_length,1,-1 do
        -- local doutput_t = clones.criterion[t]:backward(enn_state[t][1], tgt[t])
        local doutput_t = doutput[t]

        -- print ("T=", t)
        denn_state[t][1]:add(doutput_t)
        -- dlst = clones.enn[t]:backward({dt_tensor, unpack(enn_state[t-1]), dt_tensor}, denn_state[t])
        xp, vp = unpack(enn_state[t-1])
        dlst = clones.enn[t]:backward({dt_tensor, gam_tensor, gam_tensor, xp, xp, vp}, denn_state[t])

        denn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > pcutoff then -- Ignore gradients on constant inputs (gamma, dt, etc.)
                -- note we do k-1 because first item is dembeddings, and then follow the
                -- derivatives of the state, starting at index 2. I know...
                denn_state[t-1][k-pcutoff] = v
            end
        end
    end

    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip) -- Hmmm...

    grad_params[2] = 0 -- FIXME 0 the bias gradient, Pretty ugly..

    return loss, grad_params
end

-- start optimization here
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * opt.iters
local iterations_per_epoch = opt.iters
local loss0 = nil

for i = 1, iterations do
    local epoch = i / iterations_per_epoch

    local timer = torch.Timer()
    print ('run time')
    local _, loss = optim.adagrad(feval, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it

    -- exponential learning rate decay
    if i % iterations_per_epoch == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if (i % opt.save_every == 0 or i == iterations) then
        local savefile = string.format('%s/enn_epoch%.2f.t7', opt.checkpoint_dir, epoch)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d k=%.2f, (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, params[1], epoch, train_loss, grad_params:norm() / params:norm(), time))
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print "Loss Nan'd"
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
end
