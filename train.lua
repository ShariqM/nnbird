require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'gnuplot'
require 'helpers'

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
cmd:option('-learning_rate',10,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')

-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
cmd:option('-save_every',200,'Save every $1 iterations')
cmd:option('-checkpoint_dir', 'save', 'output directory where checkpoints get written')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

dt = 1 / 16000
length = 30 * 0.001 -- 30 ms
seq_length = length / dt

x_0 = 1.0
v_0 = 0.0
dt = 0.01
seq_length = 200

dt_tensor = torch.DoubleTensor(1,1):fill(dt)

protos = {}
protos.enn = ENN.enn()
protos.criterion = nn.MSECriterion()

params, grad_params = model_utils.combine_all_parameters(protos.enn)
params[2] = 0 -- 0 bias
print('number of parameters in the model: ' .. params:nElement())
params[1] = -50

clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_T_times(proto, seq_length)
end

if opt.data_gen then
    k = -50
    params[1] = k

    x  = torch.DoubleTensor(1,1):fill(x_0)
    v  = torch.DoubleTensor(1,1):fill(v_0)
    xv = torch.DoubleTensor(seq_length, 1)
    xv_data = {}

    for t=1, seq_length do
        x, v = unpack(clones.enn[t]:forward{dt_tensor, x, v})
        print (string.format("x=%.3f,v=%.3f", x[{1,1}], v[{1,1}]))
        xv[t] = torch.Tensor(1,1):fill(x[{1,1}])
        xv_data[t] = torch.Tensor(1,1):fill(x[{1,1}])
    end

    torch.save(string.format('data/k=%d.t7', -k), xv_data)
    gnuplot.plot(xv)
    sleep(100)
end

function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ forward pass -------------------
    local loss = 0
    tgt  = torch.load('data/k=50.t7')
    x  = torch.DoubleTensor(1,1):fill(x_0)
    v  = torch.DoubleTensor(1,1):fill(v_0)
    init_state = {x,v}
    local enn_state = {[0] = init_state}

    for t=1,seq_length do
        -- enn_state[t] = clones.enn[t]:forward{unpack(enn_state[t-1]), dt_tensor}
        enn_state[t] = clones.enn[t]:forward{dt_tensor, unpack(enn_state[t-1])}
        xv = enn_state[t][1]
        loss = loss + clones.criterion[t]:forward(xv, tgt[t])
    end

    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local denn_state = {[seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    -- local denn_state = {[seq_length] = {}}
    for t=seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(enn_state[t][1], tgt[t])

        denn_state[t][1]:add(doutput_t)
        local dlst = clones.enn[t]:backward({unpack(enn_state[t-1]), dt_tensor}, denn_state[t])


        denn_state[t-1] = {}
        -- print (dlst[1])
        -- print (dlst[2])
        -- print (dlst[3])
        -- debug.debug()
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the
                -- derivatives of the state, starting at index 2. I know...
                denn_state[t-1][k-1] = v
                -- print ('hi', v)
            end
        end
    end

    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip) -- Hmmm...
    -- debug.debug()

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
    local _, loss = optim.sgd(feval, params, optim_state)
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
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print "Loss Nan'd"
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    -- if loss[1] > loss0 * 3 then
        -- print('loss is exploding, aborting.')
        -- break -- halt
    -- end
end
