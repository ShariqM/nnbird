--[[ A plain implementation of Monte Carlo Exchange (Parallel Tempering) with SGD

ARGS:

- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.learningRate`      : learning rate
- `config.learningRateDecay` : learning rate decay
- `config.weightDecay`       : weight decay
- `config.weightDecays`      : vector of individual weight decays
- `config.momentum`          : momentum
- `config.dampening`         : dampening for momentum
- `config.nesterov`          : enables Nesterov momentum
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.learningRates`      : vector of individual learning rates

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

(Clement Farabet, 2012)
]]
function optim.mce(opfunc, x, config, state)
   -- (0) get/update state
    local config = config or {}
    local state = state or config

    local nwalkers = config.nwalkers or 4
    local lrates = config.learningRates or torch.linspace(1e-2, 5, nwalkers)

    -- local nwalkers = config.nwalkers or 1
    -- local lrates = config.learningRates or {1e-2}

    -- local nwalkers = config.nwalkers or 2
    -- local lrates = config.learningRates or {1e-2, 10}

    local lrd = config.learningRateDecay or 0
    local wd = config.weightDecay or 0
    local mom = config.momentum or 0
    local damp = config.dampening or mom
    local nesterov = config.nesterov or false

    if not state.params then
        state.params = {}
        for i=1, nwalkers do
            state.params[i] = state.paramGen()
            print(string.format("%d) Init=%d", i, state.params[i][1]))
        end
    end

    state.evalCounter = state.evalCounter or 0
    local nevals = state.evalCounter
    assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

    -- TODO Incorporate momentum, decay, etc.

    -- (0) vars
    fxs = torch.Tensor(nwalkers)
    dfdxs = {}
    udfdxs = {}

    -- (1) evaluate f(x) and df/dx for all params
    for k,v in pairs(state.params) do
        fxs[k], udfdxs[k] = opfunc(v)
        udfdxs[k] = udfdxs[k]:clone() --
    end

    -- (2) Sort the results by f(x) (amount of error)
    fxs, sorder = torch.sort(fxs)

    nparams = {}
    for i=1, nwalkers do
        dfdxs[i] = udfdxs[sorder[i]]
        nparams[i] = state.params[sorder[i]]
    end
    state.params = nparams

    -- (3) Update parameters with their corresponding learning rates
    for i=1, nwalkers do
        state.params[i]:add(-lrates[i], dfdxs[i])
    end

    -- (6) update evaluation counter
    state.evalCounter = state.evalCounter + 1

    -- return the best x*, f(x) before optimization
    return state.params[1], {fxs[1]}
end
