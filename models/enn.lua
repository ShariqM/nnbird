local ENN = {}

function ENN.enn()
    local dt = nn.Identity()()
    local x  = nn.Identity()() -- Past
    local v  = nn.Identity()() -- Past

    local vdot = nn.Linear(1,1)(x) -- (-kx)
    local v_t  = nn.CAddTable()({v, nn.CMulTable()({vdot,dt})})
    local x_t  = nn.CAddTable()({x, nn.CMulTable()({v_t,dt})})

    return nn.gModule({dt, x, v}, {x_t, v_t})
end

return ENN
