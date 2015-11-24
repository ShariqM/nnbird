local ERNN = {}

function ERNN.ernn()
    local x  = nn.Identity()() -- Past
    local v  = nn.Identity()() -- Past
    local dt = nn.Identity()()

    local vdot = nn.Linear(1,1)(x) -- (-kx)
    local v_t  = nn.CAddTable()({v, nn.CMulTable()({vdot,dt})})
    local x_t  = nn.CAddTable()({x, nn.CMulTable()({v_t,dt})})

    return nn.gModule({x, v, dt}, {x_t, v_t})
end

return ERNN
