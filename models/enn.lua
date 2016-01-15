local ENN = {}

function ENN.shc()
    local dt = nn.Identity()()
    local x  = nn.Identity()() -- Past
    local v  = nn.Identity()() -- Past

    local vdot = nn.Linear(1,1)(x) -- (-kx)
    local v_t  = nn.CAddTable()({v, nn.CMulTable()({vdot,dt})})
    local x_t  = nn.CAddTable()({x, nn.CMulTable()({v_t,dt})})

    return nn.gModule({dt, x, v}, {x_t, v_t})
end

function ENN.shc_scalar()
    local dt = nn.Identity()()
    local x  = nn.Identity()() -- Past
    local v  = nn.Identity()() -- Past

    local vdot = nn.Mul()(x) -- (-kx)
    local v_t  = nn.CAdd()({v, nn.CMul()({vdot,dt})})
    local x_t  = nn.CAdd()({x, nn.CMul()({v_t,dt})})
    -- local v_t  = v + nn.CMul()({vdot,dt})
    -- local x_t  = x + nn.CMul()({v_t,dt})

    return nn.gModule({dt, x, v}, {x_t, v_t})
end

function ENN.hedi()
    local dt  = nn.Identity()()
    local g   = nn.Identity()()
    local og  = nn.Identity()()

    local x   = nn.Identity()() -- Past
    local ox  = nn.Identity()() -- Past
    local v   = nn.Identity()() -- Past

    local g2   = nn.CMulTable()({g,og})
    local g2x  = nn.CMulTable()({g2,x})
    local x2   = nn.CMulTable()({x,ox})
    local x3   = nn.CMulTable()({x,x2})
    local xv   = nn.CMulTable()({x,v})
    local x2v  = nn.CMulTable()({x2,v})

    local g2x3 = nn.CMulTable()({g2,x3})
    local gx2v = nn.CMulTable()({g,x2v})
    local g2x2 = nn.CMulTable()({g2,x2})
    local gxv  = nn.CMulTable()({g,xv})

    local aterm  = nn.Linear(1,1)(g2)
    local bterm  = nn.Linear(1,1)(g2x)

    local vdot = nn.CSubTable()({aterm, bterm})
    local vdot = nn.CSubTable()({vdot, g2x3})
    local vdot = nn.CSubTable()({vdot, gx2v})
    local vdot = nn.CAddTable()({vdot, g2x2})
    local vdot = nn.CSubTable()({vdot, gxv})

    local v_t  = nn.CAddTable()({v, nn.CMulTable()({vdot,dt})})
    local x_t  = nn.CAddTable()({x, nn.CMulTable()({v_t,dt})})

    return nn.gModule({dt, g, og, x, ox, v}, {x_t, v_t})
end

function ENN.hedi_rk2e()
    local dt  = nn.Identity()()
    local g   = nn.Identity()()
    local og  = nn.Identity()()

    local x   = nn.Identity()() -- Past
    local ox  = nn.Identity()() -- Past
    local v   = nn.Identity()() -- Past

    local g2   = nn.CMulTable()({g,og})
    local g2x  = nn.CMulTable()({g2,x})
    local x2   = nn.CMulTable()({x,ox})
    local x3   = nn.CMulTable()({x,x2})
    local xv   = nn.CMulTable()({x,v})
    local x2v  = nn.CMulTable()({x2,v})

    local g2x3 = nn.CMulTable()({g2,x3})
    local gx2v = nn.CMulTable()({g,x2v})
    local g2x2 = nn.CMulTable()({g2,x2})
    local gxv  = nn.CMulTable()({g,xv})

    local aterm  = nn.Linear(1,1)(g2)
    local bterm  = nn.Linear(1,1)(g2x)

    local vdot = nn.CAddTable()({aterm, bterm})
    local vdot = nn.CSubTable()({vdot, g2x3})
    local vdot = nn.CSubTable()({vdot, gx2v})
    local vdot = nn.CAddTable()({vdot, g2x2})
    local vdot = nn.CSubTable()({vdot, gxv})

    local v_t  = nn.CAddTable()({v, nn.CMulTable()({vdot,dt})})
    local x_t  = nn.CAddTable()({x, nn.CMulTable()({v_t,dt})})

    return nn.gModule({dt, g, og, x, ox, v}, {x_t, v_t})
end




return ENN
