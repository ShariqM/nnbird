function sleep(n)
  os.execute("sleep " .. tonumber(n))
end

function data_gen(x_0, v_0, k, dt_tensor, seq_length)
    k = -50
    params[1] = k

    x   = torch.DoubleTensor(1,1):fill(x_0)
    v   = torch.DoubleTensor(1,1):fill(v_0)
    xv  = torch.DoubleTensor(seq_length, 1)
    xv2 = torch.DoubleTensor(seq_length)
    xv_data = {}

    for t=1, seq_length do
        x, v = unpack(clones.enn[t]:forward{dt_tensor, x, v})
        print (string.format("x=%.3f,v=%.3f", x[{1,1}], v[{1,1}]))
        xv[t] = torch.Tensor(1,1):fill(x[{1,1}])
        xv_data[t] = torch.Tensor(1,1):fill(x[{1,1}])
        xv2[t] = x[{1,1}]
    end

    x_dft = dft:forward(xv2)

    torch.save(string.format('data/k=%d.t7', -k), xv_data)
    torch.save(string.format('data/k=%d_dft.t7', -k), x_dft)
    gnuplot.plot(xv)
    sleep(5)
    gnuplot.plot(x_dft)
    sleep(50)
end

function graph_data(piter, seq_length, x, tgt, x_dft, tgt_dft)
    save = false
    time = true
    freq = false
    sym = '-'
    if save then gnuplot.pngfigure(string.format('graphs/time_%.4d.png', piter)) end
    if time then
        gnuplot.figure(1)
        gnuplot.xlabel('Time [increments of dt]')
        gnuplot.ylabel('x(t) [Position]')
        gnuplot.plot({'Model', xv_graph, sym, }, {'Target', tgt_graph, sym})
    end

    if save then gnuplot.plotflush() end
    if save then gnuplot.pngfigure(string.format('graphs/freq_%.4d.png', piter)) end

    if freq then
        gnuplot.figure(2)
        gnuplot.axis({0, 10, -100, 100})
        gnuplot.xlabel('Freq')
        gnuplot.ylabel('Real Amplitude')
        gnuplot.plot({'Model (Real)', x_dft[{{},1}], sym}, {'Model (Imag)', x_dft[{{},2}], sym},
                     {'Target (Real)', tgt_dft[{{},1}], sym}, {'Target (Imag)', tgt_dft[{{},2}], sym})
    end
    if save then gnuplot.plotflush() end
end

function weight_doutput(tgt_dft, doutput)
    local threshold = tgt_dft:mean() + 1 * tgt_dft:std()
    local weights = tgt_dft:clone() -- Weight gradients by energy in the freq
    weights:apply(function(i)
        if i < threshold then
            return 0
        else
            return 1
        end
    end)
    weights[1] = 1
    return torch.cmul(weights, doutput)
end
