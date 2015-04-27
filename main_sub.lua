--
----  Copyright (c) 2014, Facebook, Inc.
----  All rights reserved.
----
----  This source code is licensed under the Apache 2 license found in the
----  LICENSE file in the root directory of this source tree. 
----
ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    LookupTable = nn.LookupTable
end
require('nngraph')
require('base')
ptb = require('data_char')
comm = require('a4_communication_loop_char')
arg_gpu = {3}

-- Train 1 day and gives 82 perplexity.
--[[]
params = {batch_size=100,
                seq_length=50,
                layers=2,
                decay=1.15,
                rnn_size=1500,
                dropout=0.65,
                init_weight=0.04,
                lr=1,
                vocab_size=50,
                max_epoch=14,
                max_max_epoch=55,
                max_grad_norm=10}
]]--
-- Trains 1h and gives test 115 perplexity.

params = {batch_size=300,
                seq_length=50,
                layers=2,
                decay=2,
                rnn_size=200,
                dropout=0,
                init_weight=0.1,
                lr=1,
                vocab_size=50,
                max_epoch=4,
                max_max_epoch=13,
                max_grad_norm=5}

function transfer_data(x)
  return x:cuda()
end

--local state_train, state_valid, state_test
model = {}
--local paramx, paramdx

function lstm(i, prev_c, prev_h)
  local function new_input_sum()
    local i2h            = nn.Linear(params.rnn_size, params.rnn_size)
    local h2h            = nn.Linear(params.rnn_size, params.rnn_size)
    return nn.CAddTable()({i2h(i), h2h(prev_h)})
  end
  local in_gate          = nn.Sigmoid()(new_input_sum())
  local forget_gate      = nn.Sigmoid()(new_input_sum())
  local in_gate2         = nn.Tanh()(new_input_sum())
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_gate2})
  })
  local out_gate         = nn.Sigmoid()(new_input_sum())
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end

function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = LookupTable(params.vocab_size,
                                                    params.rnn_size)(x)}
  local next_s           = {}
  local split         = {prev_s:split(2 * params.layers)}
  for layer_idx = 1, params.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  local dropped          = nn.Dropout(params.dropout)(i[params.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s), nn.Identity()(pred)})
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module)
end

function setup()
  print("Creating a RNN LSTM network.")
  local core_network = create_network()
  paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  model.pred = {}
  for j = 0, params.seq_length do
    model.s[j] = {}
    model.pred[j] = transfer_data(torch.zeros(params.batch_size, params.vocab_size))
    for d = 1, 2 * params.layers do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
  end
  for d = 1, 2 * params.layers do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length))
  --create space storing prediction
end

function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
    end
  end
end

function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end
  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    model.err[i], model.s[i],model.pred[i] = unpack(model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[params.seq_length])
  return model.err:mean()
end

function bp(state)
  paramdx:zero()
  reset_ds()
  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local tmp = model.rnns[i]:backward({x, y, s},
                                       {derr, model.ds, transfer_data(torch.zeros(params.batch_size, params.vocab_size))})[3]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-params.lr))
end

function run_valid()
  reset_state(state_valid)
  g_disable_dropout(model.rnns)
  local len = (state_valid.data:size(1) - 1) / (params.seq_length)
  local perp = 0
  for i = 1, len do
    perp = perp + fp(state_valid)
  end
  print("Validation set perplexity : " .. g_f3(torch.exp(5.6*perp / len)))
  g_enable_dropout(model.rnns)
end

function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns)
  local perp = 0
  local len = state_test.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_test.data[i]
    local y = state_test.data[i + 1]
    local s = model.s[i - 1]
    perp_tmp, model.s[1],_ = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
  end
  print("Test set perplexity : " .. g_f3(torch.exp(5.6 * perp / (len - 1))))
  g_enable_dropout(model.rnns)
end


function query_sentences()
  g_init_gpu(arg_gpu)
  state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
  -- query_len = 10
  -- query_words = {'new','york'}
  query_len, query_words = comm.getinput()
  query_len = tonumber(query_len)
  rev_dict = ptb.table_invert(ptb.vocab_map)
  temp = comm.input_to_dict(query_words)
  temp = temp:resize(temp:size(1),1):expand(temp:size(1), params.batch_size) --batch_size
  model = torch.load('/home/user1/a4/lstm/model_char.net')
  state_query = {data=transfer_data(temp)}
  reset_state(state_query)
  g_disable_dropout(model.rnns)
  g_replace_table(model.s[0], model.start_s)
  if query_len <= #query_words then 
    print(table.concat(query_words, " "))
  else
    for i =1, (query_len-1) do
      if i<#query_words then
        y = state_query.data[i+1]
      else
        y = state_query.data[#query_words]
      end
      x = state_query.data[i]
      s = model.s[i - 1]
      _, model.s[1], query_pred = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
      g_replace_table(model.s[0], model.s[1])
      if (i+1) > #query_words then
        local _,max_index = torch.max(query_pred[1],1)
        table.insert(query_words, rev_dict[max_index[1]])
        temp = comm.input_to_dict(query_words)
        temp = temp:resize(temp:size(1),1):expand(temp:size(1), params.batch_size)
        state_query = {data=transfer_data(temp)}
      end
    end
    print(table.concat(query_words, " "))
  end
end

function submission()
  g_init_gpu(arg_gpu)
  state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
  model = torch.load('/home/user1/a4/lstm/model_char.net')
  rev_dict = ptb.table_invert(ptb.vocab_map)
  print("OK GO")
  io.flush()
  while true do
    query_words = comm.getinput_submission()
    if #query_words == 0 then break end
    temp = comm.input_to_dict(query_words)
    temp = temp:resize(temp:size(1),1):expand(temp:size(1), params.batch_size)
    state_query = {data=transfer_data(temp)}
    reset_state(state_query)
    g_disable_dropout(model.rnns)
    g_replace_table(model.s[0], model.start_s)
    x = state_query.data[1]
    y = state_query.data[1]
    _, _, query_pred = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    pred_sub = torch.totable(query_pred[1]:float())
    print(table.concat(pred_sub, " "))
    io.flush()
  end
end
--function main()
function main()
  g_init_gpu(arg_gpu)
  state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
  state_valid = {data=transfer_data(ptb.validdataset(params.batch_size))}
  print("Network parameters:")
  print(params)
  local states = {state_train, state_valid}
  for _, state in pairs(states) do
   reset_state(state)
  end
  setup()
  step = 0
  epoch = 0
  total_cases = 0
  beginning_time = torch.tic()
  start_time = torch.tic()
  print("Starting training.")
  words_per_step = params.seq_length * params.batch_size
  epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)
  --perps
  while epoch < params.max_max_epoch do
   perp = fp(state_train)
   if perps == nil then
     perps = torch.zeros(epoch_size):add(perp)
   end
   perps[step % epoch_size + 1] = perp
   step = step + 1
   bp(state_train)
   total_cases = total_cases + params.seq_length * params.batch_size
   epoch = step / epoch_size
   if step % torch.round(epoch_size / 10) == 10 then
     wps = torch.floor(total_cases / torch.toc(start_time))
     since_beginning = g_d(torch.toc(beginning_time) / 60)
     print('epoch = ' .. g_f3(epoch) ..
           ', train perp. = ' .. g_f3(torch.exp(5.6 * perps:mean())) ..
           ', wps = ' .. wps ..
           ', dw:norm() = ' .. g_f3(model.norm_dw) ..
           ', lr = ' ..  g_f3(params.lr) ..
           ', since beginning = ' .. since_beginning .. ' mins.')
   end
   if step % epoch_size == 0 then
     run_valid()
     if epoch > params.max_epoch then
         params.lr = params.lr / params.decay
     end
   end
   if step % 33 == 0 then
     cutorch.synchronize()
     collectgarbage()
   end
  end
  print("Saving model")
  torch.save('/home/user1/a4/lstm/model_char_20.net', model)
  print("Training is over.")
end
  --end

if not opt then
   cmd = torch.CmdLine()
   cmd:option('-mode', 'evaluate', 'mode: train | query | evaluate')
   opt = cmd:parse(arg or {})
end

if opt.mode == 'evaluate' then
  submission()
elseif opt.mode == 'train' then
  main()
elseif opt.mode == 'query' then
  query_sentences()
end
