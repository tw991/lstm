stringx = require('pl.stringx')
require 'io'
ptb = require('data')

local function readline()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = stringx.split(line)
  if tonumber(line[1]) == nil then error({code="init"}) end
  for i = 2,#line do
    if ptb.vocab_map[line[i]] == nil then error({code="vocab", word = line[i]}) end
  end
  return line
end

local function getinput()
  print("Query: len word1 word2 etc")
  local ok, line = pcall(readline)
  if not ok then
    if line.code == "EOF" then
      break -- end loop
    elseif line.code == "vocab" then
      print("Word not in vocabulary, only 'foo' is in vocabulary: ", line.word)
    elseif line.code == "init" then
      print("Start with a number")
    else
      print(line)
      print("Failed, try again")
    end
  else
    local len = line[1]
    local words = table.remove(line, 1)
    return len, words
  end
end

local function input_to_dict(data)
  local x = torch.zeros(#data)
  for i = 1, #data do
    x[i] = vocab_map[data[i]]
  end
  return x
end

local function query_sentences()
  g_disable_dropout(model.rnns)
  g_replace_table(model.s[0], model.start_s)
  local len, words = getinput()
  local rev_dict = ptb.table_invert(ptb.vocab_map)
  local out_words = {}
  state_query = {data=transfer_data(input_to_dict(words))}
  if len >= #words then 
    print(table.concat(words, " "))
  else
    for i =1, (len-1) do
      if i<#words then
        local y = state_query.data[i+1]
      else
        local y = state_query.data[#words]
      end
      local x = state_query.data[i]
      local s = model.s[i - 1]
      _, model.s[1], query_pred = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
      g_replace_table(model.s[0], model.s[1])
      if (i+1) > #words then
        local _,max_index = torch.max(torch.Tensor(query_pred))
        table.insert(words, max_index)
      end
    end
    for i =1, #words do
      out_words[i] = rev_dict[words[i]]
    end
    print(table.concat(words, " "))
  end
end






end