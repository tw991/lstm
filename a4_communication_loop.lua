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
  local ok, line = pcall(readline())
  if not ok then
    if line.code == "EOF" then
      --break -- end loop
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
    local table.remove(line, 1)
    return len, line
  end
end


local function input_to_dict(data)
  local x = torch.zeros(#data)
  for i = 1, #data do
    x[i] = ptb.vocab_map[data[i]]
  end
  return x
end


return {readline=readline,
        getinput=getinput,
        input_to_dict=input_to_dict}

