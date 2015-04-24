local function readline()
  local input
  io.write("Query: len word1 word2 etc\n")
  io.flush()
  io.write("IN: ")
  input = io.read()
  io.flush()
  parse_input = stringx.split(input)
  len = parse_input[1]
  table.remove(parse_input, 1)
  parse_string = table.concat(parse_input, " ")
  return len, parse_string
end

for i =1,5 do
	len, input = readline()
	print(len)
	print(input)
end