require 'nn'
require 'nngraph'

x1 = nn.Identity()()
x2 = nn.Identity()()
x3 = nn.Identity()()

h1 = nn.Linear(10,10)(x3)
h2 = nn.CMulTable()({x2,h1})
add = nn.CAddTable()({x1, h2})
m = nn.gModule({x1, x2, x3}, {add})

