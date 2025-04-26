import OperatorsTree as ot

expr = ot.OperTree.expression('x^2+sin(x+y)^2+y^2+cos(x+y)^2+y*2*x')

print(expr) #x^2+sin(x+y)^2+y^2+cos(x+y)^2+y*2*x

ot.simplify(expr)

print(expr) #x^2+y^2+sin(x+y)^2+cos(x+y)^2+2*x*y

ot.replace(expr,
           ot.new_function('f1',
                           'any1+any2+sin(x)^2+cos(x)^2+any3',
                           'x any1 any2 any3'.split(' ')),
           ot.new_function('f2',
                           'any1+any2+1+any3',
                           'x any1 any2 any3'.split(' ')))

print(expr) #x^2+y^2+1+2*x*y

ot.simplify(expr)

print(expr) #1+x^2+y^2+2*x*y

ot.replace(expr,
           ot.new_function('f1',
                           'any+a^2+b^2+2*a*b',
                           'a b any'.split(' ')),
           ot.new_function('f2',
                           'any+(a+b)^2',
                           'a b any'.split(' ')))

print(expr) #1+(x+y)^2


ot.simplify(expr)

print(expr) #1+(x+y)^2
