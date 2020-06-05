from sympy import *
x, π, σ = symbols('x, π, σ')
pdf = sqrt(1 / 2 / pi /σ ** 2) * exp(- (x - π) ** 2 / 2 / σ **2)
print(pdf.subs({'π': 0, 'σ': 1}).evalf(2))

a = 1 /σ * sqrt(1 / 2 / pi)
z = ((x - π) /σ) ** 2
b = exp(-0.5 * z)
pdf = a * b
pprint(a * b)
#pdf = pdf.subs({'π': 2, 'σ': 1})

# CDF
pprint(integrate(pdf, (x, -oo, oo), conds="none").evalf(4))

# E(x)
E = simplify(integrate(x * pdf, (x, -oo, oo), conds="none").evalf(4))
print(E)

# V(x)
V = simplify(integrate(pdf * (x - E)**2, (x, -oo, oo), conds="none").evalf(4))
pprint(V)

# MGF
t = symbols('t')

MGF = simplify(integrate(exp(t * E) * pdf, (x, -oo, oo), conds="none").evalf(4))


pprint(MGF)
