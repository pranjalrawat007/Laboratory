from secrets import choice, randbelow

print(randbelow(10))


def ber():
    return choice([0, 1])


class bernoulli():
    def __init__(self, p):
        self.p = round(p * 100)
        self.D = [0, 1]

    def sample(self, N=1):
        return [1 if randbelow(100) < self.p else 0 for i in range(N)]

    def pmf(self, value):
        if value in self.D:
            return self.sample(5000).count(value) / 5000
        else:
            return 0

    def cdf(self, value):
        result = 0
        for i in self.D:
            if i <= value:
                result += self.pmf(i)
        return result


X = bernoulli(p=0.5)
x1 = X.sample(10)
print(x1)
print(X.pmf(0), X.pmf(1))
print(X.cdf(0), X.cdf(1))


class binomial(bernoulli):
    def __init__(self, p, n):
        super().__init__(p)
        self.D = list(range(n))
        self.n = n

    def sample(self, N=1):
        result = []
        for i in range(N):
            result.append(sum(super().sample(self.n)))
        return result


X = binomial(p=0.3, n=10)


X = binomial(p=0.5, n=5000)
x1 = X.sample(5000)

import matplotlib.pyplot as plt
plt.hist(x1)
plt.show()
# print(x1)
# for i in range(10):
#    i, X.pmf(i), X.cdf(i))


'''

Y = binomial(p=0.5, n=10)
print(X.pmf(0), X.pmf(1))
print(X.cdf(0), X.cdf(1))

print(Y.sample())

class binomial(bernoulli):
    def __init__(self, p, n):
        super().__init__(p)
        self.D = [1 for i in range(p)] + [0 for i in range(100 - p)]

def ber():
    D = [0, 1]
    return choice(D)


def bernoulli(p=0.5):
    p = round(p * 100)
    D = [1 for i in range(p)] + [0 for i in range(100 - p)]
    return choice(D)


def binomial(p=0.5, n=10):
    D = [bernoulli(p=0.5) for i in range(n)]
    return sum(D)


def pmf(rv, **args):
    D = [rv(**args) for i in range(1000)]
    values, freqs = list(set(D)), []
    for i in values:
        freqs.append(D.count(i) / 1000)
    return values, freqs


print(pmf(binomial, p=0.7, n=20))
'''
