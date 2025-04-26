class ratio:
    """Рациональное число."""

    def __init__(self, numerator: int, denumerator: int = 1):
        self.numerator = numerator
        self.denumerator = denumerator
        self.normalize()

    def normalize(self):
        """Максимально уменьшает числа в числителе и знаменателе, оставляя значения дроби неизменной."""
        if self.denumerator < 0:
            self.numerator = -self.numerator
            self.denumerator = -self.denumerator
        gcd = GCD(abs(self.numerator), self.denumerator)
        self.numerator //= gcd
        self.denumerator //= gcd

    def copy(self):
        return ratio(self.numerator, self.denumerator)

    def __pos__(self):
        return ratio(self.numerator, self.denumerator)

    def __neg__(self):
        return ratio(-self.numerator, self.denumerator)

    def __abs__(self):
        return ratio(abs(self.numerator), self.denumerator)

    def invert(self):
        """Переворчивает дробь."""
        return ratio(self.denumerator, self.numerator)

    def __eq__(self, other):
        return self.numerator * other.denumerator == self.denumerator * other.numerator

    def __ne__(self, other):
        return self.numerator * other.denumerator != self.denumerator * other.numerator

    def __lt__(self, other):
        return self.numerator * other.denumerator < self.denumerator * other.numerator

    def __gt__(self, other):
        return self.numerator * other.denumerator > self.denumerator * other.numerator

    def __le__(self, other):
        return self.numerator * other.denumerator <= self.denumerator * other.numerator

    def __ge__(self, other):
        return self.numerator * other.denumerator >= self.denumerator * other.numerator

    def __add__(self, other):
        if type(other) is int:
            other = ratio(other)
        elif type(other) is float:
            return self.numerator / self.denumerator + other
        return ratio(self.numerator * other.denumerator + other.numerator * self.denumerator, self.denumerator * other.denumerator)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if type(other) is int:
            other = ratio(other)
        elif type(other) is float:
            return self.numerator / self.denumerator - other
        return self + -other

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        if type(other) is int:
            other = ratio(other)
        elif type(other) is float:
            return self.numerator * other / self.denumerator
        return ratio(self.numerator * other.numerator, self.denumerator * other.denumerator)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        if type(other) is int:
            other = ratio(other)
        elif type(other) is float:
            return self.numerator / (self.denumerator * other)
        return self * other.invert()

    def __rtruediv__(self, other):
        return self.invert() * other

    def __pow__(self, power):
        if type(power) is float:
            return (self.numerator / self.denumerator) ** power
        if type(power) is ratio:
            if power.denumerator == 1:
                power = power.numerator
            else:
                return (self.numerator / self.denumerator) ** (power.numerator / power.denumerator)
        res = ratio(1)
        if power >= 0:
            self_1 = self
        else:
            power = -power
            self_1 = self.invert()
        for i in range(power):
            res *= self
        return res

    def __rpow__(self, numerator):
        return numerator ** (self.numerator / self.denumerator)

    def __str__(self):
        if self.denumerator == 0:
            sign = ''
            if self.numerator == -1:
                sign = '-'
            elif self.numerator == 1:
                sign = '+'
            return sign + '∞'
        elif self.denumerator == 1:
            return str(self.numerator)
        return str(self.numerator) + '/' + str(self.denumerator)

    def __hash__(self):
        return hash((self.numerator, self.denumerator))

    def __bool__(self):
        return self.numerator != 0 or self.denumerator == 0

    def __float__(self):
        return self.numerator / self.denumerator

    def __int__(self):
        return self.numerator // self.denumerator

def GCD(a: int, b: int) -> int:
    """Находит наибольший общий делитель между неотрицательными числами "a" и "b"."""
    while a != 0 and b != 0:
        if a < b:
            b %= a
        else:
            a %= b
    if a == 0 and b == 0:
        return float('inf')
    elif a == 0:
        return b
    else:
        return a

if __name__ == '__main__':
    a = ratio(100, 222)
    b = ratio(47, 39)
    print(f'a = {a};\nb = {b};\na + b = {a + b};\na - b = {a - b};\na * b = {a * b};\na / b = {a / b};')
