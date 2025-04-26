from OperatorsTree import *
from random import Random

generator = type(0 for _ in range(1))

zero = ratio(0)
unit = ratio(1)

def f_add(x: 'ratio | float', y: 'ratio | float') -> 'ratio | float':
    """Возвращает сумму двух чисел."""
    return x + y

def f_sub(x: 'ratio | float', y: 'ratio | float') -> 'ratio | float':
    """Возвращает разность двух чисел."""
    return x - y

def f_mul(x: 'ratio | float', y: 'ratio | float') -> 'ratio | float':
    """Возвращает произведение двух чисел."""
    return x * y

def f_div(x: 'ratio | float', y: 'ratio | float') -> 'ratio | float':
    """Возвращает частное двух чисел."""
    return x / y

def f_pow(x: 'ratio | float', y: 'ratio | float') -> 'ratio | float':
    """Возвращает результат возведения x в степень y."""
    return x ** y

def f_pos(x: 'ratio | float') -> 'ratio | float':
    """Возвращает положительное значение x."""
    return x

def f_neg(x: 'ratio | float') -> 'ratio | float':
    """Возвращает отрицательное значение x."""
    return -x

def f_ide(x: 'ratio | float') -> 'ratio | float':
    """Возвращает прямое значение x."""
    return x

def f_inv(x: 'ratio | float') -> 'ratio | float':
    """Возвращает обратное значение x."""
    return 1 / x

class Algebra:
    """Алгебра со своими правилами.
associativity - множество существующих операций и имена собственных функций, на которых выполняется ассоциативность
(то есть можно в любом порядке проводить рассчёты),
commutativity - множество существующих операций и имена собственных функций, на которых выполняется коммутативность
(то есть результат операции над двумя элементами не зависит от их положения относительно друг друга),
**other - другие правила алгебры.

Существующие операции обозначаются константами: T_ADD - сложение, T_MUL - умножение, и так далее...
Собственная (произвольная) функция должна существовать, чтобы не возникло проблем в дальнейшем, если вы решили
добавить её как ассоциативную или коммутативную операцию.
У каждой пары функций в других правилах алгебры должны совпадать аргументы."""

    def __init__(self,
                 associativity: set = None,
                 commutativity: set = None,
                 **other: {str: (str, str)}):
        if associativity is None:
            associativity = set()
        if commutativity is None:
            commutativity = set()
        self.asso = associativity
        self.comm = commutativity
        self.rules = other

    def random(self,
               lvls: 'int|[int, int]' = 3,
               brs: 'int|[int, int]' = 5,
               types: set = {T_CONST: 1, T_VAR: 1, F_SIN: 1, F_COS: 1, F_TAN: 1, F_COTAN: 1, F_LOG: 1, T_POW: 1, T_DIV: 1, T_MUL: 1, T_SUB: 1, T_ADD: 1},
               rnd: 'int|Random' = None,
               MAX_CONST_VALUE: 'int|float|ratio' = 10) -> OperTree:
        """Возвращает случайное математическое выражение."""
        #print(f'{lvls = }\n{brs = }\n{types = }\n{rnd = }\n{MAX_CONST_VALUE = }')
        if type(lvls) is int:
            lvls = [lvls, lvls]
        elif type(lvls) is not list:
            raise TypeError(f'"lvls" must have int or list type but {type(lvls)} is given.')
        elif len(lvls) != 2 or any([type(x) is not int for x in lvls]):
            raise TypeError(f'"lvls" must be [int, int] but it is {lvls}.')
        elif not 0 <= lvls[0] <= lvls[1] > 0:
            raise ValueError(f'"lvls" must be [a, b] where 0 <= a <= b > 0')
        if type(brs) is int:
            brs = [brs, brs]
        elif type(brs) is not list:
            raise TypeError(f'"brs" must have int or list type but {type(brs)} is given.')
        elif len(brs) != 2 or any([type(x) is not int for x in brs]):
            raise TypeError(f'"lvls" must be [int, int] but it is {brs}.')
        elif not 0 <= brs[0] <= brs[1] > 0:
            raise ValueError(f'"brs" must be [a, b] where 0 <= a <= b > 0')
        if type(types) is not dict:
            raise TypeError(f'"types" must have dict type but {type(types)} is given.')
        elif any(type(x) is not int or not type(types[x]) in {int, float, ratio} for x in types):
            raise ValueError(f'Keys in "types" must have int type and values in "types" must be numbers.')
        if rnd is None:
            rnd = Random()
        elif type(rnd) is int:
            rnd = Random(rnd)
        elif type(rnd) is not Random:
            TypeError(f'"rnd" must have int or Random type but {type(rnd)} is given.')
        ops = {t: types[t] for t in types if t in {T_POW, T_DIV, T_MUL, T_SUB, T_ADD}}
        fns = {t: types[t] for t in types if t in {F_SIN, F_COS, F_TAN, F_COTAN, F_LOG, F_FUNC}}
        vls = {t: types[t] for t in types if t in {T_FLOAT, T_CONST, T_VAR, T_LOCVAR}}
        ops_r = sum(ops[t] for t in ops)
        fns_r = sum(fns[t] for t in fns)
        vls_r = sum(vls[t] for t in vls)
        if lvls[0] in [0, 1]:
            if lvls[1] == 1:
                op = random(vls, rnd)
            else:
                op = random(ops | fns | vls, rnd)
        else:
            op = random(ops | fns, rnd)
        if op == T_FLOAT:
            trs = None
            val = (rnd.random() - 0.5) * MAX_CONST_VALUE
        elif op == T_CONST:
            trs = None
            val = ratio(rnd.randint(-MAX_CONST_VALUE, MAX_CONST_VALUE), rnd.randint(1, MAX_CONST_VALUE))
        elif op == T_VAR:
            trs = None
            val = f'var{rnd.randint(0, MAX_CONST_VALUE)}'
        elif op == T_LOCVAR:
            trs = None
            val = f'locvar{rnd.randint(0, MAX_CONST_VALUE)}'
        elif op == T_POW:
            trs = [self.random(lvls = [max(0, lvls[0] - 1),
                                       max(1, lvls[1] - 1)],
                               brs = brs,
                               types = types,
                               rnd = rnd,
                               MAX_CONST_VALUE = MAX_CONST_VALUE)
                   for _ in range(2)]
            val = None
        elif op in {T_DIV, T_MUL, T_SUB, T_ADD, F_FUNC}:
            trs = [self.random(lvls = [max(0, lvls[0] - 1),
                                       max(1, lvls[1] - 1)],
                               brs = brs,
                               types = types,
                               rnd = rnd,
                               MAX_CONST_VALUE = MAX_CONST_VALUE)
                   for _ in range(rnd.randint(max(1, brs[0]), brs[1]))]
            val = None
            if op == F_FUNC:
                val = f'func{rnd.randint(0, MAX_CONST_VALUE)}'
        elif op in {F_SIN, F_COS, F_TAN, F_COTAN}:
            trs = [self.random(lvls = [max(0, lvls[0] - 1),
                                       max(1, lvls[1] - 1)],
                               brs = brs,
                               types = types,
                               rnd = rnd,
                               MAX_CONST_VALUE = MAX_CONST_VALUE)]
            val = None
        elif op == F_LOG:
            trs = [self.random(lvls = [max(0, lvls[0] - 1),
                                       max(1, lvls[1] - 1)],
                               brs = brs,
                               types = types,
                               rnd = rnd,
                               MAX_CONST_VALUE = MAX_CONST_VALUE)
                   for _ in range(rnd.randint(1, 2))]
            val = None
        else:
            raise Exception('ERROR')
        res = OperTree(op, trs, val)
        return res

    def expression(self,
                   s: str,
                   loc_vars: list = None,
                   warnings: list = None) -> OperTree:
        """Возвращает операционное дерево, построенное по строке."""
        if loc_vars is None:
            loc_vars = []
        if warnings is None:
            warnings = []
        s = s.replace(' ', '')
        while s != '' and pairOf(s, 0) == len(s) - 1:
            s = s[1:-1]
        if s == '':
            raise ValueError(f"s does not contain a mathematical expression")
        #Сложение и вычитание:
        res = OperTree(T_ADD, [])
        count = 0
        token = ''
        for sym in s:
            if sym == '(':
                count += 1
            elif sym == ')':
                count -= 1
            elif count == 0:
                if sym in '+-':
                    if len(token) != 0:
                        res.trees.append(token)
                    token = ''
            token += sym
        if count < 0:
            raise SyntaxError("')' was never closed in math expression")
        elif count > 0:
            raise SyntaxError("'(' was never closed in math expression")
        if len(token) == 1 and token in '+-':
            raise SyntaxError(f"Expected a mathematical expression after '{token}'")
        elif len(token) != 0:
            res.trees.append(token)
        if len(res.trees) > 0:
            if len(res.trees) == 1:
                if res.trees[0][0] == '-':
                    res.type = T_SUB
                    res.trees[0] = self.expression(res.trees[0][1:], loc_vars, warnings)
                    return res
                if res.trees[0][0] == '+':
                    s = res.trees[0][1:]
                else:
                    s = res.trees[0]
            else:
                res.trees = [self.expression(tree, loc_vars, warnings) for tree in res.trees]
                return res
        #Умножение и деление:
        res.type = T_MUL
        res.trees = []
        count = 0
        token = ''
        for sym in s:
            if sym == '(':
                count += 1
            elif sym == ')':
                count -= 1
            elif count == 0:
                if sym in '*/':
                    if len(token) != 0:
                        res.trees.append(token)
                    token = ''
            token += sym
        if count < 0:
            raise SyntaxError("')' was never closed in math expression")
        elif count > 0:
            raise SyntaxError("'(' was never closed in math expression")
        if len(token) == 1 and token in '*/':
            raise SyntaxError(f"Expected a mathematical expression after '{token}'")
        elif len(token) != 0:
            res.trees.append(token)
        if len(res.trees) > 0:
            if len(res.trees) == 1:
                if res.trees[0][0] == '/':
                    res.type = T_DIV
                    res.trees[0] = self.expression(res.trees[0][1:], loc_vars, warnings)
                    return res
                if res.trees[0][0] == '*':
                    s = res.trees[0][1:]
                else:
                    s = res.trees[0]
            else:
                res.trees = [self.expression(tree, loc_vars, warnings) for tree in res.trees]
                return res
        #Возведение в степень:
        res.type = T_POW
        res.trees = None
        count = 0
        for p_pos, sym in enumerate(s):
            if sym == '(':
                count += 1
            elif sym == ')':
                count -= 1
            elif count == 0:
                if sym == '^':
                    break
        else:
            p_pos = -1
        if p_pos != -1:
            res.trees = [self.expression(s[:p_pos], loc_vars, warnings),
                         self.expression(s[p_pos + 1:], loc_vars, warnings)]
            return res
        #Встроенные функции:
        for n, func in enumerate(MainFuncs):
            if s[:len(func)] == func:
                if s[len(func)] != '(':
                    raise InvalidVariableNameError("Variable name cannot be the same as an existing function '{func}'")
                elif pairOf(s, len(func)) != len(s) - 1:
                    raise InvalidVariableNameError("Variable name can not consist of '(' and ')' symbols")
                res.type = F_SIN + n
                res.trees = [OperTree.expression(exp, loc_vars, warnings) for exp in s[len(func) + 1:-1].split(',') if exp != '']
                if len(res.trees) not in (args := MainFuncsParamsCount[func]):
                    warnings.append(f"Function '{func}' takes no arguments, but was given {len(res.trees)} arguments"
                                    if len(args) == 0 else
                                    f"Function '{func}' takes {list(args)[0]} only arguments, but was given {len(res.trees)} arguments"
                                    if len(args) == 1 else
                                    f"Function '{func}' takes {tuple(args)} only arguments, but was given {len(res.trees)} arguments")
                return res
        #Переменные и константы:
        res.val = number(s)
        if res.val is None:
            if (br := s.find('(')) == -1:
                if loc_vars is not None and s in loc_vars:
                    res.type = T_LOCVAR
                else:
                    res.type = T_VAR
                res.val = s
##                if locVars is None or s not in locVars:
##                    if VARS.get(s, None) is None:
##                        VARS[s] = OperTree(T_CONST, None, 0)
            else:
                if pairOf(s, br) != len(s) - 1:
                    raise InvalidVariableNameError("Variable name can not consist of '(' and ')' symbols")
                res.type = F_FUNC
                res.val = s[:br]
                res.trees = [OperTree.expression(exp, loc_vars, warnings) for exp in s[br + 1:-1].split(',') if exp != '']
                if res.val in FUNCS:
                    if len(res.trees) != len(FUNCS[res.val].vars):
                        warnings.append(f"Function '{func}' takes one only argument, but was given {len(res.trees)} arguments")
                else:
                    warnings.append(f"There is no function '{func}'")
        else:
            res.type = T_CONST
        return res

    def sort(self,
             expr: OperTree) -> None:
        """Сортирует дерево, раскрывает лишние скобки. Нужно перед сравнением."""
        if type(expr) is not OperTree:
            raise TypeError('"expr" must have "OperTree" type')
        if expr.trees is not None:
            for tree in expr.trees:
                self.sort(tree)
        if expr.type in {T_CONST, T_VAR, T_LOCVAR, F_SIN, F_COS, F_TAN, F_COTAN, F_LOG, F_FUNC}:
            return
        elif expr.type in {T_ADD, T_SUB, T_MUL, T_DIV}:
            if expr.type in self.comm:
                if expr.type in self.asso:
                    trees = []
                    for tree in expr.trees:
                        if tree.type == expr.type:
                            trees += tree.trees
                        else:
                            trees.append(tree)
                    sort_list(trees, cmp)
                    expr.trees = trees
                else:
                    if len(expr.trees) > 1:
                        if (expr.trees[1] < expr.trees[0] or
                            expr.trees[0].type != expr.type and expr.trees[1].type == expr.type):
                            expr.trees[0], expr.trees[1] = expr.trees[1], expr.trees[0]
                        if expr.trees[0].type == expr.type:
                            expr.trees = expr.trees[0].trees + expr.trees[1:]
                    elif len(expr.trees) == 1 and expr.trees[0].type == expr.type:
                        expr.trees = expr.trees[0].trees
            elif expr.trees[0].type == expr.type:
                if len(expr.trees) > 1:
                    expr.trees = expr.trees[0].trees + expr.trees[1:]
                elif len(expr.trees) == 1:
                    expr.trees = expr.trees[0].trees

    def calc_consts(self,
                    expr: OperTree,
                    **kwargs) -> None:
        """Считает все константные выражения в выражении, учитывая текущую алгебру."""
        if type(expr) is not OperTree:
            raise TypeError('"expr" must have "OperTree" type')
        if expr.trees is not None:
            for tree in expr.trees:
                self.calc_consts(tree, **kwargs)
        for obj in {'f_add', 'f_mul', 'f_pos', 'f_neg', 'f_ide', 'f_inv', 'zero', 'unit'}:
            exec(f'{obj} = kwargs.get("{obj}", {obj})')
        if expr.type in {T_CONST, T_VAR, T_LOCVAR, F_SIN, F_COS, F_TAN, F_COTAN, F_LOG, F_FUNC}:
            return
        else:
            for oper, func, unary, neutral in zip([T_ADD, T_SUB, T_MUL, T_DIV],
                                                  [f_add, f_add, f_mul, f_mul],
                                                  [f_pos, f_neg, f_ide, f_inv],
                                                  [zero,  zero,  unit,  unit]):
                if expr.type is oper:
                    if oper in self.asso:
                        if oper in self.comm:
                            const = neutral
                            for tree in expr.trees:
                                if tree.type in {T_FLOAT, T_CONST}:
                                    const = tree.val if const is None else func(const, tree.val)
                                    tree.val = ...
                            expr.trees = [OperTree((T_CONST if type(const) is ratio else T_FLOAT), None, const)] + expr.trees
                        else:
                            const = None
                            for tree in expr.trees:
                                if tree.type in {T_FLOAT, T_CONST}:
                                    if const is None:
                                        const = tree
                                    else:
                                        const.val = func(const.val, tree.val)
                                        tree.val = ...
                                else:
                                    const = None
                    elif len(expr.trees) > 1:
                        const = expr.trees[0]
                        if const.type in {T_FLOAT, T_CONST}:
                            for tree in expr.trees[1:]:
                                if tree.type in {T_FLOAT, T_CONST}:
                                    const.val = func(const.val, tree.val)
                                    tree.val = ...
                                else:
                                    break
                    expr.trees = [tree for tree in expr.trees if tree.val not in {..., neutral}]
                    if len(expr.trees) == 0:
                        expr.type = T_CONST
                        expr.trees = None
                        expr.val = neutral
                    elif len(expr.trees) == 1:
                        expr.type = expr.trees[0].type
                        expr.val = unary(expr.trees[0].val)
                        expr.trees = expr.trees[0].trees
                    break
            else:
                if expr.type is T_POW:
                    return
                    if expr.trees[0].type in {T_FLOAT, T_CONST} and expr.trees[1].type in {T_FLOAT, T_CONST}:
                        const = f_pow(expr.trees[0].val, expr.trees[1].val)
                        expr.type = T_CONST if type(const) is ratio else T_FLOAT
                        expr.trees = None
                        expr.val = const
                else:
                    raise UndefinedOperationError(f"Operation with number {expr.type} is undefined")

    def replace(self,
                expr: OperTree,
                old_expr: OperTree,
                new_expr: OperTree,
                _vars: {'str': OperTree} = None,
                _replace: bool = True) -> bool:
        """Подставляет "new_expr" вместо "old_expr" в "expr", учитывая алгебру. Все деревья должны быть отсортированы в той алгебре, в которой вы считаете!!!"""
        if any(type(e) is not OperTree for e in {expr, old_expr, new_expr}):
            raise TypeError('All parametrs must have "OperTree" type')
        if _vars is None:
            _vars = {}
        if self.trees is not None:
            for tree in self.trees:
                self.replace(tree, old_expr, new_expr)
        if old_expr.type is T_LOCVAR:
            if old_expr.val in _vars:
                if expr == _vars[old_expr.val]:
                    return True
                else:
                    return False
            else:
                _vars[old_expr.val] = expr.copy()
                return True
        elif old_expr.type in {T_FLOAT, T_CONST, T_VAR}:
            return old_expr == expr
        elif old_expr.type == expr.type:
            if expr.type in {T_ADD, T_SUB, T_MUL, T_DIV, F_SIN, F_COS, F_TAN, F_COTAN, F_LOG}:
                if len(expr.trees) < len(old_expr.trees):
                    return False
                if expr.type in self.asso:
                    if expr.type in self.comm:
                        ...
                    else:
                        ...
                elif expr.type in self.comm:
                    ...
                else:
                    ...
            elif expr.type == T_POW:
                ...
            elif expr.type == F_FUNC:
                ...
            else:
                raise UndefinedOperationError(f"Operation with number {expr.type} is undefined")

def objects_count(expr: OperTree) -> int:
    """Возвращает количество объектов в математическом выражении."""
    if expr is None:
        return 0
    if expr.type in {T_FLOAT, T_CONST, T_VAR, T_LOCVAR}:
        return 1
    return 1 + sum(objects_count(t) for t in expr.trees)

def random(d: dict, rnd: 'int|Random' = None):
    """Возвращает случайный объект из словаря, учитывая коэффициент случайности."""
    if rnd is None:
        rnd = Random()
    elif type(rnd) is int:
        rnd = Random(rnd)
    s = sum(d[t] for t in d)
    r = rnd.random() * s
    for obj in d:
        r -= d[obj]
        if r <= 0:
            return obj

def combinations(objs: [object, ...],
                 count: int) -> generator:
    """Возвращает генератор комбинаций из count числа объектов из списка объектов."""
    if (g := type(objs)) is not list:
        raise TypeError(f'"objs" must have type list, but "{g}" is given')
    if (g := type(count)) is not int:
        raise TypeError(f'"count" must have type int, but "{g}" is given')
    if len(objs) < count:
        raise ValueError(f'Count of objects in list "objs" must be not less then "count"')
    indices = list(range(count))  # Начальные индексы для первой комбинации
    length = len(objs)
    
    while True:
        yield [objs[i] for i in indices]  # Возвращаем текущую комбинацию
        
        # Найдем первый индекс, который можно увеличить
        for i in reversed(range(count)):
            if indices[i] != i + length - count:
                break
        else:
            return  # Все комбинации исчерпаны
        
        # Увеличиваем найденный индекс
        indices[i] += 1
        
        # Устанавливаем следующие индексы после увеличенного
        for j in range(i + 1, count):
            indices[j] = indices[j - 1] + 1

def decimal(q: ratio,
            d: "int|float('inf')" = 3,
            period: bool = False) -> str:
    """Если есть возможность, представляет число как десятичное с "d" цифрами после запятой (можно и с периодом (пока нельзя (уже можно)))."""
    if q.numerator < 0:
        return '-' + decimal(-q, d, period)
    den = q.denumerator
    k2 = k5 = 0
    mod = 0
    while k2 < d:
        prev = den
        den, mod = divmod(den, 2)
        if mod:
            den = prev
            break
        k2 += 1
    while k5 < d:
        prev = den
        den, mod = divmod(den, 5)
        if mod:
            den = prev
            break
        k5 += 1
    if den == 1:
        mul = 1
        k10 = k2
        if k2 < k5:
            k10 = k5
            mul = 2 ** (k5 - k2)
        elif k2 > k5:
            mul = 5 ** (k2 - k5)
        num, den = q.numerator, q.denumerator
        num *= mul
        den *= mul
        i, r = divmod(num, den)
        return f'{i}.' + f'{r:{k10}}'.replace(' ', '0')
    elif period and d == float('inf'):
        div = den
        mul = 1
        k10 = k2
        if k2 < k5:
            k10 = k5
            mul = 2 ** (k5 - k2)
        elif k2 > k5:
            mul = 5 ** (k2 - k5)
        num, den = q.numerator, q.denumerator
        num *= mul
        den = den * mul // div
        num, ost = divmod(num, div)
        i, r = divmod(num, den)
        per = 0
        k_per = 0
        rat = ratio(ost, div)
        p10 = 1
        while True:
            ost *= 10
            m, ost = divmod(ost, div)
            per = per * 10 + m
            k_per += 1
            p10 *= 10
            if ratio(per, p10 - 1) == rat:
                break
        return f'{i}.' + ('' if k10 == 0 else f'{r:{k10}}'.replace(' ', '0')) + f'({per:{k_per}})'.replace(' ', '0')
    return str(q)

if __name__ == '__main__':
    if False:
        rats = [ratio(n, d) for n, d in [(0, 1), (1, 2), (4, 3), (5, 100), (7, 9), (9, 7)]]
        for q in rats:
            print(f'{q} = {decimal(q, float("inf"), True)}\n' +
                  f'{-q} = {decimal(-q, float("inf"), True)}')
    if False:
        alg_simp = Algebra()
        alg_comm = Algebra(None, {T_ADD, T_SUB, T_MUL, T_DIV})
        alg_asso = Algebra({T_ADD, T_SUB, T_MUL, T_DIV})
        alg_full = Algebra({T_ADD, T_SUB, T_MUL, T_DIV}, {T_ADD, T_SUB, T_MUL, T_DIV})
        Expr_str = ['7-3+a+3.4+6.6',
                    '0.01+abs(-3)/3/100*99+6+7+3-9-7-8']
        for s in Expr_str:
            expr = alg_simp.expression(s)
            expr_simp = expr.copy()
            expr_comm = expr.copy()
            expr_asso = expr.copy()
            expr_full = expr.copy()
            alg_simp.calc_consts(expr_simp)
            alg_comm.calc_consts(expr_comm)
            alg_asso.calc_consts(expr_asso)
            alg_full.calc_consts(expr_full)
            print('-' * 50 + '\n' +
                  f'{expr.str_tree()}{expr} in algebra:\n' +
                  f'Simple: {expr_simp}\n{expr_simp.str_tree()}' +
                  f'Commutative: {expr_comm}\n{expr_comm.str_tree()}' +
                  f'Associative: {expr_asso}\n{expr_asso.str_tree()}' +
                  f'Full: {expr_full}\n{expr_full.str_tree()}')
    if False:
        alg = Algebra()
        for i in range(10):
            expr = alg.random(lvls = [5, 10],
                              brs = [5, 10],
                              rnd = i)
            print(f'seed: {i}\nBefore "simplify": {objects_count(expr) = }')
            simplify(expr)
            print(f'After "simplify": {objects_count(expr) = }\n')
