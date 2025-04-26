from Rational import *
from math import sin, cos, tan, asin, acos, atan, log, pi, e

half_pi = pi / 2

def cotan(x: float) -> float:
    """Return the cotangent of x (measured in radians)."""
    return 1 / tan(x)

def acotan(x: float) -> float:
    """Return the arc cotangent (measured in radians) of x."""
    return half_pi - atan(x)

class InvalidVariableNameError(Exception):
    """Неверное имя переменной или функции."""
    
    def __init__(self, message = "Invalid variable name"):
        super().__init__(message)

class ImpossibleActionError(Exception):
    """Невозможное действие."""
    
    def __init__(self, message = "Action is impossible"):
        super().__init__(message)

class RecursiveDefinitionError(Exception):
    """Рекурсивное определение."""
    
    def __init__(self, message = "There is recursive definition"):
        super().__init__(message)

class UndefinedParameterError(Exception):
    """Неопределённый параметр."""
    
    def __init__(self, message = "Parameter is undefined"):
        super().__init__(message)

class UndefinedOperationError(Exception):
    """Неопределённая операция."""
    
    def __init__(self, message = "Operation is undefined"):
        super().__init__(message)

function = type(round)

#Типы операций:
T_FLOAT = -1
T_CONST = 0
T_VAR = 1
T_LOCVAR = 2
F_SIN = 3
F_COS = 4
F_TAN = 5
F_COTAN = 6
F_LOG = 7
F_FUNC = 8
T_POW = 9
T_DIV = 10
T_MUL = 11
T_SUB = 12
T_ADD = 13

#Имена основных функций:
MainFuncs = ['sin', 'cos', 'tan', 'cotan', 'log']
MainFuncsParamsCount = {'sin': {1}, 'cos': {1}, 'tan': {1}, 'cotan': {1}, 'log': {1, 2}}

#Список глобальных переменных:
VARS = {}

#Список произвольных функций:
FUNCS = {}

class OperTree:
    """Операционное время."""

    def __init__(self, Type: int, OTs: list = None, val: object = None, Vars: list = None):
        self.type = Type
        self.trees = OTs
        self.val = val
        self.vars = Vars

    def copy(self):
        res = OperTree(self.type, None, self.val, self.vars)
        if self.trees is not None:
            res.trees = []
            for tree in self.trees:
                res.trees.append(tree.copy())
        return res

    def expression(s: str, locVars: list = None, warnings: list = None):
        """Возвращает операционное дерево, построенное по заданной строке."""
        if warnings is None:
            warnings = []
        res = OperTree(T_CONST)
        s = s.replace(' ', '')
        while pairOf(s, 0) == len(s) - 1:
            s = s[1:-1]
        if s == '':
            raise ValueError(f"s does not contain a mathematical expression")
        #Сложения и вычитания:
        while '-+' in s:
            s = s.replace('-+', '-')
        while '+-' in s:
            s = s.replace('+-', '-')
        while '--' in s:
            s = s.replace('--', '+')
        while '++' in s:
            s = s.replace('++', '+')
        while '+-' in s:
            s = s.replace('+-', '-')
        count = 0
        poss = []
        negs = []
        token = ''
        for sym in s:
            if sym == '(':
                count += 1
            elif sym == ')':
                count -= 1
            elif count == 0:
                if sym in '+-':
                    if len(token) == 0:
                        ...
                    elif token[0] == '-':
                        negs.append(token[1:])
                    elif token[0] == '+':
                        poss.append(token[1:])
                    else:
                        poss.append(token)
                    token = ''
            token += sym
        if count < 0:
            raise SyntaxError("')' was never closed in math expression")
        elif count > 0:
            raise SyntaxError("'(' was never closed in math expression")
        if len(poss) != 0 or len(negs) != 0 or token[0] in '+-':
            if token[0] == '-':
                negs.append(token[1:])
            elif token[0] == '+':
                poss.append(token[1:])
            else:
                poss.append(token)
            if len(poss) == 0:
                res.type = T_SUB
                res.trees = [OperTree.expression(exp, locVars, warnings) for exp in negs]
            elif len(negs) == 0:
                res.type = T_ADD
                res.trees = [OperTree.expression(exp, locVars, warnings) for exp in poss]
            else:
                res.type = T_ADD
                res.trees = [OperTree.expression(exp, locVars, warnings) for exp in poss]
                res.trees.append(OperTree(T_SUB, [OperTree.expression(exp, locVars, warnings) for exp in negs]))
            return res
        #Умножения и деления:
        while '/*' in s:
            s = s.replace('/*', '/')
        while '*/' in s:
            s = s.replace('*/', '/')
        while '//' in s:
            s = s.replace('//', '*')
        while '**' in s:
            s = s.replace('**', '*')
        while '*/' in s:
            s = s.replace('*/', '/')
        count = 0
        poss = []
        negs = []
        token = ''
        for sym in s:
            if sym == '(':
                count += 1
            elif sym == ')':
                count -= 1
            if count == 0:
                if sym in '*/':
                    if len(token) == 0:
                        ...
                    elif token[0] == '/':
                        negs.append(token[1:])
                    elif token[0] == '*':
                        poss.append(token[1:])
                    else:
                        poss.append(token)
                    token = ''
            token += sym
##        if count < 0:
##            raise SyntaxError("')' was never closed in math expression")
##        elif count > 0:
##            raise SyntaxError("'(' was never closed in math expression")
        if len(poss) != 0 or len(negs) != 0 or token[0] in '*/':
            if token[0] == '/':
                negs.append(token[1:])
            elif token[0] == '*':
                poss.append(token[1:])
            else:
                poss.append(token)
            if len(poss) == 0:
                res.type = T_DIV
                res.trees = [OperTree.expression(exp, locVars, warnings) for exp in negs]
            elif len(negs) == 0:
                res.type = T_MUL
                res.trees = [OperTree.expression(exp, locVars, warnings) for exp in poss]
            else:
                res.type = T_MUL
                res.trees = [OperTree.expression(exp, locVars, warnings) for exp in poss]
                res.trees.append(OperTree(T_DIV, [OperTree.expression(exp, locVars, warnings) for exp in negs]))
            return res
        #Возведения в степень:
        count = 0
        vals = []
        token = ''
        for sym in s:
            if sym == '(':
                count += 1
            elif sym == ')':
                count -= 1
            if count == 0:
                if sym in '^':
                    if token[0] == '^':
                        vals.append(token[1:])
                    else:
                        vals.append(token)
                    token = ''
            token += sym
##        if count < 0:
##            raise SyntaxError("')' was never closed in math expression")
##        elif count > 0:
##            raise SyntaxError("'(' was never closed in math expression")
        if len(vals) != 0:
            if token[0] == '^':
                vals.append(token[1:])
            else:
                vals.append(token)
            res.type = T_POW
            cur = res
            cur.trees = [OperTree.expression(vals[0], locVars, warnings), None]
            for exp in vals[1:-1]:
                cur.trees[1] = OperTree(T_POW, [OperTree.expression(exp, locVars, warnings), None])
                cur = cur.trees[1]
            cur.trees[1] = OperTree.expression(vals[-1], locVars, warnings)
##            res.trees = [OperTree.expression(exp, locVars) for exp in vals]
            return res
        #Основные функции:
        for n, func in enumerate(MainFuncs):
            if s[:len(func)] == func:
                if s[len(func)] != '(':
                    raise InvalidVariableNameError("Variable name cannot be the same as an existing function '{func}'")
                elif pairOf(s, len(func)) != len(s) - 1:
                    raise InvalidVariableNameError("Variable name can not consist of '(' and ')' symbols")
                res.type = F_SIN + n
                res.trees = [OperTree.expression(exp, locVars, warnings) for exp in s[len(func) + 1:-1].split(',') if exp != '']
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
                if locVars is not None and s in locVars:
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
                res.trees = [OperTree.expression(exp, locVars, warnings) for exp in s[br + 1:-1].split(',') if exp != '']
                if res.val in FUNCS:
                    if len(res.trees) != len(FUNCS[res.val].vars):
                        warnings.append(f"Function '{func}' takes one only argument, but was given {len(res.trees)} arguments")
                else:
                    warnings.append(f"There is no function '{func}'")
        else:
            res.type = T_CONST
        return res

    def funcs_replacement(self, Vars: {str: ['OperTree', int]} = None):
        """Заменяет похожие выражения на одни и те же переменные."""
        if self.vars is not None:
            raise ImpossibleActionError("The method 'funcs_replacement' can not be implied to a function")
        if Vars is None:
            Vars = {}
        if self.type in {T_FLOAT, T_CONST, T_VAR}:
            return
        if self.trees is not None:
            for tree in self.trees:
                tree.funcs_replacement(Vars)
        for op in Vars:
            if self == Vars[op][0]:
                Vars[op][1] += 1
                self.type = T_VAR
                self.trees = None
                self.val = op
                break
        else:
            name = f'__oper({self.type}|{len(Vars)})__'
            Vars[name] = [self.copy(), 1]
            self.type = T_VAR
            self.trees = None
            self.val = name

    def execute(self, Vars: {str: 'OperTree | ratio | float'} = None, prev_vars: {str} = None) -> 'ratio | float':
        """Возвращает результат выполнения операций."""
        if self.vars is not None:
            raise ImpossibleActionError("The method 'execute' can not be implied to a function")
        if prev_vars is None:
            prev_vars = set()
        if Vars is None:
            Vars = {}
        if self.type in {T_FLOAT, T_CONST}:
            return self.val
        elif self.type is T_VAR:
            if self.val in prev_vars:
                raise RecursiveDefinitionError(f"The variable '{self.val}' is defined recursively")
            if self.val not in Vars:
                raise ImpossibleActionError(f"The variable '{self.val}' is not known")
            elif type(Vars[self.val]) in [ratio, float]:
                return Vars[self.val]
            else:
                Vars[self.val] = Vars[self.val].execute(Vars, prev_vars | {self.val})
                return Vars[self.val]
        elif self.type in {T_ADD, T_SUB}:
            res = sum(tree.execute(Vars, prev_vars) for tree in self.trees)
            if self.type is T_ADD:
                return res
            else:
                return -res
        elif self.type in {T_MUL, T_DIV}:
            res = ratio(1)
            for tree in self.trees:
                res *= tree.execute(Vars, prev_vars)
            if self.type is T_MUL:
                return res
            else:
                return 1 / res
        elif self.type is T_POW:
            return self.trees[0].execute(Vars, prev_vars) ** self.trees[1].execute(Vars, prev_vars)
        elif self.type is F_SIN:
            return sin(float(self.trees[0].execute(Vars, prev_vars)))
        elif self.type is F_COS:
            return cos(float(self.trees[0].execute(Vars, prev_vars)))
        elif self.type is F_TAN:
            return tan(float(self.trees[0].execute(Vars, prev_vars)))
        elif self.type is F_COTAN:
            return cotan(float(self.trees[0].execute(Vars, prev_vars)))
        elif self.type is F_LOG:
            return log(float(self.trees[0].execute(Vars, prev_vars)))
        elif self.type is F_FUNC:
            cop = Vars[self.val].copy()
            subs(cop, {p: tree for p, tree in zip(cop.vars, self.trees)})
            return cop.execute(Vars, prev_vars | {self.val})

    def __eq__(self, other):
        return cmp(self, other) == 0

    def __ne__(self, other):
        return cmp(self, other) != 0

    def __lt__(self, other):
        return cmp(self, other) < 0

    def __le__(self, other):
        return cmp(self, other) <= 0

    def __gt__(self, other):
        return cmp(self, other) > 0

    def __ge__(self, other):
        return cmp(self, other) >= 0

    def str_tree(self, lasts: list = None, last: bool = False):
        """Возвращает дерево на боку в виде текста."""
        if lasts is None:
            lasts = []
        res = ''
        if len(lasts) != 0:
            for l in lasts[:-1]:
                if l:
                    res += '│'
                else:
                    res += ' '
            res += '└' if last else '├'
        if self.type in {T_FLOAT, T_CONST, T_VAR, T_LOCVAR}:
            res += f'[{self.val}]'
        elif self.type == T_ADD:
            res += '{+}'
        elif self.type == T_SUB:
            res += '{-}'
        elif self.type == T_MUL:
            res += '{*}'
        elif self.type == T_DIV:
            res += '{/}'
        elif self.type == T_POW:
            res += '{^}'
        elif self.type == F_SIN:
            res += '[sin]'
        elif self.type == F_COS:
            res += '[cos]'
        elif self.type == F_TAN:
            res += '[tan]'
        elif self.type == F_COTAN:
            res += '[cotan]'
        elif self.type == F_LOG:
            res += '[log]'
        elif self.type == F_FUNC:
            res += '{' + self.val + '}'
        else:
            res += '{?}'
        if self.vars is not None:
            res += str(tuple(self.vars)) if len(self.vars) > 1 else f'({self.vars[0]})' if len(self.vars) == 1 else '()'
        res += '\n'
        if self.trees is not None:
            lasts.append(True)
            if len(self.trees) != 0:
                for tree in self.trees[:-1]:
                    res += tree.str_tree(lasts)
            lasts[-1] = False
            res += self.trees[-1].str_tree(lasts, True)
            del lasts[-1]
        return res

    def __str__(self):
        res = ''
        if self.type in {T_FLOAT, T_CONST, T_VAR, T_LOCVAR}:
            res = str(self.val)
        elif self.type == T_ADD:
            if self.trees is None or len(self.trees) == 0:
                res = '?+?'
            else:
                if self.trees[0].type == T_ADD:
                    res = '(' + str(self.trees[0]) + ')'
                else:
                    res = str(self.trees[0])
                if len(self.trees) > 1:
                    for tree in self.trees[1:]:
                        if tree.type == T_ADD:
                            res += '+(' + str(tree) + ')'
                        elif tree.type == T_SUB:
                            res += str(tree)
                        else:
                            res += '+' + str(tree)
        elif self.type == T_SUB:
            if self.trees is None or len(self.trees) == 0:
                res = '?-?'
            else:
                if self.trees[0].type in {T_ADD, T_SUB}:
                    res = '-(' + str(self.trees[0]) + ')'
                else:
                    res = '-' + str(self.trees[0])
                if len(self.trees) > 1:
                    for tree in self.trees[1:]:
                        if tree.type in {T_ADD, T_SUB}:
                            res += '-(' + str(tree) + ')'
                        else:
                            res += '-' + str(tree)
        elif self.type == T_MUL:
            if self.trees is None or len(self.trees) == 0:
                res = '?*?'
            else:
                if self.trees[0].type in {T_ADD, T_SUB, T_MUL}:
                    res = '(' + str(self.trees[0]) + ')'
                else:
                    res = str(self.trees[0])
                if len(self.trees) > 1:
                    for tree in self.trees[1:]:
                        if tree.type in {T_ADD, T_SUB, T_MUL}:
                            res += '*(' + str(tree) + ')'
                        elif tree.type == T_DIV:
                            res += str(tree)[1 :]
                        else:
                            res += '*' + str(tree)
        elif self.type == T_DIV:
            if self.trees is None or len(self.trees) == 0:
                res = '?/?'
            else:
                if (self.trees[0].type in {T_ADD, T_SUB, T_MUL, T_DIV} or
                    self.trees[0].type is T_CONST and self.trees[0].val.denumerator != 1):
                    res = '/(' + str(self.trees[0]) + ')'
                else:
                    res = '/' + str(self.trees[0])
                if len(self.trees) > 1:
                    for tree in self.trees[1:]:
                        if (tree.type in {T_ADD, T_SUB, T_MUL, T_DIV} or
                            tree.type is T_CONST and tree.val.denumerator != 1):
                            res += '/(' + str(tree) + ')'
                        else:
                            res += '/' + str(tree)
        elif self.type == T_POW:
            if self.trees is None or len(self.trees) == 0:
                res = '?^?'
            else:
                if self.trees[0].type in {T_ADD, T_SUB, T_MUL, T_DIV, T_POW}:
                    res = '(' + str(self.trees[0]) + ')'
                else:
                    res = str(self.trees[0])
                if len(self.trees) > 1:
                    for tree in self.trees[1:]:
                        if tree.type in {T_ADD, T_SUB, T_MUL, T_DIV, T_POW}:
                            res += '^(' + str(tree) + ')'
                        else:
                            res += '^' + str(tree)
        elif self.type in {F_SIN, F_COS, F_TAN, F_COTAN, F_LOG, F_FUNC}:
            if self.type == F_SIN:
                res = 'sin('
            elif self.type == F_COS:
                res = 'cos('
            elif self.type == F_TAN:
                res = 'tan('
            elif self.type == F_COTAN:
                res = 'cotan('
            elif self.type == F_LOG:
                res = 'log('
            elif self.type == F_FUNC:
                res = self.val + '('
            if self.trees is None:
                res += '???)'
            elif len(self.trees) == 0:
                res += ')'
            else:
                res += str(self.trees[0])
                if len(self.trees) > 1:
                    for tree in self.trees[1:]:
                        res += ', ' + str(tree)
                res += ')'
        else:
            res = '???'
        if res[0] == '/':
            res = '1' + res
        res = res.replace('(/', '(1/')
        return res

    def __repr__(self):
        return f'OperTree.expression(\'{str(self)}\')'

def new_function(name: str, s: str = None, params: list = None) -> OperTree:
    """Строит операционное дерево из строки "s" с параметрами "params". Возвращает операционное дерево с параметрами.
Если существуют глобальные переменные с именами из списка "params", они игнорируются при построении дерева.
(name: str) -> OperTree
Возвращает существующую функцию "name", если она есть, иначе создаёт нулевую и возвращает её."""
    if s is None:
        if (res := FUNCS.get(name, None)) is None:
            res = OperTree(T_CONST, None, ratio(0), [])
            FUNCS[name] = res
        return res
    res = OperTree.expression(s, params)
    res.vars = params
    if FUNCS.get(name, None) is not None:
        del FUNCS[name]
    FUNCS[name] = res
    return res

def pairOf(s: str, pos: int):
    """Возвращает парную скобку для данной. Если такой нет, то возвращает None."""
    count = 0
    res = pos
    if pos < 0 or pos >= len(s):
        return None
    if s[pos] == '(':
        count = 1
        for sym in s[pos + 1:]:
            res += 1
            if sym == '(':
                count += 1
            elif sym == ')':
                count -= 1
            if count == 0:
                return res
    elif s[pos] == ')':
        count = -1
        for sym in s[:pos][::-1]:
            res -= 1
            if sym == '(':
                count += 1
            elif sym == ')':
                count -= 1
            if count == 0:
                return res
    return None

def number_type(s: str):
    """Возвращает тип числа. Если это не число, возвращает None."""
    point = False
    digits = '0123456789'
    if s[0] == '-':
        s = s[1:]
    for sym in s:
        if sym == '.':
            if point:
                return None
            point = True
        elif sym not in digits:
            return None
    if point:
        return ratio
    return int

def number(s: str) -> ratio:
    """Возвращает рациональное число из строки вида: "'123.456789'", "'123456789'". Если нельзя построить дробь, то возвращает None."""
    t = number_type(s)
    if t is int:
        return ratio(int(s), 1)
    elif t is ratio:
        sl, sr = s.split('.')
        dec = 10 ** len(sr)
        return ratio((0 if sl == '' else int(sl) * dec) + (0 if sr == '' else int(sr)), dec)
    return None

def cmp(a: OperTree, b: OperTree, cmp_rule = None) -> int:
    """Сравнивает деревья на порядок. Возвращает -1, если дерево "a" стоит раньше, чем дерево "b", 0 - если они индетичны, и 1 - иначе."""
    if cmp_rule is None:
        cmp_rule = cmp
    if a is None and b is None:
        return 0
    elif a is None:
        return -1
    elif b is None:
        return 1
    else:
        if a.type < b.type:
            return -1
        elif a.type > b.type:
            return 1
        else:
            if a.type in {T_CONST, T_VAR, T_LOCVAR}:
                if a.val < b.val:
                    return -1
                elif a.val > b.val:
                    return 1
                else:
                    return 0
            else:
                if a.trees is None and b.trees is None:
                    return 0
                elif a.trees is None:
                    return -1
                elif b.trees is None:
                    return 1
                else:
                    i = 0
                    while i < len(a.trees) and i < len(b.trees):
                        comp = cmp_rule(a.trees[i], b.trees[i])
                        if comp != 0:
                            return comp
                        i += 1
                    if len(a.trees) == len(b.trees):
                        return 0
                    elif len(a.trees) < len(b.trees):
                        return -1
                    else:
                        return 1

def sort_list(L: list, cmp: function):
    """Возвращает список, сортированный по функции сравнения, которая возвращает -1 - если первый параметр этой функции меньше второго, 1 - если наоборот, и 0 - если равны."""
    res = []
    while len(L) > 1:
        min_ind = 0
        min_obj = L[0]
        for n, obj in enumerate(L[1:]):
            if cmp(obj, min_obj) == -1:
                min_ind = n + 1
                min_obj = obj
        res.append(L.pop(min_ind))
    if len(L) == 1:
        res.append(L[0])
    L.clear()
    for obj in res:
        L.append(obj)

def sort(a: OperTree, cmp_rule = None):
    """Сортирует дерево."""
    if cmp_rule is None:
        cmp_rule = cmp
    if a is None or a.trees is None:
        return
    for tree in a.trees:
        sort(tree, cmp_rule)
    if a.type in {T_ADD, T_SUB, T_MUL, T_DIV}:
        sort_list(a.trees, cmp_rule)

def oper_union(a: OperTree):
    """Объеденяет операции сложения и умножения, раскрывая скобки в сумме сумм и в произведении произведений, и заменяет одноэлементный суммы и произведения на единственный оставшийся элемент."""
    if a is None or a.trees is None:
        return
    for tree in a.trees:
        oper_union(tree)
    poss = []
    negs = []
    if a.type in {T_ADD, T_SUB}:
        if a.trees is None:
            a.type = T_CONST
            a.val = ratio(0)
            return
        for tree in a.trees:
            if tree.type == T_ADD:
                if tree.trees is None:
                    continue
                for obj in tree.trees:
                    if obj.type is T_SUB:
                        if obj.trees is None:
                            continue
                        negs += obj.trees
                    else:
                        poss.append(obj)
            elif tree.type == T_SUB:
                if tree.trees is None:
                    continue
                negs += tree.trees
            else:
                poss.append(tree)
        if a.type == T_SUB:
            poss, negs = negs, poss
        if len(poss) == 0:
            a.type = T_SUB
            a.trees = negs
        else:
            a.type = T_ADD
            a.trees = poss
            if len(negs) != 0:
                a.trees.append(OperTree(T_SUB, negs))
            if len(a.trees) == 1:
                a.type = a.trees[0].type
                a.val = a.trees[0].val
                a.trees = a.trees[0].trees
    elif a.type in {T_MUL, T_DIV}:
        if a.trees is None:
            a.type = T_CONST
            a.val = ratio(1)
            return
        for tree in a.trees:
            if tree.type == T_MUL:
                if tree.trees is None:
                    continue
                for obj in tree.trees:
                    if obj.type is T_DIV:
                        if obj.trees is None:
                            continue
                        negs += obj.trees
                    else:
                        poss.append(obj)
            elif tree.type == T_DIV:
                if tree.trees is None:
                    continue
                negs += tree.trees
            else:
                poss.append(tree)
        if a.type == T_DIV:
            poss, negs = negs, poss
        if len(poss) == 0:
            a.type = T_DIV
            a.trees = negs
        else:
            a.type = T_MUL
            a.trees = poss
            if len(negs) != 0:
                a.trees.append(OperTree(T_DIV, negs))
            if len(a.trees) == 1:
                a.type = a.trees[0].type
                a.val = a.trees[0].val
                a.trees = a.trees[0].trees

#Используйте улучшенную версию: "calc_consts"!
def del_neutrals(a: OperTree):
    """Удаляет нейтральные элементы."""
    if a is None or a.trees is None:
        return
    for tree in a.trees:
        del_neutrals(tree)
    trees = []
    if a.type in {T_ADD, T_SUB}:
        neutral = ratio(0)
    elif a.type in {T_MUL, T_DIV}:
        neutral = ratio(1)
    else:
        return
    for tree in a.trees:
        if tree.type != T_CONST or tree.val != neutral:
            trees.append(tree)
    if len(trees) == 0:
        a.type = T_CONST
        a.trees = None
        a.val = neutral
    else:
        a.trees = trees

def calc_consts(a: OperTree):
    """Считает все константные выражения в дереве."""
    if a is None or a.trees is None:
        return
    for tree in a.trees:
        calc_consts(tree)
    if a.type in {T_ADD, T_SUB, T_MUL, T_DIV}:
        if a.type in {T_ADD, T_SUB}:
            neutral = ratio(0)
            oper = lambda x, y: x + y
            inv_oper = lambda x, y: x - y
            inv_type = T_SUB
        else:
            neutral = ratio(1)
            oper = lambda x, y: x * y
            inv_oper = lambda x, y: x / y
            inv_type = T_DIV
        const = neutral
        trees = []
        for tree in a.trees:
            if tree.type == inv_type:
                if tree.trees[0].type == T_CONST:
                    const = inv_oper(const, tree.trees[0].val)
                    tree.trees = tree.trees[1:]
                    if len(tree.trees) == 0:
                        del tree
                        continue
                    if len(tree.trees) == 1:
                        tree.type = tree.trees[0].type
                        tree.val = tree.trees[0].val
                        tree.trees = tree.trees[0].trees
                trees.append(tree)
            elif tree.type == T_CONST:
                const = oper(const, tree.val)
            else:
                trees.append(tree)
        if const == neutral:
            a.trees = trees
        else:
            a.trees = [OperTree(T_CONST, None, const)] + trees
        if len(a.trees) == 0:
            a.type = T_CONST
            a.trees = None
            a.val = neutral
        elif len(a.trees) == 1:
            if a.type in {T_ADD, T_MUL}:
                a.type = a.trees[0].type
                a.val = a.trees[0].val
                a.trees = a.trees[0].trees
            elif a.trees[0].type == T_CONST:
                if a.type == T_SUB:
                    a.type = T_CONST
                    a.val = -a.trees[0].val
                    a.trees = None
                else:
                    a.type = T_CONST
                    a.val = neutral / a.trees[0].val
                    a.trees = None
    elif a.type == T_POW:
        if a.trees[0].type == a.trees[1].type == T_CONST and a.trees[1].val.denumerator == 1:
            const = ratio(1)
            val = a.trees[0].val
            if (num := a.trees[1].val.numerator) < 0:
                val = const / a.trees[0].val
                num = -num
            for _ in range(num):
                const *= val
            a.type = T_CONST
            a.trees = None
            a.val = const
        elif a.trees[1].type == T_CONST and a.trees[1].val == ratio(1):
            a.type = a.trees[0].type
            a.val = a.trees[0].val
            a.trees = a.trees[0].trees

def are_proportionals(a: OperTree, b: OperTree) -> bool:
    """Проверяет на текущем узле пропорциональность деревьев (все неконстантные поддеревья должны полностью совпадать). Если "a" и "b" пропорциональны, возвращает True, иначе - False.
Стоит вызывать после "calc_consts" и "sort"."""
    if a is None and b is None:
        return True
    elif a is None or b is None:
        return False
    if a.type == T_MUL and b.type == T_MUL:
        if a.trees[0].type == T_CONST:
            atrees = a.trees[1:]
        else:
            atrees = a.trees
        if b.trees[0].type == T_CONST:
            btrees = b.trees[1:]
        else:
            btrees = b.trees
        return atrees == btrees
    elif a.type == T_MUL:
        if a.trees[0].type == T_CONST:
            atrees = a.trees[1:]
        else:
            atrees = a.trees
        return atrees == [b]
    elif b.type == T_MUL:
        if b.trees[0].type == T_CONST:
            btrees = b.trees[1:]
        else:
            btrees = b.trees
        return [a] == btrees
    else:
        return a == b

def calc_proportionals(a: OperTree):
    """Считает все пропорциональные выражения в дереве."""
    if a is None or a.trees is None:
        return
    for tree in a.trees:
        calc_proportionals(tree)
    if a.type in {T_ADD, T_SUB}:
        props = []
        for tree in a.trees:
            for prop in props:
                if are_proportionals(tree, prop[0]):
                    prop.append(tree)
                    break
            else:
                props.append([tree])
        trees = []
        for prop in props:
            unit = ratio(1)
            const = ratio(0)
            main = None
            if prop[0].type == T_MUL:
                if len(prop[0].trees) == 2:
                    main = OperTree(T_MUL, [prop[0].trees[1].copy()])
                else:
                    main = OperTree(T_MUL, [t.copy() for t in prop[0].trees[1:]])
            else:
                main = OperTree(T_MUL, [prop[0].copy()])
            for tree in prop:
                if tree.type == T_MUL:
                    if tree.trees[0].type == T_CONST:
                        const += tree.trees[0].val
                    else:
                        const += unit
                else:
                    const += unit
            main.trees = [OperTree(T_CONST, val = const)] + main.trees
            trees.append(main)
        a.trees = trees
        calc_consts(a)
calc_sum_eqs = calc_proportionals

def are_eq_in_mul(a: OperTree, b: OperTree) -> bool:
    """Проверяет на текущем узле равенство деревьев с точностью до возведения в степень."""
    if a is None and b is None:
        return True
    elif a is None or b is None:
        return False
    if a.type == T_POW and b.type == T_POW:
        return a.trees[0] == b.trees[0]
    elif a.type == T_POW:
        return a.trees[0] == b
    elif b.type == T_POW:
        return a == b.trees[0]
    else:
        return a == b

def calc_mul_eqs(a: OperTree):
    """Складывает степени одинаковых выражений в произведении."""
    if a is None or a.trees is None:
        return
    for tree in a.trees:
        calc_mul_eqs(tree)
    if a.type == T_MUL:
        eqs = []
        for tree in a.trees:
            for eq in eqs:
                if are_eq_in_mul(tree, eq[0]):
                    eq.append(tree)
                    break
            else:
                eqs.append([tree])
        trees = []
        for eq in eqs:
            const = ratio(0)
            unit = ratio(1)
            power = OperTree(T_ADD, [])
            main = OperTree(T_POW, [(eq[0].trees[0] if eq[0].type == T_POW else eq[0]), power])
            for tree in eq:
                if tree.type == T_POW:
                    power.trees.append(tree.trees[1])
                else:
                    const += unit
            power.trees.append(OperTree(T_CONST, None, const))
            trees.append(main)
        a.trees = trees
        calc_proportionals(a)

def simplify(a: OperTree):
    """Пременяет серию функций к выражению, упрощая его."""
    oper_union(a)
    calc_mul_eqs(a)
    calc_consts(a)
    calc_sum_eqs(a)
    sort(a)

def is_same(a: OperTree, b: OperTree, known: {str: OperTree} = None) -> {str: OperTree}:
    """Если дерево "b" схоже по структуре с деревом "a", возвращает словарь значений параметров для дерева "b", иначе - False."""
    if a is None or b is None:
        return False
    if known is None:
        known = {}
    if b.type in {T_CONST, T_VAR}:
        if a == b:
            return known
        else:
            return False
    elif b.type == T_LOCVAR:
        value = known.get(b.val, None)
        if value is None:
            known[b.val] = a.copy()
        elif a == value:
            return known
        else:
            return False
    elif a.type == b.type:
        if len(a.trees) == len(b.trees):
            for a_tree, b_tree in zip(a.trees, b.trees):
                res = is_same(a_tree, b_tree, known)
                if res is False:
                    return False
            return known
        else:
            return False
    else:
        return False

def subs(main: OperTree, params: {str: OperTree}):
    """Подставляет значения параметров вместо параметров."""
    if main is None:
        return
    if main.trees is not None:
        for tree in main.trees:
            subs(tree, params)
    elif main.type == T_LOCVAR:
        val = params.get(main.val, None)
        if val is None:
            raise Exception(f"The Parameter '{main.val}' is undefined")
        main.type = val.type
        main.val = val.val
        main.trees = val.trees
    main.vars = None

def replace(main: OperTree, old: OperTree, new: OperTree):
    """Заменяет все поддеревья old на new в дереве main.
Если в old есть параметры (они должны совпадать с параметрами дерева new), то делает замену с точностью до выражения."""
    if main is None:
        return
    if old.vars != new.vars:
        raise Exception('old and new has different params.')
    if main.trees is not None:
        for tree in main.trees:
            replace(tree, old, new)
    vs = is_same(main, old)
    if vs is False:
        return
    new_tree = new.copy()
    subs(new_tree, vs)
    main.type = new_tree.type
    main.val = new_tree.val
    main.trees = new_tree.trees

def apply_formulae(main: OperTree, *formulae: (OperTree, OperTree)):
    """main - дерево основного выражения,
formulae - список кортежей из двух функций преобразования из первой во вторую форму.
Применяет формулы "formulae" для дерева "main"."""
    for formula in formulae:
        replace(main, *formula)

def change_params(func: OperTree, **params: str):
    """Меняет имена параметров функции."""
    if func is None:
        return
    if func.vars == list(params.keys()):
        func.vars = list(params.values())
    params = {p: OperTree(T_LOCVAR, None, params[p]) for p in params}
    subs(func, params)

if __name__ == '__main__':
    if False:
        Expr_str = ['a+b',          'a+b',
                    '2*(a+b+3-3)',  'a+b',
                    'a+b',          '2*(a+b)',
                    '1.5*3*x*y*z',  '7.25*9.3*x*y*z',
                    '2*x*y*z',      '2*(x*y*z)',
                    '2*(a+b+c)',    '2*a+2*b+2*c',
                    'exp(x)',       '9*exp(2*x)',
                    'a+a+a+b+2*b',  '2*a+2*b']
        for i in range(len(Expr_str) // 2):
            A = OperTree.expression(Expr_str[2 * i])
            B = OperTree.expression(Expr_str[2 * i + 1])
            print(f'A = {A}\nB = {B}\n' + '-' * 50)
            oper_union(A)
            oper_union(B)
            calc_mul_eqs(A)
            calc_mul_eqs(B)
            calc_consts(A)
            calc_consts(B)
            calc_sum_eqs(A)
            calc_sum_eqs(B)
            sort(A)
            sort(B)
            print(f'A = {A}\nB = {B}\nA ~ B is {are_proportionals(A, B)}\n' + '-' * 50)
            print('A\n', A.str_tree(), '\nB\n', B.str_tree(), sep = '', end = '\n' + '=' * 50 + '\n')
    if True:
        Expr_str = ['\nПрименение свойства дистрибутивности с помощью формулы для суммы из трёх слагаемых',
                    '(a+b)*(x^y+1)+(a+b)*(sin(z)-7)+(a+b)*(5*x*y*z)',       'd*a+d*b+d*c',          'd*(a+b+c)',                ['a', 'b', 'c', 'd'],
                    '\nНеуспех при применении свойства дистрибутивности для суммы из трёх слагаемых, потому что "a+b" не то же самое, что и "b+a"',
                    '(a+b)*(x^y+1)+(b+a)*(sin(z)-7)+(a+b)*(5*x*y*z)',       'd*a+d*b+d*c',          'd*(a+b+c)',                ['a', 'b', 'c', 'd'],
                    '\nПрименение главного тригонометрического тождества',
                    'sin(x*y+z)^2+cos(x*y+z)^2',                            'sin(x)^2+cos(x)^2',    '1',                        ['x'],
                    '\nПрименение главного тригонометрического тождества',
                    'sin(x*y+z)^2+cos(x^2*y^2+z^2+(sin(y)^2+cos(y)^2))^2',  'sin(x)^2+cos(x)^2',    '1',                        ['x'],
                    '\nЗамена возведения в степень числа Эйлера на функцию "exp"',
                    '1+e^(x*y)',                                            'e^x',                  'exp(x)',                   ['x'],
                    
                    '\nСложение матриц 2x2, если аргументы в "matrix_2_2" идут в том же порядке, что и значения в матрице, если читать её сверху вниз строки, слева направо элементы строк',
                    'matrix_2_2(a, b, c, d)+matrix_2_2(e, f, g, h)',
                    'matrix_2_2(a_1_1, a_1_2, a_2_1, a_2_2)+matrix_2_2(b_1_1, b_1_2, b_2_1, b_2_2)',
                    'matrix_2_2(a_1_1+b_1_1, a_1_2+b_1_2, a_2_1+b_2_1, a_2_2+b_2_2)',
                    ['a_1_1', 'a_1_2', 'a_2_1', 'a_2_2', 'b_1_1', 'b_1_2', 'b_2_1', 'b_2_2'],

                    '\nПеремножение матриц 2x2, если аргументы в "matrix_2_2" идут в том же порядке, что и значения в матрице, если читать её сверху вниз строки, слева направо элементы строк',
                    'matrix_2_2(a, b, c, d)*matrix_2_2(e, f, g, h)',
                    'matrix_2_2(a_1_1, a_1_2, a_2_1, a_2_2)*matrix_2_2(b_1_1, b_1_2, b_2_1, b_2_2)',
                    'matrix_2_2(a_1_1*b_1_1+a_1_2*b_2_1, a_1_1*b_1_2+a_1_2*b_2_2, a_2_1*b_1_1+a_2_2*b_2_1, a_2_1*b_2_1+a_2_2*b_2_2)',
                    ['a_1_1', 'a_1_2', 'a_2_1', 'a_2_2', 'b_1_1', 'b_1_2', 'b_2_1', 'b_2_2'],

                    '\nСложение комплексных чисел',
                    '(a+b*i)+((c+t*9)+d^2*i)',                              '(a+b*i)+(c+d*i)',      '((a+c)+(b+d)*i)',          ['a', 'b', 'c', 'd'],
                    '\nПроизведение комплексных чисел',
                    '(a+b*i)*((c+t*9)+d^2*i)',                              '(a+b*i)*(c+d*i)',      '((a*c-b*d)+(a*d+b*c)*i)',  ['a', 'b', 'c', 'd'],
                    '\nВыделение полного квадрата из суммы из трёх элементов, где левый и правый элементы являются квадратами, а средний элемент их произведение в том же порядке, домноженное слева на константу',
                    'a^2+3*a*b+b^2',                                        'a^2+c*a*b+b^2',        '(a+b)^2+(c-2)*a*b',        ['a', 'b', 'c'],
                    '\nВыделение полного квадрата из суммы из трёх элементов, где левый и правый элементы являются квадратами, а средний элемент их произведение в том же порядке',
                    'a^2+a*b+b^2',                                          'a^2+a*b+b^2',          '(a+b)^2-a*b',              ['a', 'b'],
                    '\nДелаем биквадратное уравнение относительно сложного выражения',
                    '(b/4)*(sin(x+a)-1/2)^4+(2*b)*(sin(x+a)-1/2)^2+(3*b)',          'a^4',          '(a^2)^2',                  ['a'],
                    
                    'Затем заменяем это сложное выражение на переменную "t"',
                    '(b/4)*((sin(x+a)-1/2)^2)^2+(2*b)*(sin(x+a)-1/2)^2+(3*b)',      '(a-b)^2',                          't',                                            ['a', 'b'],
                    'Находим дискриминант квадратного уравнения (для этого нужно создать копию выражения, чтобы не испортить само выражение)',
                    '(b/4)*t^2+(2*b)*t+(3*b)',                                      'a*x^2+b*x+c',                      'b^2-4*a*c',                                    ['x', 'a', 'b', 'c'],
                    'После раскрытия порядковых скобок, приведения подобных и подсчёта констант получаем, что дискриминант равен "b^2", а общее решение такое, где D корень из дискриминанта',
                    '(b/4)*t^2+(2*b)*t+(3*b)',                                      'a*x^2+b*x+c',                      'solutions((2*b-D)/(2*a), (2*b+D)/(2*a))',      ['x', 'a', 'b', 'c'],
                    'После подстановки корня из дискриминанта получаем',
                    'solutions((8*b-D)/b, (8*b+D)/b)',                              'solutions((b2-D)/a2, (b2+D)/a2)',  'solutions((b2-abs(b))/a2, (b2+abs(b))/a2)',    ['a2', 'b2']]
        elems = 5
        for i in range(len(Expr_str) // elems):
            Description = Expr_str[elems * i]
            Main = OperTree.expression(Expr_str[elems * i + 1])
            Old = new_function(f'old{i}', Expr_str[elems * i + 2], Expr_str[elems * i + 4])
            New = new_function(f'new{i}', Expr_str[elems * i + 3], Expr_str[elems * i + 4])
            print(Description + ':\n', Main, '\n', Old, '->', New)
            replace(Main, Old, New)
            print(Main, '=' * 50, sep = '\n')
    if False:
        Expr_str = ['(a+b)+2*(a+b)+c*(a+b)',    {'a': '7^2', 'b': '5', 'c': '10^9'},
                    'sin(cos(tan(cotan(a))))',  {'a': 'b', 'b': 'c', 'c': '3'},
                    'det(a, b, c, d)',          {'a': 'cos(phi)', 'b': '-sin(phi)', 'c': 'sin(phi)', 'd': 'cos(phi)', 'phi': '25', 'det': ('a*d-b*c', ['a', 'b', 'c', 'd'])},
                    'a+b',                      {'a': 'b'},
                    'a+b',                      {'a': 'b', 'b': 'a'}]
        for i in range(len(Expr_str) // 2):
            Main = OperTree.expression(Expr_str[2 * i])
            Subs = {p: (OperTree.expression(Expr_str[2 * i + 1][p])
                        if type(Expr_str[2 * i + 1][p]) is str else
                        new_function(f'__({p}|{i})__', Expr_str[2 * i + 1][p][0], Expr_str[2 * i + 1][p][1]))
                    for p in Expr_str[2 * i + 1]}
            sbs = {p: Subs[p].copy() for p in Subs}
            try:
                print(f'{Main} = {Main.execute(Subs)}', *[f'{s}{"" if sbs[s].vars is None else tuple(sbs[s].vars)} = {sbs[s]}' for s in Subs], sep = '\n', end = '\n\n')
            except RecursiveDefinitionError as err:
                print(f'{Main}', *[f'{s}{"" if sbs[s].vars is None else tuple(sbs[s].vars)} = {sbs[s]}' for s in Subs], err, sep = '\n', end = '\n\n')
            except ImpossibleActionError as err:
                print(f'{Main}', *[f'{s}{"" if sbs[s].vars is None else tuple(sbs[s].vars)} = {sbs[s]}' for s in Subs], err, sep = '\n', end = '\n\n')
