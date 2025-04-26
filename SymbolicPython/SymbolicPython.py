import sympy
from Algebra import *

def to_sympy_Rational(q: ratio) -> sympy.Rational:
    """Преобразует рациональное значение из Rational в рациональное значение из SymPy и возвращает его."""
    return sympy.Rational(q.numerator, q.denumerator)

def to_Rational_ratio(q: sympy.Rational) -> ratio:
    """Преобразует рациональное значение из SymPy в рациональное значение из Rational и возвращает его."""
    return ratio(*(int(n) for n in str(q).split('/')))

def to_sympy_Expr(expr: OperTree) -> sympy.Expr:
    """Преобразует выражение из OperatorsTree в выражение библиотеки SymPy."""
    if (t := type(expr)) is not OperTree:
        raise TypeError(f'Type of "expr" must be "OperTree", but its type is "{t}"')
    if expr.type is T_FLOAT:
        return sympy.Float(expr.val)
    if expr.type is T_CONST:
        return to_sympy_Rational(expr.val)
    if expr.type in {T_VAR, T_LOCVAR}:
        return sympy.Symbol(expr.val)
    if expr.type is T_ADD:
        return sympy.Add(*(to_sympy_Expr(tree) for tree in expr.trees))
    if expr.type is T_MUL:
        return sympy.Mul(*(to_sympy_Expr(tree) for tree in expr.trees))
    if expr.type is T_SUB:
        return sympy.Add(*(-to_sympy_Expr(tree) for tree in expr.trees))
    if expr.type is T_DIV:
        return sympy.Mul(*(1 / to_sympy_Expr(tree) for tree in expr.trees))
    if expr.type is T_POW:
        return sympy.Pow(*(to_sympy_Expr(tree) for tree in expr.trees))
    if expr.type is F_SIN:
        return sympy.sin(to_sympy_Expr(expr.trees[0]))
    if expr.type is F_COS:
        return sympy.cos(to_sympy_Expr(expr.trees[0]))
    if expr.type is F_TAN:
        return sympy.tan(to_sympy_Expr(expr.trees[0]))
    if expr.type is F_COTAN:
        return sympy.cot(to_sympy_Expr(expr.trees[0]))
    if expr.type is F_LOG:
        return sympy.log(*(to_sympy_Expr(tree) for tree in expr.trees))
    if expr.type is F_FUNC:
        func = FUNCS.get(expr.val, None)
        if func is None:
            func = sympy.symbols(expr.val, cls = sympy.Function)
            return func(*(to_sympy_Expr(tree) for tree in expr.trees))
        func = func.copy()
        subs(func, dict(zip(func.vars, expr.trees)))
        return to_sympy_Expr(func)

def to_OperTree(expr: sympy.Expr) -> OperTree:
    """Преобразует выражение из библиотеки Sympy в выражение из OperTree."""
    if not issubclass((t := type(expr)), sympy.Expr):
        raise TypeError(f'Type of "expr" must be "sympy.Expr", but its type is "{t}"')
    if isinstance(expr, sympy.Number):
        if isinstance(expr, sympy.Float):
            return OperTree(T_FLOAT, None, float(expr))
        return OperTree(T_CONST, None, to_Rational_ratio(expr))
    if isinstance(expr, sympy.Symbol):
        return OperTree(T_VAR, None, str(expr))
    if isinstance(expr, sympy.Add):
        return OperTree(T_ADD, [to_OperTree(tree) for tree in expr.args])
    if isinstance(expr, sympy.Mul):
        return OperTree(T_MUL, [to_OperTree(tree) for tree in expr.args])
    if isinstance(expr, sympy.Pow):
        return OperTree(T_POW, [to_OperTree(tree) for tree in expr.args])
    if isinstance(expr, sympy.sin):
        return OperTree(F_SIN, [to_OperTree(expr.args[0])])
    if isinstance(expr, sympy.cos):
        return OperTree(F_COS, [to_OperTree(expr.args[0])])
    if isinstance(expr, sympy.tan):
        return OperTree(F_TAN, [to_OperTree(expr.args[0])])
    if isinstance(expr, sympy.cot):
        return OperTree(F_COTAN, [to_OperTree(expr.args[0])])
    if isinstance(expr, sympy.log):
        return OperTree(F_LOG, [to_OperTree(tree) for tree in expr.args])
    if isinstance(expr, (sympy.Function, sympy.core.function.UndefinedFunction)):
        return OperTree(F_FUNC, [to_OperTree(tree) for tree in expr.args], expr.name)

if __name__ == '__main__':
    if False:
        a = sympy.Rational(3, 7)
        b = sympy.Rational(10, 5)
        print(*(f'{q} = {to_Rational_ratio(q)}' for q in (a, b)), sep = '\n')
    if True:
        det = new_function('det', 'a*d-b*c', ['a', 'b', 'c', 'd'])
        my_expr = OperTree.expression('x*y+x^z^2-t/2.5+sin(z)*cos(z)/log(x, y)-det(a, b, c, d)^exp(e)')
        my_expr.trees[0].trees.append(OperTree(T_FLOAT, None, 1.25))
        sp_expr = to_sympy_Expr(my_expr)
        my_expr_again = to_OperTree(sp_expr)
        print(f'{my_expr}\n{sp_expr}\n{my_expr_again}')
