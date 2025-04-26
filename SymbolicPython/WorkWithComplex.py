from Algebra import *

#Мнимая еденица:
ImaginaryUnit = OperTree.expression('i')
#Наша алгебра:
general = Algebra()

#Создаём своё правило сортировки дерева для удобства работы в нашей алгебре:
def cmp_rule_for_complex_algebra(x: OperTree, y: OperTree) -> int:
    """Задаёт правило сортировки деревьев для комплексной алгебры."""
    x_i = x == ImaginaryUnit
    y_i = y == ImaginaryUnit
    if x_i and y_i:
        return 0
    elif x_i:
        return -1
    elif y_i:
        return 1
    else:
        return cmp(x, y, cmp_rule_for_complex_algebra)

def i_selector(expr: OperTree):
    """Выделяет мнимую еденицу."""
    if expr is None or expr.trees is None:
        return
    if expr.type == T_MUL and len(expr.trees) > 2:
        params = [f'a{i}' for i in range(1, len(expr.trees))]
        par_mul = '*'.join(params)
        f = (general.expression(f'i*{par_mul}', params),
             general.expression(f'i*({par_mul})', params))
        replace(expr, *f)
    for tree in expr.trees:
        i_selector(tree)

neg_to_mul_of_neg_unit = general.expression('-a', ['a']), general.expression('(-1)*a', ['a'])

expr = general.expression('8*a*i+b*c-c*i-9*i-7*p+(i*9)*(i*p)')
print(expr)

calc_consts(expr)
print(expr, expr.str_tree(), sep = '\n')

replace(expr, *neg_to_mul_of_neg_unit)
print(expr)

sort(expr, cmp_rule_for_complex_algebra)
print(expr)

i_selector(expr)
print(expr)
