import re
from decimal import Decimal, ROUND_HALF_UP
import sympy as sp
import signal
from sympy.stats import Normal, cdf

sympy_reserved = {'', 'exp', 'log', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sqrt', 'pi', 'Symbol', 'Integral',
                  'Derivative', 'Limit', 'Sum', 'Product', 'erf', 'N'}


def N(x):
    return cdf(Normal('X', 0, 1))(x).evalf()


def preprocess_formula(formula):
    return formula


def sympy_check(formula_list, origin_variables_set, logger):
    # Define all variables
    filtered_variables_set = origin_variables_set - sympy_reserved
    variables = sp.symbols(' '.join(filtered_variables_set))
    variables_dict = dict(zip(filtered_variables_set, variables))
    logger.info("variables_dict: {0}".format(variables_dict))

    formula_list_after_check = []
    for formula in formula_list:
        try:
            logger.info("formula: {0}".format(formula))
            formula = preprocess_formula(formula)
            logger.info("formula: {0}".format(formula))
            formula = re.sub(r'\be\^', 'exp', formula)
            formula = (formula.strip().replace('^', '**').replace("'", "")
                       .replace(",", "")).replace("√", "sqrt")
            left, right = formula.split('=')
            equation = sp.sympify(left, locals=dict(variables_dict, **{"N": N})) - sp.sympify(right, locals=dict(variables_dict, **{"N": N}))
            logger.info("Sympy for equation: {0}".format(equation))
            formula_list_after_check.append(formula)
        except Exception as e:
            logger.info("Error: {0}, remove formula: {1}".format(str(e), formula))
            continue
    return formula_list_after_check


def timeout(seconds=10, error_message="Timeout"):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


@timeout(seconds=5)
def paths_to_sympy(all_formula_paths, variables_set, input_values_dict, target_v, logger):
    # Define all variables
    filtered_variables_set = variables_set - sympy_reserved
    variables = sp.symbols(' '.join(filtered_variables_set))
    variables_dict = dict(zip(filtered_variables_set, variables))
    logger.info("variables_dict: {0}".format(variables_dict))
    try:
        # Construct equations
        equations = []
        for formula in all_formula_paths:
            left, right = formula.split('=')
            equation = sp.sympify(left, locals=dict(variables_dict, **{"N": N})) - sp.sympify(right, locals=dict(variables_dict, **{"N": N}))
            equations.append(sp.Eq(equation, 0))
        logger.info("equations: {0}".format(equations))
        known_values = {variables_dict[key]: value for key, value in input_values_dict.items() if key in variables_dict}
        logger.info("known_values: {0}".format(known_values))
        equations = [eq.subs(known_values) for eq in equations]
        logger.info("equations: {0}".format(equations))
        all_solutions = sp.solve(equations, dict=True)
        logger.info("all_solutions: {0}".format(all_solutions))
        for sol in all_solutions:
            logger.info("variables_dict[target_v]: {0}, sol: {1}".format(variables_dict[target_v], sol))
            if variables_dict[target_v] in sol:
                target_solution = sol[variables_dict[target_v]].subs(sol)

                while True:
                    new_solution = target_solution.subs(sol).simplify()
                    if new_solution == target_solution:
                        break
                    target_solution = new_solution
                return format_num(target_solution)
        else:
            return None
    except Exception as e:
        output = "Error: " + str(e)
        logger.info("Error: " + str(e))
        return output


@timeout(seconds=10)
def list_to_sympy(formula_list, variables_set, input_values_dict, logger):
    # Define all variables
    filtered_variables_set = variables_set - sympy_reserved
    variables = sp.symbols(' '.join(filtered_variables_set))
    variables_dict = dict(zip(filtered_variables_set, variables))
    logger.info("variables_dict: {0}".format(variables_dict))
    try:
        # Construct equations
        equations = []
        for formula in formula_list:
            left, right = formula.split('=')
            equation = sp.sympify(left, locals=dict(variables_dict, **{"N": N})) - sp.sympify(right, locals=dict(variables_dict, **{"N": N}))
            equations.append(sp.Eq(equation, 0))
        logger.info("equations: {0}".format(equations))
        known_values = {variables_dict[key]: value for key, value in input_values_dict.items() if key in variables_dict}
        logger.info("known_values: {0}".format(known_values))
        equations = [eq.subs(known_values) for eq in equations]
        logger.info("equations: {0}".format(equations))
        all_solutions = sp.solve(equations, dict=True)
        return all_solutions
    except Exception as e:
        output = "Error: " + str(e)
        logger.info("Error: " + str(e))
        return output


def format_num(computed_value):
    try:
        value_as_decimal = Decimal(float(computed_value))
    except Exception:
        return computed_value
    two_places = Decimal('0.0001')
    value_rounded = value_as_decimal.quantize(two_places, rounding=ROUND_HALF_UP)
    formatted_num = format(value_rounded, 'f').rstrip('0').rstrip('.')
    return formatted_num
