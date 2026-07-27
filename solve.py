# import os
from sympy import symbols, Eq, solve, sympify

# os.system("cls")

variable = ["a","y","z","A","Y","Z"]
def solve_eq(equation):
    
    count = 0
    for i in variable:
        if i in equation:
            count += 1
            var = i

    if count == 0:

        if "=" in equation and equation[-1] == "=":
            try:
                return eval(equation[:-1])
            except:
                return "Invalid equation or no equation detected"
        
        elif "=" not in equation:
            try:
                return eval(equation)
            except:
                return "Invalid equation or no equation detected"
        
        else:
            return "Invalid equation or no equation detected"
    
    elif count > 1:
        return [equation,"Enter equation of single variable"]

        
    elif count == 1:
        try:
            x = symbols(var)
            equation = equation.split("=")
            print(equation)
            equation = Eq(sympify(equation[0]),sympify(equation[1]))
            return solve(equation, x)
        
        except:
            return "Invalid equation or no equation detected"

# print(solve_eq("2+3*y=3"))
