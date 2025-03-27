from calculator_operations import MathOperations

class Calculator:
    def __init__(self):
        self.math_operations = MathOperations()
        
    def perform_operations(self, a, b):
        print(f"{a}와 {b}의 덧셈결과 : {self.math_operations.add(a,b)}")
        print(f"{a}와 {b}의 뺄셈결과 : {self.math_operations.subtract(a,b)}")
        print(f"{a}와 {b}의 곱셈결과 : {self.math_operations.multiply(a,b)}")
        print(f"{a}와 {b}의 나눗셈결과 : {self.math_operations.divide(a,b)}")
        
        
if __name__ == "__main__":
    calc = Calculator()
    calc.perform_operations(10,20)
    calc.perform_operations(10,0)