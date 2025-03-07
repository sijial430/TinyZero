import re
import random
import ast
import operator
import math


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r'\d+', equation_str)]
        
        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        
        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str, power_operation=False, floor_operation=False):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        if not power_operation and not floor_operation:
            allowed_pattern = r'^[\d+\-*/().\s]+$'
        elif power_operation and not floor_operation:
            allowed_pattern = r'^[\d+\-*/().\s\*math.sqrt]+$'
        elif power_operation and floor_operation:
            allowed_pattern = r'^[\d+\-*/().\s\*math.sqrt\/\/\%\s]+$'
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": __builtins__, "math": math}, {})
        return result
    except Exception as e:
        print(f"Error evaluating equation: {e}")
        return None


def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1., power_operation=False, floor_operation=False):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth['target']
    numbers = ground_truth['numbers']
    
    equation = extract_solution(solution_str=solution_str)
    # do_print = random.randint(1, 64) == 1
    do_print = True
    
    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0
    
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score
        
    # Evaluate equation
    try:
        result = evaluate_equation(equation, power_operation, floor_operation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score
            
        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score 


# if __name__ == "__main__":
#     solution_str = "Assistant: Let me solve this step by step.\n<think> (1 + 2) / 3 </think>\n<answer> (1 + 2) / 3 </answer>"
#     ground_truth = {"target": 1, "numbers": [1, 2, 3]}
#     print(compute_score(solution_str, ground_truth, power_operation=False, floor_operation=False)) 
#     #1
    
#     solution_str = "Assistant: Let me solve this step by step.\n<think> (1 + 2) /+ 3 </think>\n<answer> (1 + 2) /+ 3 </answer>"
#     ground_truth = {"target": 1, "numbers": [1, 2, 3]}
#     print(compute_score(solution_str, ground_truth, power_operation=True, floor_operation=True)) 
#     #0: edge case

#     solution_str = "Assistant: Let me solve this step by step.\n<think> (1 + 2) ** 3 </think>\n<answer> (1 + 2) ** 3 </answer>"
#     ground_truth = {"target": 1, "numbers": [1, 2, 3]}
#     print(compute_score(solution_str, ground_truth, power_operation=True, floor_operation=False)) 
#     #0.1
    
#     solution_str = "Assistant: Let me solve this step by step.\n<think> math.sqrt(1 + 3) ** 2 </think>\n<answer> math.sqrt(1 + 3) ** 2 </answer>"
#     ground_truth = {"target": 4, "numbers": [1, 2, 3]}
#     print(compute_score(solution_str, ground_truth, power_operation=True, floor_operation=False)) 
#     #1
    
#     solution_str = "Assistant: Let me solve this step by step.\n<think> math.sqrt(1 + 2) ** 3 </think>\n<answer> math.sqrt(1 + 2) ** 3 </answer>"
#     ground_truth = {"target": 4, "numbers": [1, 2, 3]}
#     print(compute_score(solution_str, ground_truth, power_operation=True, floor_operation=False)) 
#     #0.1

#     solution_str = "Assistant: Let me solve this step by step.\n<think> 1 * 3 ** 2 </think>\n<answer> 1 * 3 ** 2 </answer>"
#     ground_truth = {"target": 9, "numbers": [1, 2, 3]}
#     print(compute_score(solution_str, ground_truth, power_operation=True, floor_operation=False)) 
#     #1

#     solution_str = "Assistant: Let me solve this step by step.\n<think> (1 + 2) // 3 </think>\n<answer> (1 + 2) // 3 </answer>"
#     ground_truth = {"target": 1, "numbers": [1, 2, 3]}
#     print(compute_score(solution_str, ground_truth, power_operation=True, floor_operation=True)) 
#     #1
    
#     solution_str = "Assistant: Let me solve this step by step.\n<think> 1 * 2 // 3 </think>\n<answer> 1 * 2 // 3 </answer>"
#     ground_truth = {"target": 1, "numbers": [1, 2, 3]}
#     print(compute_score(solution_str, ground_truth, power_operation=True, floor_operation=True)) 
#     #0.1
