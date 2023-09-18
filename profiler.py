import cProfile
import importlib
import pstats
import sys
from typing import Any

def profile_function(module: Any, function_name: str, output_file: str) -> None:
    """
    Profile a specific function within a Python module using cProfile and save the results to a file.

    Args:
        module (Any): The module to profile.
        function_name (str): The name of the function to profile.
        output_file (str): The name of the file to save profiling results.

    Returns:
        None: This function doesn't return anything.
    """
    if hasattr(module, function_name):
        function_to_profile = getattr(module, function_name)
    else:
        print(f"Error: {module.__name__} does not have a function named {function_name}")
        sys.exit(1)

    profiler = cProfile.Profile()
    profiler.enable()

    result = function_to_profile()

    profiler.disable()

    with open(output_file, 'w') as file:
        stats = pstats.Stats(profiler, stream=file)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats()

if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("ERROR: Bad arguments. Usage: python profiler.py module_name function_name [output_file.txt]")
        sys.exit(1)

    module_name = sys.argv[1]
    function_name = sys.argv[2]
    if len(sys.argv) == 4:
        output_file = sys.argv[3]
    else:
        output_file = 'profile_results.txt'

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        print(f"Error: Unable to import module {module_name}")
        sys.exit(1)

    profile_function(module, function_name, output_file)
