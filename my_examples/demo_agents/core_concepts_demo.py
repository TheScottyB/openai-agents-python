from typing import Dict, Tuple
from pydantic import BaseModel
import asyncio
import sys
from functools import wraps

# TOOL IMPLEMENTATION
class UnitConverter:
    """Tool for converting between different units of measurement."""
    
    @staticmethod
    def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
        """Convert temperature between Celsius, Fahrenheit and Kelvin."""
        conversions = {
            ('c', 'f'): lambda x: x * 9/5 + 32,
            ('f', 'c'): lambda x: (x - 32) * 5/9,
            ('c', 'k'): lambda x: x + 273.15,
            ('k', 'c'): lambda x: x - 273.15,
            ('f', 'k'): lambda x: (x - 32) * 5/9 + 273.15,
            ('k', 'f'): lambda x: (x - 273.15) * 9/5 + 32
        }
        key = (from_unit.lower()[0], to_unit.lower()[0])
        if key not in conversions:
            raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
        return conversions[key](value)

    @staticmethod
    def convert_length(value: float, from_unit: str, to_unit: str) -> float:
        """Convert length between meters, feet, inches, and miles."""
        units = {
            'm': 1.0,
            'ft': 0.3048,
            'in': 0.0254,
            'mi': 1609.34
        }
        try:
            return value * units[from_unit] / units[to_unit]
        except KeyError:
            raise ValueError(f"Invalid units: {from_unit} to {to_unit}")

    @staticmethod
    def convert_weight(value: float, from_unit: str, to_unit: str) -> float:
        """Convert weight between kilograms, pounds, and ounces."""
        units = {
            'kg': 1.0,
            'lb': 0.453592,
            'oz': 0.0283495
        }
        if from_unit not in units or to_unit not in units:
            raise ValueError(f"Invalid weight units: {from_unit} to {to_unit}")
        return value * units[from_unit] / units[to_unit]

class MathConstant:
    """Tool providing mathematical constants and their explanations."""
    
    CONSTANTS: Dict[str, Tuple[float, str]] = {
        'pi': (3.14159265359, "The ratio of a circle's circumference to its diameter"),
        'e': (2.71828182846, "The base of the natural logarithm"),
        'golden_ratio': (1.61803398875, 
                        "The ratio where the ratio of the sum of quantities to the larger quantity "
                        "is equal to the ratio of the larger quantity to the smaller one"),
        'avogadro': (6.02214076e23, 
                    "The number of constituent particles in one mole of a substance"),
        'plank': (6.62607015e-34, 
                 "The fundamental quantum of action in quantum mechanics")
    }
    
    @classmethod
    def get_constant(cls, name: str) -> Tuple[float, str]:
        """Retrieve a mathematical constant and its explanation."""
        if name not in cls.CONSTANTS:
            raise ValueError(f"Unknown constant: {name}")
        return cls.CONSTANTS[name]

class CalculatorTool:
    """Main calculator tool incorporating unit conversion and constants."""
    
    def __init__(self):
        self.unit_converter = UnitConverter()
        self.math_constants = MathConstant()

class CalculatorAgent:
    """Agent that uses the calculator tools."""
    
    def __init__(self):
        self.tool = CalculatorTool()

def handle_errors(func):
    """Error handling decorator that works with both sync and async functions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return None

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            return None

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return wrapper

@handle_errors
async def demonstrate_unit_conversions(agent):
    print("\n=== Unit Conversion Examples ===\n")
    
    # Temperature conversions
    print("Temperature:")
    conv = agent.tool.unit_converter.convert_temperature(0, 'c', 'f')
    print(f"0°C is {conv:.1f}°F")
    
    conv = agent.tool.unit_converter.convert_temperature(212, 'f', 'k')
    print(f"212°F is {conv:.2f}K")
    
    # Length conversions
    print("\nLength:")
    conv = agent.tool.unit_converter.convert_length(1, 'mi', 'm')
    print(f"1 mile is {conv:.2f} meters")
    
    conv = agent.tool.unit_converter.convert_length(12, 'in', 'ft')
    print(f"12 inches is {conv:.2f} feet")
    
    # Weight conversions
    print("\nWeight:")
    conv = agent.tool.unit_converter.convert_weight(16, 'oz', 'lb')
    print(f"16 ounces is {conv:.2f} pounds")
    
    conv = agent.tool.unit_converter.convert_weight(1, 'kg', 'oz')
    print(f"1 kilogram is {conv:.2f} ounces")

@handle_errors
async def demonstrate_math_constants(agent):
    print("\n=== Mathematical Constants ===\n")
    
    constants = ['pi', 'e', 'golden_ratio', 'avogadro', 'plank']
    for const in constants:
        value, desc = agent.tool.math_constants.get_constant(const)
        print(f"{const.capitalize()}: {value}\n    {desc}\n")


async def async_main():
    agent = CalculatorAgent()
    
    # Demonstrate unit conversions with error handling
    await demonstrate_unit_conversions(agent)
    
    # Demonstrate mathematical constants with error handling
    await demonstrate_math_constants(agent)
    
    # Test error handling
    print("\n=== Error Handling Test ===")
    print("Testing invalid temperature conversion:")
    try:
        show_error = agent.tool.unit_converter.convert_temperature(100, 'c', 'xyz')
    except ValueError as e:
        print(f"{e}")
    
    print("\nTesting invalid length conversion:")
    try:
        show_error = agent.tool.unit_converter.convert_length(10, 'km', 'mi')
    except ValueError as e:
        print(f"{e}")
    
    print("\nTesting invalid math constant:")
    try:
        show_error = agent.tool.math_constants.get_constant('unknown')
    except ValueError as e:
        print(f"{e}")

if __name__ == "__main__":
    asyncio.run(async_main())


