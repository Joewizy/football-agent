from langchain_core.tools import tool


@tool
def calculator(a: float, b: float) -> float:
    """
    Add two numbers together and return the result.
    """
    return a + b
    