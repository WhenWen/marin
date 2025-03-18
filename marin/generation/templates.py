STEP_BY_STEP_TEMPLATE = """
You will be given a problem. Please reason step by step, and put your final answer within \boxed{}:
{example}
"""

# Added for test case maybe remove latter

MATH_PROOF_TEMPLATE = """
You will be given a mathematical proposition to prove. Work through the proof methodically, providing justification for each step, and conclude with your final proof within \boxed{}:
{example}
"""  # noqa: E501

CODE_DEBUG_TEMPLATE = """
You will be given a piece of code that contains bugs. Analyze the code step by step, identify each issue, explain why it's problematic, and provide the corrected solution within \boxed{}:
{example}
"""  # noqa: E501

LOGICAL_ANALYSIS_TEMPLATE = """
You will be given a logical argument or scenario. Analyze each component carefully, evaluate the validity of the reasoning, and provide your conclusion within \boxed{}:
{example}
"""  # noqa: E501

DECISION_MAKING_TEMPLATE = """
You will be presented with a complex decision problem. Identify all relevant factors, weigh the pros and cons of each option, and recommend the optimal decision with justification within \boxed{}:
{example}
"""  # noqa: E501

SCIENTIFIC_METHOD_TEMPLATE = """
You will be given a scientific question or hypothesis. Apply the scientific method by defining variables, formulating hypotheses, designing experiments, analyzing theoretical results, and presenting your conclusion within \boxed{}:
{example}
"""  # noqa: E501

ETHICAL_DILEMMA_TEMPLATE = """
You will be presented with an ethical dilemma. Examine the situation from multiple ethical frameworks, consider the stakeholders and consequences, and provide your most balanced ethical analysis within \boxed{}:
{example}
"""  # noqa: E501

HISTORICAL_ANALYSIS_TEMPLATE = """
You will be given a historical event or development. Analyze the causes, context, consequences, and significance step by step, and provide your comprehensive historical interpretation within \boxed{}:
{example}
"""  # noqa: E501

ALGORITHM_DESIGN_TEMPLATE = """
You will be given a computational problem. Design an algorithm to solve it by breaking down the problem, developing a solution approach, analyzing complexity, and presenting your final algorithm within \boxed{}:
{example}
"""  # noqa: E501
