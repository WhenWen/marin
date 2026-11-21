"""Count static dense matrix arithmetic in the lowered StableHLO call graph."""
import math
from functools import lru_cache
from jaxlib.mlir.dialects import stablehlo

def matrix_flops(lower):
    module=lower.lowered.compiler_ir('stablehlo')
    funcs={f.attributes['sym_name'].value:f for f in module.body.operations}
    def visit(op):
        name=op.operation.name
        if name=='func.call':
            return total(op.attributes['callee'].value)
        if name=='stablehlo.dot_general':
            numbers=stablehlo.DotDimensionNumbers(op.attributes['dot_dimension_numbers'])
            inner=math.prod(op.operands[0].type.shape[i] for i in numbers.lhs_contracting_dimensions)
            return 2*inner*math.prod(op.results[0].type.shape)
        parts=sum(visit(child) for reg in op.regions for block in reg.blocks for child in block.operations)
        if parts and name in ('stablehlo.while','stablehlo.case','stablehlo.if'):
            raise ValueError(f'nonstatic control flow contains a dot: {name}')
        return parts
    @lru_cache(None)
    def total(sym):return visit(funcs[sym])
    return total('main')
