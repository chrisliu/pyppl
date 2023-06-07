import ast
import inspect

from pyppl.lang import observe, NotObservable
from pyppl.types import (
    ProbVar,
    ProbBool,
)
from pyppl.inference import (
    _cur_inference_technique,
    SamplingInference,
    ExactInference,
)
from typing import Any, Dict, Self, Union

# Reference to built in `compile` function that'll be overwritten.
__builtin_compile = compile


def _get_reference(node: ast.expr, env: Dict[Any, Any]) -> Any:
    # Warning: this only searches globals (should search in the appropriate
    # call frame instead).
    if isinstance(node, ast.Name):
        return env[node.id]
    elif isinstance(node, ast.Attribute):
        def recursive_attribute_unpack(node: ast.expr) -> Any:
            if isinstance(node, ast.Name):
                return env[node.id]
            elif isinstance(node, ast.Attribute):
                ctx = recursive_attribute_unpack(node.value)
                return getattr(ctx, node.attr)
            else:
                raise ValueError(f"Unrecognized attribute value {type(node)}")

        return recursive_attribute_unpack(node)
    else:
        raise ValueError(f"Unrecognized reference type {type(node)}")


class SamplingTransform(ast.NodeTransformer):
    def __init__(self: Self, caller_env: Dict[Any, Any]) -> None:
        self._caller_env = caller_env

    def visit_Expr(self: Self, stmt_node: ast.Expr) -> Union[ast.Expr, ast.If]:
        expr_node: ast.expr = stmt_node.value
        if isinstance(expr_node, ast.Call):
            callee = _get_reference(expr_node.func, self._caller_env)
            if callee is observe:
                # Replace `observe(...)` with early exit logic.
                # ```
                # if not observe(...)
                #   return NotObservable
                # ```
                # Warning: pyppl_observe hard codes `pyppl.observe`, which
                #          assumes the user didn't rename the pyppl package
                #          with `import pyppl as ...`
                not_cond = ast.UnaryOp(ast.Not(), expr_node)
                pyppl_observe = ast.Attribute(
                    value=ast.Name('pyppl', ctx=ast.Load()),
                    attr=NotObservable.__name__,
                    ctx=ast.Load())
                if_observe_node = ast.If(not_cond,
                                         [ast.Return(pyppl_observe)],
                                         [])
                return ast.copy_location(if_observe_node, expr_node)
        return stmt_node

    def visit_FunctionDef(self: Self, node: ast.FunctionDef) -> ast.AST:
        def is_pyppl_decorator(node: ast.expr) -> bool:
            if not (isinstance(node, ast.Name) or
                    isinstance(node, ast.Attribute)):
                return False
            else:
                ref = _get_reference(node, self._caller_env)
                return ref is compile

        node.decorator_list = [decorator
                               for decorator in node.decorator_list
                               if not is_pyppl_decorator(decorator)]
        return super().generic_visit(node)


def compile(func):
    func_src = inspect.getsource(func)
    src_lines = [line for line in func_src.split('\n') if len(line) != 0]
    leading = [len(line) - len(line.lstrip(' ')) for line in src_lines]
    indent = min(leading)
    src_lines = [line[indent:] for line in src_lines]
    func_src = '\n'.join(src_lines)

    cur_frame = inspect.currentframe()
    assert cur_frame is not None
    caller_frame = cur_frame.f_back
    assert caller_frame is not None

    # Insert correct logic for observe statements.
    mode = 'exec'
    func_ast = ast.parse(func_src, mode=mode)
    transformed_ast = SamplingTransform(caller_frame.f_locals).visit(func_ast)
    transformed_ast = ast.fix_missing_locations(transformed_ast)

    # Compile an executable version of the transformed function.
    cc = __builtin_compile(transformed_ast, filename='<ast>', mode=mode)
    # Save function in local context.
    globs = caller_frame.f_globals
    locs = caller_frame.f_locals
    exec(cc, globs, locs)
    transformed_func = locs[func.__name__]

    # Exact Inference stuff

    # cfg = pycfg.gen_cfg(func_src)
    # sources = [k for k, v in cfg.items() if len(v.parents) == 0]
    # if len(sources) > 1:
    #     raise RuntimeError("PyPPL does not support nested functions")

    # for k, v in cfg.items():
    #     print(ast.dump(v.ast_node))
    #     print(k, len(v.parents), len(v.children))
    # print(cfg)

    def contextual_execution(*args: Any, **kwargs: Any) -> ProbVar:
        # Sample logic goes here.
        inference = _cur_inference_technique()
        if inference is None:
            raise RuntimeError("Not in an active inference context.")
        elif isinstance(inference, SamplingInference):
            return inference.sample(transformed_func, ProbBool,
                                    *args, **kwargs)
        elif isinstance(inference, ExactInference):
            raise NotImplementedError("ExactInference is unsupported.")
        else:
            raise RuntimeError(f"Unsupported inference type {type(inference)}")

    return contextual_execution
