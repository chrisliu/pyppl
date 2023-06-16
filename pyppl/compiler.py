import ast
import inspect
import pycfg
import pprint

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
from typing import Any, Dict, Iterator, List, Self, Tuple, Type, Union, overload

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
            if isinstance(node, ast.Call):
                ref = _get_reference(node.func, self._caller_env)
            elif isinstance(node, ast.Name) or isinstance(node, ast.Attribute):
                ref = _get_reference(node, self._caller_env)
            else:
                return False

            return ref is compile

        node.decorator_list = [decorator
                               for decorator in node.decorator_list
                               if not is_pyppl_decorator(decorator)]
        return super().generic_visit(node)


def get_cfg(src):
    cfg = pycfg.PyCFG()
    cfg.gen_cfg(src)
    cache = pycfg.CFGNode.cache
    g = {}
    for k, v in cache.items():
        j = v.to_json()
        at = j['at']
        parents_at = [cache[p].to_json()['at'] for p in j['parents']]
        children_at = [cache[c].to_json()['at'] for c in j['children']]
        if at not in g:
            g[at] = {'parents': set(), 'children': set()}
        # remove dummy nodes
        ps = set([p for p in parents_at if p != at])
        cs = set([c for c in children_at if c != at])
        g[at]['parents'] |= ps
        g[at]['children'] |= cs
        if v.calls:
            g[at]['calls'] = v.calls
        g[at]['function'] = cfg.functions_node[v.lineno()]
    return g


class BasicBlock:
    def __init__(self: Self, stmts: Union[pycfg.CFGNode, List[pycfg.CFGNode]]):
        if isinstance(stmts, pycfg.CFGNode):
            self._stmts = [stmts]
        else:
            self._stmts = stmts

        self.parents: List[int] = list()
        self.children: List[int] = list()

    def __iter__(self: Self) -> Iterator[pycfg.CFGNode]:
        return iter(self._stmts)

    @overload
    def __getitem__(self: Self, val: int) -> pycfg.CFGNode:
        ...

    @overload
    def __getitem__(self: Self, val: slice) -> List[pycfg.CFGNode]:
        ...

    def __getitem__(self: Self, val: Union[int, slice]
                    ) -> Union[pycfg.CFGNode, List[pycfg.CFGNode]]:
        return self._stmts[val]

    def __add__(self: Self, other: Self) -> Self:
        if not isinstance(other, BasicBlock):
            raise ValueError(f"Cannot add a BasicBlock with {type(other)}")
        self.children = other.children
        self._stmts += other._stmts
        return self

    def __repr__(self: Self) -> str:
        return pprint.pformat({
            'parents': self.parents,
            'children': self.children,
            'stmts': self._stmts,
        })


def bblockify(cfg: Dict[int, pycfg.CFGNode]) -> Dict[int, BasicBlock]:
    """Converts a statement cfg into a basic block cfg.

    :param cfg: A statement cfg (the return of pycfg.gen_cfg).
    :return: A basic block cfg.
    """

    bb_cfg: Dict[int, BasicBlock] = dict()
    node_remap: Dict[int, int] = dict()  # Map sCFG stmt node -> bbCFG cfg node

    def get_bb_from_stmt(stmt_id: int) -> Tuple[int, BasicBlock]:
        """Gets (and optionally allocates) the BB."""
        stmt_node = cfg[stmt_id]
        bb_id = node_remap.setdefault(stmt_id, stmt_id)
        bb = bb_cfg.setdefault(bb_id, BasicBlock(stmt_node))
        return bb_id, bb

    # Merge statements into basic blocks.
    for stmt_id, stmt_node in cfg.items():
        parent_bb_id, parent_bb = get_bb_from_stmt(stmt_id)

        # If the parent has one child and that child only has one parent,
        # then it's safe to merge.
        if len(stmt_node.children) == 1:
            child_stmt_id = stmt_node.children[0].rid
            child_stmt_node = cfg[child_stmt_id]
            if len(child_stmt_node.parents) == 1:
                child_bb_id, child_bb = get_bb_from_stmt(child_stmt_id)
                merged_bb_node = parent_bb + child_bb
                bb_cfg[parent_bb_id] = merged_bb_node
                for node in child_bb:
                    node_remap[node.rid] = parent_bb_id
                del bb_cfg[child_bb_id]

    # Update the CFG node parents and children.
    for bb in bb_cfg.values():
        # Update entry and exit points.
        entry_stmt = bb[0]
        bb.parents = [node_remap[parent_stmt.rid]
                      for parent_stmt in entry_stmt.parents]
        exit_stmt = bb[-1]
        bb.children = [node_remap[child_stmt.rid]
                       for child_stmt in exit_stmt.children]

        # Remove all parents and children for basic block statements.
        for stmt in bb:
            stmt.parents = list()
            stmt.children = list()

    return bb_cfg


def compile(*, return_types: Type[ProbVar]):
    def compile_impl(func):
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
        transformed_ast = SamplingTransform(
            caller_frame.f_locals).visit(func_ast)
        transformed_ast = ast.fix_missing_locations(transformed_ast)

        # Compile an executable version of the transformed function.
        cc = __builtin_compile(transformed_ast, filename='<ast>', mode=mode)
        # Save function in local context.
        globs = caller_frame.f_globals
        locs = caller_frame.f_locals
        exec(cc, globs, locs)
        transformed_func = locs[func.__name__]

        # Exact Inference stuff
        cfg = pycfg.gen_cfg(func_src)
        sources = [k for k, v in cfg.items() if len(v.parents) == 0]
        if len(sources) > 1:
            raise RuntimeError("PyPPL does not support nested functions")

        for k, v in cfg.items():
            print(ast.dump(v.ast_node))
            print(type(v))
            print(k, len(v.parents), len(v.children))
        from pprint import pprint
        pprint(cfg)

        cfg = bblockify(cfg)
        print("BB CFG")
        pprint(cfg)

        def contextual_execution(*args: Any, **kwargs: Any) -> ProbVar:
            # Sample logic goes here.
            inference = _cur_inference_technique()
            if inference is None:
                raise RuntimeError("Not in an active inference context.")
            elif isinstance(inference, SamplingInference):
                return inference.sample(transformed_func, return_types,
                                        *args, **kwargs)
            elif isinstance(inference, ExactInference):
                raise NotImplementedError("ExactInference is unsupported.")
            else:
                raise RuntimeError(
                    f"Unsupported inference type {type(inference)}")

        return contextual_execution
    return compile_impl
