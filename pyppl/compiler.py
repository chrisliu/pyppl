import ast
import inspect
from sys import hash_info
import pycfg
import pprint

from collections import defaultdict
from copy import deepcopy
from pyppl.lang import observe, NotObservable
from pyppl.types import (
    ProbVar,
    ProbBool,
)
from pyppl.inference import (
    _cur_inference_technique,
    SamplingInference,
    ExactInference,
    MCMC,
)
from typing import Any, Dict, Iterator, List, Literal, Optional, Self, Set, Tuple, Type, Union, overload

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


class DominanceAnalysis:
    class DominatorTree:
        def __init__(self: Self,
                     bb_id: int,
                     children: Optional[List[Self]] = None) -> None:
            self.bb_id = bb_id
            self.children = [] if children is None else children

        def __repr__(self: Self) -> str:
            return "{cls}({id}, {children})".format(
                cls=type(self).__name__,
                id=self.bb_id,
                children=self.children)

    def __init__(self: Self,
                 cfg: Dict[int, BasicBlock],
                 root_id: Optional[int] = None,
                 key: Literal['parent', 'children'] = 'parent'
                 ) -> None:
        if key != 'parent' and key != 'children':
            raise ValueError(f"Bad key {key}")
        if root_id is None:
            root_id = list(cfg.keys())[0]

        self._cfg = cfg
        self._root_id = root_id
        self._key = key
        self._all_bb_ids = set(self._cfg.keys())
        self._rem_bb_ids = self._all_bb_ids - {self._root_id}

        self.dominators: Dict[int, Set[int]] = dict()
        self.idominators: Dict[int, Set[int]] = dict()
        self.dominance_frontier: Dict[int, Set[int]] = dict()
        self.dominator_tree = DominanceAnalysis.DominatorTree(self._root_id)

        self._construct_dominators()
        self._construct_idominators()
        self._construct_dominance_frontier()
        self._construct_dominator_tree()

    def _construct_dominators(self: Self) -> None:
        doms: Dict[int, Set[int]] = {
            bb_id: self._all_bb_ids for bb_id in self._rem_bb_ids
        }
        doms[self._root_id] = {self._root_id}

        has_changes = True
        while has_changes:
            has_changes = False
            for bb_id in self._rem_bb_ids:
                if self._key == 'parent':
                    predecessors = self._cfg[bb_id].parents
                elif self._key == 'children':
                    predecessors = self._cfg[bb_id].children
                else:
                    raise ValueError(f"Unrecognized key {self._key}")
                predecessor_doms = [doms[pred_bb_id]
                                    for pred_bb_id in predecessors]

                if len(predecessor_doms) != 0:
                    updated_doms = set.intersection(*predecessor_doms)
                else:
                    updated_doms = set()
                updated_doms |= {bb_id}  # Each node dominates itself.

                if doms[bb_id] != updated_doms:
                    doms[bb_id] = updated_doms
                    has_changes = True

        self.dominators = doms

    def _construct_idominators(self: Self) -> None:
        idoms: Dict[int, Set[int]] = {
            bb_id: dom - {bb_id}
            for bb_id, dom in self.dominators.items()
        }

        for bb_id in self._rem_bb_ids:
            dominators = list(idoms[bb_id])
            for dom_id in dominators:
                # Ignore if it's been removed already.
                if dom_id not in idoms[bb_id]:
                    continue

                for dom_dom_id in (idoms[bb_id] - {dom_id}):
                    if dom_dom_id in idoms[dom_id]:
                        idoms[bb_id] -= {dom_dom_id}

        # Sanity check correctness.
        for bb_id, idom in idoms.items():
            if len(idom) > 1:
                raise RuntimeError(
                    "{bb} cannot have more than one immediate dominator {dom}"
                    .format(bb=bb_id, dom=idom))

        self.idominators = idoms

    def _construct_dominance_frontier(self: Self) -> None:
        # construct dominance frontier.
        dominance_frontier: Dict[int, Set[int]] = defaultdict(set)
        for bb_id, bb in self._cfg.items():
            if self._key == 'parent':
                predecessors = bb.parents
            elif self._key == 'children':
                predecessors = bb.children
            else:
                raise ValueError(f"Unrecognized key {self._key}")

            if len(predecessors) > 1:
                for pred_bb_id in predecessors:
                    runner_bb_id = pred_bb_id
                    while runner_bb_id not in self.idominators[bb_id]:
                        dominance_frontier[runner_bb_id] |= {bb_id}

                        runner_idoms = self.idominators[runner_bb_id]
                        if len(runner_idoms) == 0:
                            break
                        runner_bb_id = list(runner_idoms)[0]
        dominance_frontier = {
            bb_id: dominance_frontier[bb_id] for bb_id in self._all_bb_ids}

        self.dominance_frontier = dominance_frontier

    def _construct_dominator_tree(self: Self) -> None:
        nodes: Dict[int, DominanceAnalysis.DominatorTree] = {
            self._root_id: self.dominator_tree
        }

        def get_node(bb_id: int) -> DominanceAnalysis.DominatorTree:
            return nodes.setdefault(bb_id,
                                    DominanceAnalysis.DominatorTree(bb_id))

        for bb_id, idoms in self.idominators.items():
            sub_node = get_node(bb_id)
            for idom_id in idoms:  # Should only be one
                dom_node = get_node(idom_id)
                dom_node.children.append(sub_node)

    @property
    def doms(self: Self) -> Dict[int, Set[int]]:
        return self.dominators

    @property
    def idoms(self: Self) -> Dict[int, Set[int]]:
        return self.idominators

    @property
    def df(self: Self) -> Dict[int, Set[int]]:
        return self.dominance_frontier

    @property
    def dt(self: Self) -> DominatorTree:
        return self.dominator_tree


class SSATransformer:
    def __init__(self: Self, cfg: Dict[int, BasicBlock]) -> None:
        self._da = DominanceAnalysis(cfg)

        self._cfg = cfg
        self._assign: Dict[int, Set[str]] = defaultdict(set)

        self._construct_assignments()
        self._insert_phis()
        self._rename_variables()

    def _construct_assignments(self: Self) -> None:
        """Compute the variables assigned in each bb."""

        for bb_id, bb in self._cfg.items():
            for stmt in bb:
                if stmt.ast_node is None:
                    raise ValueError(f"Bad statement has no ast {stmt}")
                ast_node = stmt.ast_node
                visit = getattr(
                    self, f'_assignment_visit_{type(ast_node).__name__}')
                visit(ast_node, bb_id)

        self._assign = {
            bb_id: self._assign[bb_id] for bb_id in self._cfg.keys()}

        # Since we're dealing with Python, a variable is also defined in a BB
        # if all dominated paths define it.
        defsites: Dict[str, Set[int]] = defaultdict(set)
        for bb_id, assignments in self._assign.items():
            for var in assignments:
                defsites[var] |= {bb_id}
        defsites = dict(defsites)
        phi_assign: Dict[int, Set[str]] = defaultdict(set)

        # Dry run phi placement algorithm.
        for var in defsites.keys():
            worklist = list(defsites[var])
            while len(worklist) != 0:
                bb_id = worklist.pop()
                for df_bb_id in self._da.df[bb_id]:
                    if var not in phi_assign[df_bb_id]:
                        bb = self._cfg[df_bb_id]
                        phi_assign[df_bb_id] |= {var}
                        if var in self._assign[df_bb_id]:
                            worklist.append(df_bb_id)  # Propogate this assign
        all_assign = {bb_id: self._assign[bb_id] | phi_assign[bb_id]
                      for bb_id in self._cfg.keys()}

    def _assignment_visit_AnnAssign(self: Self,
                                    annassign: ast.AnnAssign,
                                    bb_id: int
                                    ) -> None:
        target = annassign.target
        if not isinstance(target, ast.Name):
            raise ValueError("Unrecognized target {target}".format(
                target=type(target).__name__))
        if target.id == 'enter':
            # Enter a function, log the parameters as assigned.
            annotation = annassign.annotation
            if not isinstance(annotation, ast.Call):
                raise ValueError(
                    "Unrecognized annotation for enter {ty}".format(
                        ty=type(annotation).__name__))

            for arg in annotation.args:
                if not isinstance(arg, ast.Name):
                    raise ValueError("Unrecognized argument type {ty}".format(
                        ty=type(arg).__name__))
                self._assign[bb_id] |= {arg.id}

            if len(annotation.keywords) > 0:
                raise NotImplementedError("No support for keywords")
        elif target.id == '_if':
            pass
        elif target.id == 'exit':
            pass
        else:
            raise ValueError("Unrecognized target name {name}".format(
                name=target.id))

    def _assignment_visit_Assign(self: Self, assign: ast.Assign, bb_id: int
                                 ) -> None:
        for target in assign.targets:
            if not isinstance(target, (ast.Name, ast.Tuple, ast.List)):
                raise ValueError("Unrecognized assignment to {ty}".format(
                    ty=type(target).__name__))

            if isinstance(target, (ast.Tuple, ast.List)):
                raise NotImplementedError("Unsupported unpacking assignment")
            elif isinstance(target, ast.Name):
                self._assign[bb_id] |= {target.id}
            else:
                raise NotImplementedError(
                    "Unsupported assginment to {ty}".format(
                        ty=type(target).__name__))

    def _assignment_visit_Expr(self: Self, expr: ast.Expr, bb_id: int
                               ) -> None:
        pass

    def _assignment_visit_Return(self: Self, ret: ast.Return, bb_id: int
                                 ) -> None:
        pass

    def _insert_phis(self: Self) -> None:
        # BBs that define each variable.
        defsites: Dict[str, Set[int]] = defaultdict(set)
        for bb_id, assignments in self._assign.items():
            for var in assignments:
                defsites[var] |= {bb_id}
        defsites = dict(defsites)
        all_assign = deepcopy(self._assign)

        phi_assign: Dict[int, Set[str]] = defaultdict(set)

        for var in defsites.keys():
            worklist = list(defsites[var])
            while len(worklist) != 0:
                bb_id = worklist.pop()
                for df_bb_id in self._da.df[bb_id]:
                    if var not in phi_assign[df_bb_id]:
                        bb = self._cfg[df_bb_id]
                        phi_stmt = ast.AnnAssign(
                            target=ast.Name(id=var, ctx=ast.Store()),
                            annotation=ast.Name(id='phi', ctx=ast.Load()),
                            value=ast.Tuple(
                                elts=[ast.Name(id=var, ctx=ast.Load())
                                      for _ in range(len(bb.parents))],
                                ctx=ast.Load()),
                            simple=True)
                        bb._stmts.insert(0, pycfg.CFGNode(ast=phi_stmt))

                        phi_assign[df_bb_id] |= {var}
                        if var in self._assign[df_bb_id]:
                            worklist.append(df_bb_id)  # Propogate this assign

                        # Since we're dealing with Python, a variable is also
                        # defined in a BB if all dominated paths define it.
                        def update_assigns(
                                node: DominanceAnalysis.DominatorTree) -> None:
                            bb_id = node.bb_id
                            if len(node.children) == 0:
                                all_assign[bb_id] |= phi_assign[bb_id]
                                return

                            for sub in node.children:
                                update_assigns(sub)

                            child_assigns = set.intersection(
                                *[all_assign[sub.bb_id]
                                  for sub in node.children])
                            new_assigns = child_assigns - self._assign[bb_id]
                            if len(new_assigns) != 0:
                                self._assign[bb_id] |= new_assigns
                                worklist.append(bb_id)
                        update_assigns(self._da.dt)

        pprint.pprint(self._cfg)

    def _rename_variables(self: Self) -> None:
        count: Dict[str, int] = defaultdict(int)
        stack: Dict[str, List[int]] = defaultdict(list)

        def rename_bb(dt_node: DominanceAnalysis.DominatorTree) -> None:
            bb = self._cfg[dt_node.bb_id]

            bad_phis: Set[pycfg.CFGNode] = set()
            assignments: List[str] = list()
            for stmt in bb:
                ast_node = stmt.ast_node
                if ast_node is None:
                    raise ValueError(f"Bad statement has no ast {stmt}")
                visit = getattr(
                    self, f'_rename_visit_{type(ast_node).__name__}')
                phi, assigns = visit(ast_node, count, stack)
                if phi is not None:
                    bad_phis |= {phi}
                assignments += assigns

            print("BB", dt_node.bb_id)
            print(bb)

            for dt_child in dt_node.children:
                rename_bb(dt_child)
            for var in assignments:
                stack[var].pop()

        rename_bb(self._da.dt)

    def _rename_visit_AnnAssign(self: Self,
                                annassign: ast.AnnAssign,
                                count: Dict[str, int],
                                stack: Dict[str, List[int]]
                                ) -> Tuple[Optional[pycfg.PyCFG], List[str]]:
        target = annassign.target
        if not isinstance(target, ast.Name):
            raise ValueError("Unrecognized target {target}".format(
                target=type(target).__name__))

        annotation = annassign.annotation
        assignments = list()
        if target.id == 'enter':
            # Enter a function, log the parameters as assigned.
            if not isinstance(annotation, ast.Call):
                raise ValueError(
                    "Unrecognized annotation for enter {ty}".format(
                        ty=type(annotation).__name__))

            for arg in annotation.args:
                if isinstance(arg, ast.Name):
                    var = arg.id
                    var_id = count[var]
                    count[var] += 1
                    stack[var].append(var_id)
                    assignments.append(var)
                    arg.id = self._rename(var, var_id)
                else:
                    raise ValueError("Unrecognized argument type {ty}".format(
                        ty=type(arg).__name__))

            if len(annotation.keywords) > 0:
                raise NotImplementedError("No support for keywords")
        elif target.id == '_if':
            if isinstance(annotation, ast.BoolOp):
                self._rename_visit_BoolOp(annotation, count, stack)
            elif isinstance(annotation, ast.Call):
                self._rename_visit_Call(annotation, count, stack)
            elif isinstance(annotation, ast.Name):
                self._rename_visit_Name(annotation, count, stack)
            else:
                raise ValueError("Unrecognized if condition {ty}".format(
                    ty=type(annotation).__name__))
        elif target.id == 'exit':
            pass
        else:
            if not isinstance(annotation, ast.Name) or annotation.id != 'phi':
                raise ValueError("Unrecognized target name {name}".format(
                    name=target.id))
            # Assign new.
            var = target.id
            var_id = count[var]
            count[var] += 1
            stack[var].append(var_id)
            assignments.append(var)
            target.id = self._rename(var, var_id)

        return None, assignments

    def _rename_visit_Assign(self: Self,
                             assign: ast.Assign,
                             count: Dict[str, int],
                             stack: Dict[str, List[int]]
                             ) -> Tuple[Optional[pycfg.PyCFG], List[str]]:
        assignments = list()
        for target in assign.targets:
            if not isinstance(target, (ast.Name, ast.Tuple, ast.List)):
                raise ValueError("Unrecognized assignment to {ty}".format(
                    ty=type(target).__name__))

            if isinstance(target, (ast.Tuple, ast.List)):
                raise NotImplementedError("Unsupported unpacking assignment")
            elif isinstance(target, ast.Name):
                # Rename uses.
                value = assign.value
                if isinstance(value, ast.Name):
                    self._rename_visit_Name(value, count, stack)
                elif isinstance(value, ast.Call):
                    self._rename_visit_Call(value, count, stack)
                elif isinstance(value, ast.BoolOp):
                    self._rename_visit_BoolOp(value, count, stack)

                # Assign new.
                var = target.id
                var_id = count[var]
                count[var] += 1
                stack[var].append(var_id)
                assignments.append(var)
                target.id = self._rename(var, var_id)
            else:
                raise NotImplementedError(
                    "Unsupported assginment to {ty}".format(
                        ty=type(target).__name__))

        return None, assignments

    def _rename_visit_Expr(self: Self,
                           expr: ast.Assign,
                           count: Dict[str, int],
                           stack: Dict[str, List[int]]
                           ) -> Tuple[Optional[pycfg.PyCFG], List[str]]:
        value = expr.value
        if isinstance(value, ast.Call):
            return self._rename_visit_Call(value, count, stack)
        else:
            raise ValueError("Unrecognized expr type {ty}".format(
                ty=type(value).__name__))

        return None, list()

    def _rename_visit_Return(self: Self,
                             assign: ast.Assign,
                             count: Dict[str, int],
                             stack: Dict[str, List[int]]
                             ) -> Tuple[Optional[pycfg.PyCFG], List[str]]:
        return None, list()

    def _rename_visit_Call(self: Self,
                           call: ast.Call,
                           count: Dict[str, int],
                           stack: Dict[str, List[int]]
                           ) -> Tuple[Optional[pycfg.PyCFG], List[str]]:
        for arg in call.args:
            if isinstance(arg, ast.Call):
                self._rename_visit_Call(arg, count, stack)
            elif isinstance(arg, ast.Name):
                self._rename_visit_Name(arg, count, stack)
            else:
                raise ValueError("Unsupported arg type {ty}".format(
                    ty=type(arg).__name__))

        return None, list()

    def _rename_visit_BoolOp(self: Self,
                             boolop: ast.BoolOp,
                             count: Dict[str, int],
                             stack: Dict[str, List[int]]
                             ) -> Tuple[Optional[pycfg.PyCFG], List[str]]:
        for value in boolop.values:
            if isinstance(value, ast.Name):
                self._rename_visit_Name(value, count, stack)
            elif isinstance(value, ast.Call):
                self._rename_visit_Call(value, count, stack)
            else:
                raise ValueError("Unsupported value type {ty}".format(
                    ty=type(value).__name__))
        return None, list()

    def _rename_visit_Name(self: Self,
                           name: ast.Name,
                           count: Dict[str, int],
                           stack: Dict[str, List[int]]
                           ) -> Tuple[Optional[pycfg.PyCFG], List[str]]:
        var = name.id
        if len(stack[var]) == 0:
            raise UnboundLocalError(f"{var} not defined")
        var_id = stack[var][-1]
        name.id = self._rename(var, var_id)
        return None, list()

    def _rename(self: Self, var: str, id: int) -> str:
        return f'*{id}*{var}*'


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

        cfg = bblockify(cfg)
        SSATransformer(cfg)

        def contextual_execution(*args: Any, **kwargs: Any) -> ProbVar:
            # Sample logic goes here.
            inference = _cur_inference_technique()
            if inference is None:
                raise RuntimeError("Not in an active inference context.")
            elif isinstance(inference, SamplingInference):
                return inference.sample(transformed_func, return_types,
                                        *args, **kwargs)
            elif isinstance(inference, MCMC):
                return inference.sample(transformed_func, return_types,
                                        *args, **kwargs)
            elif isinstance(inference, ExactInference):
                raise NotImplementedError("ExactInference is unsupported.")
            else:
                raise RuntimeError(
                    f"Unsupported inference type {type(inference)}")

        return contextual_execution
    return compile_impl
