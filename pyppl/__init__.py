import abc
import ast
import contextvars
import functools
import inspect
import sys
from types import FrameType
from typing import Any, Dict, List, Self, Optional, Set


class Compiler:
    '''A JIT compiler that constructs the PyPPL AST to the appropriate
    inference context.'''

    skip_frame: Optional[FrameType] = None
    skip_lineno: Optional[int] = None

    @staticmethod
    def trace(frame: FrameType, event: str, arg: Optional[Any]):
        if Compiler.skip_frame is not None:
            if Compiler.skip_frame == frame:
                print(Compiler.skip_frame, frame)
                assert Compiler.skip_lineno is not None
                frame.f_lineno = Compiler.skip_lineno
                Compiler.skip_frame = None
                Compiler.skip_lineno = None
        frame.f_trace = Compiler.trace
        return Compiler.trace

    @staticmethod
    def skip(frame: FrameType, lineno: int):
        assert Compiler.skip_frame is None and Compiler.skip_lineno is None
        Compiler.skip_frame = frame
        Compiler.skip_lineno = lineno
        frame.f_trace = Compiler.trace

    @staticmethod
    @functools.cache
    def compile(src_ast: ast.AST, frame: FrameType):
        if isinstance(src_ast, ast.If):
            Compiler.compile_if(src_ast, frame)
        else:
            raise
        Compiler.skip(frame, src_ast.end_lineno + 1)

    @staticmethod
    def compile_if(if_ast: ast.AST, frame: FrameType):
        ...

    @staticmethod
    @functools.cache
    def parse(source: str) -> ast.AST:
        return ast.parse(source)

    @staticmethod
    def compile_caller(rel_idx: int = 1):
        '''Compile the caller.

        rel_idx is the caller frame relative this function's caller.
        (0 is this function's caller's frame)
        '''

        call_stack = inspect.stack()
        frame_idx = rel_idx + 1
        assert len(call_stack) >= frame_idx
        caller_frame = call_stack[frame_idx].frame
        caller_source = inspect.getsource(caller_frame)

        # Get AST node at caller line.
        source_ast = Compiler.parse(caller_source)
        caller_traceback = inspect.getframeinfo(caller_frame)
        caller_ast: Optional[ast.AST] = None
        for node in ast.walk(source_ast):
            if isinstance(node, ast.expr) or isinstance(node, ast.stmt):
                if node.lineno == caller_traceback.lineno:
                    caller_ast = node
                    break

        if caller_ast is not None:
            Compiler.compile(caller_ast, caller_frame)
        else:
            raise RuntimeError("Could not locate the caller's AST")


class Model:
    def __enter__(self):
        self._sys_trace = sys.gettrace()
        sys.settrace(Compiler.trace)

    def __exit__(self, exc_type, exc_value, exc_tb):
        sys.settrace(None)


class ProbabilisticVariable:
    def __bool__(self) -> bool:
        '''Handle when it's the condition of an if-statement.'''
        Compiler.compile_caller()
        return True


class Flip(ProbabilisticVariable):
    ...


class Scope:
    ...


class AST(abc.ABC):
    @abc.abstractclassmethod
    def eval(self, scope: Scope):
        ...


class Atom(AST):
    def eval(self, scope: Scope):
        pass


class Variable(Atom):
    def __init__(self, name: str):
        self._name: str = name

    @property
    def name(self) -> str:
        return self._name


class Expr(AST):
    ...


class ExecutableExpr(Expr):
    ...


class AssignExpr(Expr):
    ...


class Body(AST):
    def __init__(self, exprs: Optional[List[Expr]]):
        self._exprs: List[Expr] = [] if exprs is None else exprs

    @property
    def exprs(self):
        return self._exprs

    def __iter__(self):
        return iter(self._exprs)


class If(AST):
    ...


class InferenceContext:
    '''Represents the current inference technique.'''
    __global: contextvars.ContextVar[Optional[Self]] =\
        contextvars.ContextVar('global_inference_context', default=None)

    def __init__(self):
        self._free_variables: Set[ProbabilisticVariable] = set()

    def register_free_variable(self, var: ProbabilisticVariable):
        self._free_variables.add(var)

    @property
    def free_variables(self) -> Set[ProbabilisticVariable]:
        return self._free_variables

    def __enter__(self):
        self._token = InferenceContext.__global.set(self)

    def __exit__(self, exc_type, exc_value, exc_tb):
        InferenceContext.__global.reset(self._token)

    @ staticmethod
    def _cur_context() -> 'InferenceContext':
        if InferenceContext.__global.get() is None:
            InferenceContext.__global.set(InferenceContext())
        return InferenceContext.__global.get()


def report(v: ProbabilisticVariable):
    print(v)


if __name__ == '__main__':
    with Model() as model:
        cond = Flip()
        if cond and cond:
            print("SOMETHING HAPPENED")
            res = 0
        else:
            print("SOMETHING HAPPENED")
            res = 1
        ...

    # if Flip():
    #     ...

    # v = some_random_var_func()
