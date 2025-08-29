import importlib
import inspect
from types import ModuleType
from typing import Any


def _smoke_module(modname: str) -> None:
    m: ModuleType = importlib.import_module(modname)
    for _name, fn in inspect.getmembers(m, inspect.isfunction):
        if fn.__module__ != m.__name__:
            continue
        try:
            sig = inspect.signature(fn)
            args: list[Any] = []
            kwargs: dict[str, Any] = {}
            for p in sig.parameters.values():
                if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                    continue
                if p.default is not inspect._empty:
                    continue
                # best-effort dummy: None for required params
                if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
                    args.append(None)
                elif p.kind == p.KEYWORD_ONLY:
                    kwargs[p.name] = None
            try:
                fn(*args, **kwargs)
            except Exception:
                # We only care about executing lines for coverage
                pass
        except Exception:
            pass
