"""Extract ground-truth support locations from a SWE-bench-style patch.

A *support location* is a repository-defined function / method / class that is
invoked by an edited statement in the patch.  The augmentation rule comes from
the paper section "Issue Localization Benchmark Augmentation":

    - Bare-name calls            -> resolved function / class
    - Class instantiations       -> Class.__init__
    - self.method / super().method -> method on the enclosing class (via MRO)
    - Local var.method           -> method on the inferred class (via MRO)
    - var(args) (object call)    -> Class.__call__ on the inferred class

Type inference is intentionally restricted to "Strategy A" (cheap, high
precision):

    1. Parameter / variable annotations.
    2. ``self`` / ``super()`` resolved against the lexically enclosing class.
    3. Local ``var = ClassName(...)`` assignments inside the same function body.

Anything that cannot be resolved with these rules is dropped.
"""

from __future__ import annotations

import ast
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _safe_parse(source: Optional[str]) -> Optional[ast.Module]:
    if source is None:
        return None
    try:
        return ast.parse(source)
    except Exception:
        return None


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def _dotted_name(node: Optional[ast.AST]) -> str:
    """Render Name / Attribute / Subscript chains as a dotted string.

    Used both for class references in annotations and for the head of a Call.
    Returns an empty string if the chain involves something unsupported (e.g.
    a function call or arbitrary expression in the middle).
    """
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        if not prefix:
            return ""
        return prefix + "." + node.attr
    if isinstance(node, ast.Subscript):
        return _dotted_name(node.value)
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    return ""


# ---------------------------------------------------------------------------
# Repo-wide symbol index
# ---------------------------------------------------------------------------

class RepoSymbolIndex:
    """A small static index over a repo, used to decide whether a name refers
    to a function / class defined inside the repo and to walk class MROs."""

    SKIP_DIRS = {
        ".git", ".github", "__pycache__", "node_modules", "build", "dist",
        "site-packages", ".tox", ".eggs", ".pytest_cache", ".mypy_cache",
        "venv", "env", ".venv", ".env",
    }

    def __init__(self, repo_dir: str):
        self.repo_dir = os.path.abspath(repo_dir)
        # rel_file -> ast.Module (cached so later mutations on disk don't matter)
        self.trees: Dict[str, ast.Module] = {}
        # rel_file -> list of import dicts: {alias, kind: 'module'|'from', module, name, level}
        self.imports: Dict[str, List[Dict]] = {}
        # rel_file -> {name: ('class'|'function', node)}
        self.toplevel: Dict[str, Dict[str, Tuple[str, ast.AST]]] = {}
        # rel_file -> {class_name: {method_name}}
        self.class_methods: Dict[str, Dict[str, Set[str]]] = {}
        # rel_file -> {class_name: [base_dotted_str, ...]}
        self.class_bases: Dict[str, Dict[str, List[str]]] = {}
        # dotted module path -> rel_file (for absolute import resolution)
        self.module_to_file: Dict[str, str] = {}

        self._build()
        self._compute_module_paths()

    # -- building -----------------------------------------------------------

    def _build(self) -> None:
        for root, dirs, files in os.walk(self.repo_dir):
            dirs[:] = [
                d for d in dirs
                if d not in self.SKIP_DIRS and not (d.startswith(".") and d != ".")
            ]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, self.repo_dir).replace(os.sep, "/")
                src = _read_text(full)
                tree = _safe_parse(src)
                if tree is None:
                    continue
                self.trees[rel] = tree
                self._index_file(rel, tree)

    def _index_file(self, rel: str, tree: ast.Module) -> None:
        imports: List[Dict] = []
        toplevel: Dict[str, Tuple[str, ast.AST]] = {}
        cmethods: Dict[str, Set[str]] = {}
        cbases: Dict[str, List[str]] = {}

        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    asname = alias.asname or alias.name.split(".")[0]
                    imports.append({
                        "alias": asname, "kind": "module",
                        "module": alias.name, "name": None, "level": 0,
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                level = node.level or 0
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    asname = alias.asname or alias.name
                    imports.append({
                        "alias": asname, "kind": "from",
                        "module": module, "name": alias.name, "level": level,
                    })
            elif isinstance(node, ast.ClassDef):
                toplevel[node.name] = ("class", node)
                methods: Set[str] = set()
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.add(sub.name)
                cmethods[node.name] = methods
                cbases[node.name] = [
                    s for s in (_dotted_name(b) for b in node.bases) if s
                ]
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                toplevel[node.name] = ("function", node)

        self.imports[rel] = imports
        self.toplevel[rel] = toplevel
        self.class_methods[rel] = cmethods
        self.class_bases[rel] = cbases

    def _compute_module_paths(self) -> None:
        """Detect package layout via __init__.py and build dotted -> rel map."""
        package_dirs: Set[str] = set()
        for rel in self.trees:
            if rel.endswith("/__init__.py"):
                package_dirs.add(rel[: -len("/__init__.py")])
            elif rel == "__init__.py":
                package_dirs.add("")

        for rel in self.trees:
            if rel == "__init__.py":
                continue
            if rel.endswith("/__init__.py"):
                pkg = rel[: -len("/__init__.py")]
                parts = pkg.split("/")
                acc = ""
                ok = True
                for i, p in enumerate(parts):
                    acc = p if i == 0 else acc + "/" + p
                    if acc not in package_dirs:
                        ok = False
                        break
                if ok:
                    self.module_to_file[".".join(parts)] = rel
            elif rel.endswith(".py"):
                stem = rel[:-3]
                parts = stem.split("/")
                acc = ""
                ok = True
                for i in range(len(parts) - 1):
                    acc = parts[i] if i == 0 else acc + "/" + parts[i]
                    if acc not in package_dirs:
                        ok = False
                        break
                if ok:
                    self.module_to_file[".".join(parts)] = rel

    # -- queries ------------------------------------------------------------

    def is_class(self, rel: str, name: str) -> bool:
        tl = self.toplevel.get(rel, {})
        return name in tl and tl[name][0] == "class"

    def is_function(self, rel: str, name: str) -> bool:
        tl = self.toplevel.get(rel, {})
        return name in tl and tl[name][0] == "function"

    def resolve_module_to_file(self, module: str,
                               current_rel: Optional[str] = None,
                               level: int = 0) -> Optional[str]:
        if level > 0 and current_rel is not None:
            cur_parts = current_rel.split("/")
            pkg_parts = cur_parts[:-1]  # drop file name
            for _ in range(level - 1):
                if pkg_parts:
                    pkg_parts = pkg_parts[:-1]
            base = "/".join(pkg_parts)
            tail = module.replace(".", "/") if module else ""
            target_path = (base + "/" + tail).strip("/") if tail else base
            candidates = [
                target_path + ".py" if target_path else "",
                (target_path + "/__init__.py") if target_path else "__init__.py",
            ]
            for c in candidates:
                if c and c in self.trees:
                    return c
            return None

        if not module:
            return None
        target_path = module.replace(".", "/")
        for c in (target_path + ".py", target_path + "/__init__.py"):
            if c in self.trees:
                return c
        return self.module_to_file.get(module)


# ---------------------------------------------------------------------------
# Statement selection (line numbers -> smallest enclosing ast.stmt)
# ---------------------------------------------------------------------------

def index_stmt_context(tree: ast.Module
                       ) -> Dict[int, Tuple[Optional[str], Optional[str]]]:
    """Map id(stmt) -> (enclosing_class, enclosing_func)."""
    ctx: Dict[int, Tuple[Optional[str], Optional[str]]] = {}

    def visit(node: ast.AST, class_name: Optional[str], func_name: Optional[str]):
        if isinstance(node, ast.stmt):
            ctx[id(node)] = (class_name, func_name)

        new_class, new_func = class_name, func_name
        if isinstance(node, ast.ClassDef):
            new_class, new_func = node.name, None
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            new_func = node.name

        for child in ast.iter_child_nodes(node):
            visit(child, new_class, new_func)

    visit(tree, None, None)
    return ctx


def select_smallest_enclosing_stmts(
    tree: Optional[ast.Module],
    line_numbers: Iterable[int],
) -> List[Tuple[ast.stmt, Optional[str], Optional[str]]]:
    """For each line, pick the smallest ast.stmt that contains it.

    Imports are skipped (they are excluded from the original edit-location
    extraction as well).  Returns deduped list of (stmt, class, func).
    """
    line_set = {ln for ln in line_numbers if ln is not None}
    if not line_set or tree is None:
        return []

    ctx = index_stmt_context(tree)
    stmts = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.stmt) and not isinstance(n, (ast.Import, ast.ImportFrom))
    ]

    selected: Dict[int, ast.stmt] = {}
    for ln in line_set:
        best, best_range = None, None
        for stmt in stmts:
            s = stmt.lineno
            e = getattr(stmt, "end_lineno", s) or s
            if s <= ln <= e:
                rng = e - s
                if best_range is None or rng < best_range:
                    best, best_range = stmt, rng
        if best is not None:
            selected[id(best)] = best

    out: List[Tuple[ast.stmt, Optional[str], Optional[str]]] = []
    for stmt in selected.values():
        cn, fn = ctx.get(id(stmt), (None, None))
        out.append((stmt, cn, fn))
    return out


# ---------------------------------------------------------------------------
# Call extraction from a stmt
# ---------------------------------------------------------------------------

def extract_calls_from_stmt(stmt: ast.stmt) -> List[ast.Call]:
    """Collect ast.Call nodes inside ``stmt``.

    For function / class def stmts we only look at the signature surface
    (decorators, default values, annotations, base classes).  For other
    statements we walk the whole body but skip nested function / class /
    lambda definitions, since calls inside them belong to those inner units
    and would not be triggered by the outer statement.
    """
    calls: List[ast.Call] = []

    def _walk_collect(node: ast.AST) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                calls.append(sub)

    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        for d in stmt.decorator_list:
            _walk_collect(d)
        args = stmt.args
        for default in list(args.defaults) + list(args.kw_defaults):
            if default is not None:
                _walk_collect(default)
        for arg_list in (args.args, args.kwonlyargs,
                         getattr(args, "posonlyargs", None) or []):
            for arg in arg_list:
                if arg.annotation is not None:
                    _walk_collect(arg.annotation)
        if stmt.returns is not None:
            _walk_collect(stmt.returns)
        return calls

    if isinstance(stmt, ast.ClassDef):
        for d in stmt.decorator_list:
            _walk_collect(d)
        for b in stmt.bases:
            _walk_collect(b)
        for kw in stmt.keywords:
            _walk_collect(kw.value)
        return calls

    def walk_skip_defs(node: ast.AST) -> None:
        if isinstance(node, ast.Call):
            calls.append(node)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef, ast.Lambda)):
                continue
            walk_skip_defs(child)

    walk_skip_defs(stmt)
    return calls


# ---------------------------------------------------------------------------
# Type environment
# ---------------------------------------------------------------------------

def _walk_skip_nested(root: ast.AST):
    yield root

    def visit(node: ast.AST):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef, ast.Lambda)):
                continue
            yield child
            yield from visit(child)

    yield from visit(root)


def build_type_env(
    func_node: Optional[ast.AST],
    current_rel: str,
    repo_index: RepoSymbolIndex,
) -> Dict[str, Tuple[str, str]]:
    """Map var_name -> (rel_file, class_name) for vars that we can prove are
    instances of repo-defined classes inside ``func_node``.

    Sources (Strategy A):
      * parameter annotations
      * ``var: Type`` annotations (AnnAssign)
      * simple ``var = ClassName(...)`` assignments

    If a var is reassigned with a different / unknown type, it is dropped.
    """
    if func_node is None or not hasattr(func_node, "args"):
        return {}

    env: Dict[str, Tuple[str, str]] = {}
    conflicts: Set[str] = set()

    def add(var: str, info: Tuple[str, str]) -> None:
        if var in conflicts:
            return
        if var in env and env[var] != info:
            conflicts.add(var)
            env.pop(var, None)
            return
        env[var] = info

    def drop(var: str) -> None:
        conflicts.add(var)
        env.pop(var, None)

    args = func_node.args
    arg_lists = [args.args, args.kwonlyargs]
    if getattr(args, "posonlyargs", None):
        arg_lists.insert(0, args.posonlyargs)
    for al in arg_lists:
        for a in al:
            if a.annotation is not None:
                info = _resolve_class_name(_dotted_name(a.annotation),
                                           current_rel, repo_index)
                if info:
                    add(a.arg, info)

    for n in _walk_skip_nested(func_node):
        if n is func_node:
            continue
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            info = _resolve_class_name(_dotted_name(n.annotation),
                                       current_rel, repo_index)
            if info:
                add(n.target.id, info)
            else:
                drop(n.target.id)
        elif isinstance(n, ast.Assign):
            if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                tname = n.targets[0].id
                if isinstance(n.value, ast.Call):
                    rhs = _resolve_class_name(_dotted_name(n.value.func),
                                              current_rel, repo_index)
                    if rhs:
                        add(tname, rhs)
                    else:
                        drop(tname)
                else:
                    drop(tname)

    return env


# ---------------------------------------------------------------------------
# Name / class resolution
# ---------------------------------------------------------------------------

def _resolve_class_name(
    name_str: str,
    current_rel: str,
    repo_index: RepoSymbolIndex,
) -> Optional[Tuple[str, str]]:
    """Resolve ``name_str`` (possibly dotted) to a repo class (rel, class_name)."""
    if not name_str:
        return None
    parts = name_str.split(".")
    head = parts[0]

    matched = None
    for imp in repo_index.imports.get(current_rel, []):
        if imp["alias"] == head:
            matched = imp
            break

    if matched is None:
        if len(parts) == 1 and repo_index.is_class(current_rel, head):
            return (current_rel, head)
        return None

    if matched["kind"] == "module":
        target_rel = repo_index.resolve_module_to_file(
            matched["module"], current_rel=current_rel, level=0)
        if target_rel and len(parts) >= 2 and repo_index.is_class(target_rel, parts[1]):
            return (target_rel, parts[1])
        return None

    # from-import
    target_rel = repo_index.resolve_module_to_file(
        matched["module"], current_rel=current_rel, level=matched["level"])
    if target_rel is None:
        return None
    imp_name = matched["name"]
    if repo_index.is_class(target_rel, imp_name):
        if len(parts) == 1:
            return (target_rel, imp_name)
        return None
    # imported a submodule: from pkg import mod -> mod.Cls
    sub_target = repo_index.resolve_module_to_file(
        f"{matched['module']}.{imp_name}" if matched["module"] else imp_name,
        current_rel=current_rel, level=matched["level"])
    if sub_target and len(parts) >= 2 and repo_index.is_class(sub_target, parts[1]):
        return (sub_target, parts[1])
    return None


def _resolve_function_call(
    name_str: str,
    current_rel: str,
    repo_index: RepoSymbolIndex,
) -> Optional[Tuple[str, str, str]]:
    """Resolve a Name (possibly dotted) used as a callable.

    Returns (rel, kind, name) where kind is 'class' or 'function'.
    """
    if not name_str:
        return None
    parts = name_str.split(".")
    head = parts[0]

    matched = None
    for imp in repo_index.imports.get(current_rel, []):
        if imp["alias"] == head:
            matched = imp
            break

    if matched is None:
        if len(parts) == 1:
            tl = repo_index.toplevel.get(current_rel, {})
            if head in tl:
                kind, _ = tl[head]
                return (current_rel, kind, head)
        return None

    if matched["kind"] == "module":
        target_rel = repo_index.resolve_module_to_file(
            matched["module"], current_rel=current_rel, level=0)
        if target_rel and len(parts) >= 2:
            tl = repo_index.toplevel.get(target_rel, {})
            if parts[1] in tl:
                kind, _ = tl[parts[1]]
                return (target_rel, kind, parts[1])
        return None

    target_rel = repo_index.resolve_module_to_file(
        matched["module"], current_rel=current_rel, level=matched["level"])
    if target_rel is None:
        return None
    imp_name = matched["name"]
    tl = repo_index.toplevel.get(target_rel, {})
    if imp_name in tl:
        if len(parts) == 1:
            kind, _ = tl[imp_name]
            return (target_rel, kind, imp_name)
        return None
    sub_target = repo_index.resolve_module_to_file(
        f"{matched['module']}.{imp_name}" if matched["module"] else imp_name,
        current_rel=current_rel, level=matched["level"])
    if sub_target and len(parts) >= 2:
        sub_tl = repo_index.toplevel.get(sub_target, {})
        if parts[1] in sub_tl:
            kind, _ = sub_tl[parts[1]]
            return (sub_target, kind, parts[1])
    return None


def _find_method_in_mro(
    rel: str,
    cls: str,
    method: str,
    repo_index: RepoSymbolIndex,
    seen: Optional[Set[Tuple[str, str]]] = None,
) -> Optional[Tuple[str, str]]:
    if seen is None:
        seen = set()
    if (rel, cls) in seen:
        return None
    seen.add((rel, cls))
    if method in repo_index.class_methods.get(rel, {}).get(cls, set()):
        return (rel, cls)
    for base in repo_index.class_bases.get(rel, {}).get(cls, []):
        info = _resolve_class_name(base, rel, repo_index)
        if info is None:
            continue
        hit = _find_method_in_mro(info[0], info[1], method, repo_index, seen)
        if hit:
            return hit
    return None


def resolve_call(
    call: ast.Call,
    current_rel: str,
    enclosing_class: Optional[str],
    type_env: Dict[str, Tuple[str, str]],
    repo_index: RepoSymbolIndex,
) -> Tuple[str, Optional[str]]:
    """Return (status, entity).

    status is one of: 'resolved', 'name_external', 'attr_unresolved',
    'no_method', 'other'.
    entity, when status == 'resolved', is "rel_file:Qualifier".
    """
    func = call.func

    # name(...)
    if isinstance(func, ast.Name):
        info = _resolve_function_call(func.id, current_rel, repo_index)
        if info is not None:
            rel, kind, name = info
            if kind == "class":
                init_loc = _find_method_in_mro(rel, name, "__init__", repo_index)
                if init_loc:
                    return ("resolved", f"{init_loc[0]}:{init_loc[1]}.__init__")
                return ("no_method", None)
            return ("resolved", f"{rel}:{name}")
        # local instance -> __call__
        if func.id in type_env:
            rel, cls = type_env[func.id]
            call_loc = _find_method_in_mro(rel, cls, "__call__", repo_index)
            if call_loc:
                return ("resolved", f"{call_loc[0]}:{call_loc[1]}.__call__")
            return ("no_method", None)
        return ("name_external", None)

    # obj.method(...)
    if isinstance(func, ast.Attribute):
        method = func.attr
        receiver = func.value

        # self.method
        if isinstance(receiver, ast.Name) and receiver.id == "self":
            if not enclosing_class:
                return ("attr_unresolved", None)
            hit = _find_method_in_mro(current_rel, enclosing_class, method, repo_index)
            if hit:
                return ("resolved", f"{hit[0]}:{hit[1]}.{method}")
            return ("attr_unresolved", None)

        # super().method  /  super(Cls, self).method
        if (isinstance(receiver, ast.Call)
                and isinstance(receiver.func, ast.Name)
                and receiver.func.id == "super"):
            if not enclosing_class:
                return ("attr_unresolved", None)
            for base in repo_index.class_bases.get(current_rel, {}).get(enclosing_class, []):
                base_info = _resolve_class_name(base, current_rel, repo_index)
                if base_info is None:
                    continue
                hit = _find_method_in_mro(base_info[0], base_info[1], method, repo_index)
                if hit:
                    return ("resolved", f"{hit[0]}:{hit[1]}.{method}")
            return ("attr_unresolved", None)

        # var.method  (var typed via type_env, or var is an imported module)
        if isinstance(receiver, ast.Name):
            var = receiver.id
            if var in type_env:
                rel, cls = type_env[var]
                hit = _find_method_in_mro(rel, cls, method, repo_index)
                if hit:
                    return ("resolved", f"{hit[0]}:{hit[1]}.{method}")
                return ("attr_unresolved", None)
            for imp in repo_index.imports.get(current_rel, []):
                if imp["alias"] == var and imp["kind"] == "module":
                    target_rel = repo_index.resolve_module_to_file(imp["module"], level=0)
                    if target_rel is None:
                        return ("name_external", None)
                    tl = repo_index.toplevel.get(target_rel, {})
                    if method not in tl:
                        return ("attr_unresolved", None)
                    kind, _ = tl[method]
                    if kind == "class":
                        init_loc = _find_method_in_mro(target_rel, method,
                                                       "__init__", repo_index)
                        if init_loc:
                            return ("resolved",
                                    f"{init_loc[0]}:{init_loc[1]}.__init__")
                        return ("no_method", None)
                    return ("resolved", f"{target_rel}:{method}")
            return ("attr_unresolved", None)

        return ("attr_unresolved", None)

    return ("other", None)


# ---------------------------------------------------------------------------
# High-level entry: per-instance support extraction
# ---------------------------------------------------------------------------

def _find_func_node(
    tree: ast.Module,
    class_name: Optional[str],
    func_name: str,
) -> Optional[ast.AST]:
    if class_name is None:
        for node in tree.body:
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == func_name):
                return node
        return None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for sub in node.body:
                if (isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and sub.name == func_name):
                    return sub
    return None


def extract_support_for_instance(
    repo_index: RepoSymbolIndex,
    file_diffs: List[Dict],
    edit_entity_strs: Iterable[str] = (),
) -> Tuple[List[str], Dict]:
    """Run support-location extraction for one benchmark instance.

    file_diffs: list of dicts, one per edited python file, with keys::

        {
          "rel_file":    str,
          "old_source":  str,            # contents BEFORE the patch
          "new_source":  str,            # contents AFTER the patch
          "delete_lines": [int, ...],    # 1-based, in OLD file coords
          "add_lines":    [int, ...],    # 1-based, in NEW file coords
        }

    edit_entity_strs: set of "<file>:<qualifier>" strings to subtract from
    the support set (complementary definition).

    Returns (sorted unique support entities, stats dict).
    """
    edit_set = set(edit_entity_strs)
    support: Set[str] = set()
    stats = {
        "total_calls": 0,
        "resolved": 0,
        "by_status": defaultdict(int),
    }

    for fd in file_diffs:
        rel = fd["rel_file"]
        for source, lines in (
            (fd.get("old_source") or "", fd.get("delete_lines") or []),
            (fd.get("new_source") or "", fd.get("add_lines") or []),
        ):
            if not source or not lines:
                continue
            tree = _safe_parse(source)
            if tree is None:
                continue
            for stmt, class_name, func_name in select_smallest_enclosing_stmts(tree, lines):
                func_node = (
                    _find_func_node(tree, class_name, func_name)
                    if func_name is not None else None
                )
                type_env = (
                    build_type_env(func_node, rel, repo_index)
                    if func_node is not None else {}
                )
                for call in extract_calls_from_stmt(stmt):
                    status, entity = resolve_call(
                        call, rel, class_name, type_env, repo_index)
                    stats["total_calls"] += 1
                    stats["by_status"][status] += 1
                    if status == "resolved" and entity is not None:
                        support.add(entity)
                        stats["resolved"] += 1

    support -= edit_set
    stats["by_status"] = dict(stats["by_status"])
    return sorted(support), stats
