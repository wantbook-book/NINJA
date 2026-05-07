import ast
import json
import os

from tools.RepoSearch.preprocess_data import extract_structure
from tools.utils.utils import load_json


def _load_structure(instance_id: str):
    project_file_loc = os.environ.get("PROJECT_FILE_LOC", None)
    d = load_json(f"{project_file_loc}/{instance_id}.json")
    return d["structure"]


def _get_repo_items(instance_id: str):
    structure = _load_structure(instance_id)
    return extract_structure(structure)


def _get_file_content(file_name: str, instance_id: str):
    files, _, _ = _get_repo_items(instance_id)
    for item in files:
        if item[0] == file_name:
            return "\n".join(item[-1])
    return None


def _parse_file_ast(file_name: str, instance_id: str):
    file_content = _get_file_content(file_name, instance_id)
    if file_content is None:
        return None, None, "You provide a wrong file name. Please try another file name again."

    try:
        return ast.parse(file_content), file_content, None
    except SyntaxError as exc:
        return None, file_content, f"Failed to parse {file_name}: {exc}"


def _unparse_expr(node):
    if node is None:
        return None
    return ast.unparse(node)


def _format_arg(arg: ast.arg, default=None, prefix: str = "") -> str:
    rendered = f"{prefix}{arg.arg}"
    if arg.annotation is not None:
        rendered += f": {_unparse_expr(arg.annotation)}"
    if default is not None:
        rendered += f" = {_unparse_expr(default)}"
    return rendered


def _format_parameters(args: ast.arguments) -> str:
    parts = []
    positional_args = list(args.posonlyargs) + list(args.args)
    defaults = [None] * (len(positional_args) - len(args.defaults)) + list(args.defaults)

    for index, arg in enumerate(args.posonlyargs):
        parts.append(_format_arg(arg, defaults[index]))
    if args.posonlyargs:
        parts.append("/")

    for index, arg in enumerate(args.args, start=len(args.posonlyargs)):
        parts.append(_format_arg(arg, defaults[index]))

    if args.vararg is not None:
        parts.append(_format_arg(args.vararg, prefix="*"))
    elif args.kwonlyargs:
        parts.append("*")

    for kwarg, default in zip(args.kwonlyargs, args.kw_defaults):
        parts.append(_format_arg(kwarg, default))

    if args.kwarg is not None:
        parts.append(_format_arg(args.kwarg, prefix="**"))

    return ", ".join(parts)


def _build_function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    signature = f"{prefix} {node.name}({_format_parameters(node.args)})"
    if node.returns is not None:
        signature += f" -> {_unparse_expr(node.returns)}"
    return signature


def _build_class_signature(node: ast.ClassDef) -> str:
    parents = [_unparse_expr(base) for base in node.bases]
    keywords = []
    for keyword in node.keywords:
        if keyword.arg is None:
            keywords.append(f"**{_unparse_expr(keyword.value)}")
        else:
            keywords.append(f"{keyword.arg}={_unparse_expr(keyword.value)}")

    signature_args = [item for item in parents + keywords if item]
    if signature_args:
        return f"class {node.name}({', '.join(signature_args)})"
    return f"class {node.name}"


def _serialize_class_node(node: ast.ClassDef):
    return {
        "signature": _build_class_signature(node)
    }


def _find_class_node(tree: ast.AST, class_name: str):
    matched_classes = [
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if not matched_classes:
        return None
    return sorted(matched_classes, key=lambda node: (node.lineno, node.end_lineno or node.lineno))[0]

def get_functions_of_class(class_name: str, instance_id: str):
    files, classes, functions = _get_repo_items(instance_id)

    functions_in_class = []
    for item in classes:
        if item['name'] == class_name:
            for method in item['methods']:
                functions_in_class.append(method['name'])

    return str(functions_in_class)


def get_functions_of_file(file_name: str, instance_id: str):
    files, classes, functions = _get_repo_items(instance_id)
    func_in_file = []
    for item in functions:
        if item['file'] == file_name:
            func_in_file.append(item['name'])

    return str(func_in_file)


def get_classes_of_file(file_name: str, instance_id: str):
    files, classes, functions = _get_repo_items(instance_id)
    classes_in_file = []
    for item in classes:
        if item['file'] == file_name:
            classes_in_file.append(item['name'])

    return str(classes_in_file)


def get_code_of_file(file_name: str, instance_id: str):
    files, classes, functions = _get_repo_items(instance_id)
    file_content = "You provide a wrong file name. Please try another file name again."
    for item in files:
        if item[0] == file_name:
            file_content = "\n".join(item[-1])

    return file_content


def get_code_of_class(file_name: str, class_name: str, instance_id: str):
    files, classes, functions = _get_repo_items(instance_id)

    for item in classes:
        if item['file'] == file_name and item['name'] == class_name:
            return "\n".join(item['class_content'])
    return "You provide a wrong file name or class name. Please try another file name again."


def get_code_of_class_function(file_name: str, class_name: str, func_name: str, instance_id: str):
    files, classes, functions = _get_repo_items(instance_id)

    for item in classes:
        if item['file'] == file_name and item['name'] == class_name:
            for method in item['methods']:
                if method['name'] == func_name:
                    return "\n".join(method['method_content'])

    return "You provide a wrong file name or class name or function name. Please try another file name again. It may be a file function."


def get_code_of_file_function(file_name: str, func_name: str, instance_id: str):
    files, classes, functions = _get_repo_items(instance_id)

    for item in functions:
        if item['file'] == file_name and item['name'] == func_name:
            return "\n".join(item['text'])

    return "You provide a wrong file name or function name. Please try another file name again. It may be a class function."


def get_methods_of_class(file_name: str, class_name: str, instance_id: str):
    tree, _, error = _parse_file_ast(file_name, instance_id)
    if error is not None:
        return json.dumps({"error": error}, ensure_ascii=False)

    class_node = _find_class_node(tree, class_name)
    if class_node is None:
        return json.dumps(
            {"error": "You provide a wrong file name or class name. Please try another file name again."},
            ensure_ascii=False,
        )

    methods = [
        _build_function_signature(node)
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return json.dumps({"methods": methods}, ensure_ascii=False)


def get_file_functions(file_name: str, instance_id: str):
    tree, _, error = _parse_file_ast(file_name, instance_id)
    if error is not None:
        return json.dumps({"error": error}, ensure_ascii=False)

    functions = [
        _build_function_signature(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return json.dumps({"functions": functions}, ensure_ascii=False)


def get_file_classes(file_name: str, instance_id: str):
    tree, _, error = _parse_file_ast(file_name, instance_id)
    if error is not None:
        return json.dumps({"error": error}, ensure_ascii=False)

    classes = [_serialize_class_node(node) for node in tree.body if isinstance(node, ast.ClassDef)]
    return json.dumps(
        {
            "file_name": file_name,
            "classes": classes,
        },
        ensure_ascii=False,
    )

def get_all_of_files(instance_id: str):
    files, classes, functions = _get_repo_items(instance_id)
    file_names = set()
    for item in files:
        file_names.add(item[0])

    return list(file_names)

def get_imports_of_file(file_name: str, instance_id: str):
    files, classes, functions = _get_repo_items(instance_id)
    imports = []
    for item in files:
        if item[0] == file_name:
            for line in item[-1]:
                if line.startswith("import") or (line.startswith("from") and "import" in line):
                    imports.append(line)
    return imports


if __name__ == '__main__':
    # print(get_functions_of_class('WCS', 'astropy__astropy-7746'))
    # print(get_code_of_file_function('astropy/wcs/wcs.py', '_return_single_array', 'astropy__astropy-7746'))
    # print(get_all_of_files('get_all_of_files'))
    print(get_imports_of_file('sympy/matrices/expressions/blockmatrix.py', 'sympy__sympy-17630'))
