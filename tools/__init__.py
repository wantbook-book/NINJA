from .repo_search_tools import (
    get_code_of_class,
    get_code_of_class_function,
    get_code_of_file_function,
    get_file_classes,
    get_file_functions,
    get_methods_of_class,
)

from .multi_agent_tools import (
    get_callers,
    get_callees,
    get_code_of_function,
    get_code_of_class_method,
    find_files_by_content_repo,
    find_files_by_name_repo,
)

tools_registries = {
    'repo_search_tools': {
        'get_code_of_class': get_code_of_class,
        'get_code_of_class_function': get_code_of_class_function,
        'get_code_of_file_function': get_code_of_file_function,
        'get_methods_of_class': get_methods_of_class,
        'get_file_functions': get_file_functions,
        'get_file_classes': get_file_classes,
    },
    'multi_agent_sub_tools': {
        'get_file_functions': get_file_functions,
        'get_file_classes': get_file_classes,
        'get_code_of_function': get_code_of_function,
        'get_code_of_class_method': get_code_of_class_method,
        'get_callers': get_callers,
        'get_callees': get_callees,
    },
    'multi_agent_main_tools': {
        'find_files_by_content': find_files_by_content_repo,
        'find_files_by_name': find_files_by_name_repo,
    },
}
