
class InstanceContext:
    """Context object to store instance-specific state"""
    def __init__(self, instance_id: str, instance_data: dict):
        self.instance_id = instance_id
        self.instance_data = instance_data
        self.all_file = None
        self.all_class = None
        self.all_func = None
        self.entity_searcher = None
        self.dependency_searcher = None
        self.graph = None
        self.repo_save_dir = None