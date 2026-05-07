import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
# Requirements
from plugins.location_tools import (
    # AgentSkillsPlugin,
    LocationToolsRequirement,
)
# from openhands.runtime.plugins.jupyter import JupyterPlugin, JupyterRequirement
from plugins.requirement import PluginRequirement #, Plugin

__all__ = [
    # 'Plugin',
    'PluginRequirement',
    # 'AgentSkillsRequirement',
    # 'AgentSkillsPlugin',
    # 'JupyterRequirement',
    # 'JupyterPlugin',
    'LocationToolsRequirement'
]

# ALL_PLUGINS = {
#     'jupyter': JupyterPlugin,
#     'agent_skills': AgentSkillsPlugin,
# }


