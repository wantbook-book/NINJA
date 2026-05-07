import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from dataclasses import dataclass

from plugins.location_tools import locationtools
from plugins.requirement import PluginRequirement #, Plugin, 


@dataclass
class LocationToolsRequirement(PluginRequirement):
    name: str = 'location_tools'
    documentation: str = locationtools.DOCUMENTATION


# class LocationToolsPlugin(Plugin):
#     name: str = 'location_tools'
