""" dynamically load settings

author baiyu
"""
from mycode.conf import global_settings

class Settings:
    def __init__(self, settings):

        for attr in dir(settings):
            if attr.isupper():
                setattr(self, attr, getattr(settings, attr))

settings = Settings(global_settings)