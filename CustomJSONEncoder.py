import json


class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles objects with __dict__ attribute 
    and converts other non-serializable objects to strings
    """
    def default(self, obj):
        try:
            return obj.__dict__
        except AttributeError:
            return str(obj)