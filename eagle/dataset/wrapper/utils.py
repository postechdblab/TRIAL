from typing import *


def cache_and_get_dic_key_type(cls, dic_name: str, dic: Dict) -> Any:
    key_type_cache_var_name = f"_{dic_name}_key_type"
    assert type(dic) == dict, f"dic must be a dict, but got {type(dic)}"

    if len(dic) == 0:
        return None

    # Get the type
    if not hasattr(cls, key_type_cache_var_name):
        key_type = type(list(dic.keys())[0])
        setattr(cls, key_type_cache_var_name, key_type)
    return getattr(cls, key_type_cache_var_name)
