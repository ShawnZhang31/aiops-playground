import os
import json
import pandas as pd
from pandas import DataFrame

def load_rizhiyi_json_file_as_dataframe(root_dir:str, filename:str)->DataFrame:
    """将日志易的json格式的数据加载为Pandas的DataFrame格式的数据

    Args:
        root_dir (str): 数据文件的目录
        filename (str): 数据文件的名称

    Returns:
        DataFrame: 放回pandas DataFrame格式的数据
    """
    file_path = os.path.join(root_dir, filename)
    data_array = []
    
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            try:
                js_data = json.loads(line)
                data_array.append(js_data)
            except ValueError:
                print("the params passed to json.loads() must be json format string")
                raise
    return DataFrame(data_array)