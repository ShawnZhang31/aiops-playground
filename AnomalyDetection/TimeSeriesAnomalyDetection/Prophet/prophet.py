import os
import json
# from ./../Utils.data.preprocessing import load_rizhiyi_json_file_as_dataframe
from data.preprocessing import load_rizhiyi_json_file_as_dataframe
import pandas as pd
import numpy as np

root_dir = "./../../../data/riviyi_test"
filename = "o2_process_load_percent.json"
# filename = "o1.json"
# filename = "o2.json"

# 以DataFrame的格式，加载负载均衡的日志记录
log_data:pd.DataFrame = load_rizhiyi_json_file_as_dataframe(root_dir, filename)
# 根据时间戳排序
log_data.sort_values('timestamp', inplace=True, ignore_index=True)


def format_timestr(timestr:str):
    """
    重新格式化一下日志易的时间格式，原来的格式我看不习惯
    将日志易的2020-09-30:23:58:00.000格式化为2020-09-30 23:58:00
    """
    timestr = timestr.split('.')[0]
    strs = timestr.split(':')
    return strs[0] + " " + strs[1] + ":" + strs[2] + ":" + strs[3]
log_data['ts'] = [format_timestr(ts) for ts in log_data['ts']]

# 日志易的数据存在一段时间被断采的问题，需要找出断采的时间并补全上去
ts_array = log_data['ts'].to_numpy()

ts_range = pd.date_range(ts_array[0], ts_array[-1], freq='min')

# 集群负载值
cluser_values = np.zeros(len(ts_range))
ts_full = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in ts_range]
# log_data_full = pd.DataFrame(data={'value':cluser_values, 'ts':ts_full})
log_data_full = pd.DataFrame(data={'ts':ts_full})

# 使用左连接填充数据
# 一个一个的迭代太慢了，左连接可以瞬间完成
full_data = pd.merge(log_data_full, log_data, how='left', on='ts')

import plotly as py
import plotly.graph_objects as go
import plotly.express as px

# 绘制没有补充前的数据
fig = px.line(full_data, x='ts', y='value', title="Cluster Load Value without Fillna")
fig.show()
# py.offline.plot(fig, filename="./output/cluster_load_without_fillna.html")


# 是的pandas的fillna填充nan数据
full_data_fillna = full_data.fillna(method='ffill')
fig_full_data = px.line(full_data_fillna, x='ts', y='value', title="Cluster Load Value after Fillna")
fig_full_data.show()
# py.offline.plot(full_data_fillna, filename="./output/cluster_load_after_fillna.html")

# prophet的要求columns timestamp需要使用ds，value需要使用y
full_data_ds = full_data_fillna.rename(columns={"ts":"ds", 'value':'y'})

num_test = 24*60
train_data = pd.DataFrame({'ds':full_data_ds['ds'][:-num_test], "y":full_data_ds['y'][:-num_test]})

test_data = pd.DataFrame({'ds':full_data_ds['ds'][-num_test:], "y":full_data_ds['y'][-num_test:]})

from fbprophet import Prophet
import json
from fbprophet.serialize import model_to_json, model_from_json
import os

model_path = './output/json_model.json'

#如果有已经训练完成的模型，使用训练完成的模型
if os.path.exists(model_path):
    with open(model_path, 'r') as fin:
        model = model_from_json(json.load(fin))
else:
    model = Prophet()
    model.fit(train_data)
    # 保存模型
    with open(model_path, 'w') as fout:
        json.dump(model_to_json(model), fout)

# 使用模型进行预测
forecast_range = pd.DataFrame({'ds':test_data['ds']})
forecast = model.predict(forecast_range)

# 显示预测结果
fig_forecast = model.plot(forecast)
fig_forecast.show()
# py.offline.plot(fig_forecast, filename="./output/model_forecast.html")



# 会议预测区域的上下界
import plotly.graph_objects as go
train_line = go.Scatter(x=train_data['ds'], y=train_data['y'], name="train_data", line=dict(color="royalblue", width=2))
test_line = go.Scatter(x=test_data['ds'], y=test_data['y'], name="test_data", line=dict(color="firebrick", width=2))
predict_line = go.Scatter(x=forecast['ds'], y=forecast['yhat'], name="predict", line=dict(color='rgb(0, 255, 0)', width=2))
upper_bound = go.Scatter(name='Upper Bound', x=forecast['ds'], y=forecast['yhat_upper'] , mode='lines',
        line=dict(width=0.2, color="rgb(255, 188, 0)"), fillcolor='rgba(68, 68, 68, 0.2)')

lower_bound = go.Scatter(name='Lower Bound', x=forecast['ds'], y=forecast['yhat_lower'] , mode='lines',
        line=dict(width=0.2, color="rgb(255, 0, 255)"), fillcolor='rgba(68, 68, 68, 0.2)', fill='tonexty')

fig_outlier = go.Figure(data=[train_line, test_line, predict_line, upper_bound, lower_bound])
fig_outlier.update_layout(autosize=False, width=1400, height=400)
fig_outlier.show()
# py.offline.plot(fig_outlier, filename="./output/forecast_outliers.html")

