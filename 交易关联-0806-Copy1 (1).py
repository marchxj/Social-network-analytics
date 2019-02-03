
# coding: utf-8

# In[1]:

from  hxdb import * # 华夏TD包 
import datetime # 时间相关
import time # 计时相关
import pandas as pd # 数据存储
import numpy as np
from  sklearn.linear_model  import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt # 画图相关
get_ipython().magic('matplotlib inline')
pd.set_option('display.max_columns', None) # 显示所有列
import warnings #警告
warnings.filterwarnings('ignore') #忽略所有warning

#udaexec=teradata.UdaExec() # 服务器配置
#session=udaexec.connect("${dataSourceName}") #  服务器配置

a = time.time()
sql = '''select * from tmp_data.dep_flow_3 '''
dep_flow_3 = tdsql(sql)
b = time.time()
print('time used: ', b - a)

#原始数据：窗口期：（三个月 4、5、6月） ，共29万交易关联
#Db_Cr_Ind:交易方向  Txn_Cs：交易总笔数 Lj_Amt：交易总金额  Cntrpty_Ind：我行客户
dep_flow_3.head(100)

#除去部分对手客户号四位的流水   票据清算一类  如Cntrpty_Cust_Id='1796'
grouped_flow = dep_flow_3.loc[(dep_flow_3['Cntrpty_Cust_Id'].isin(list(dep_flow_3['Cust_No']))) | (dep_flow_3['Cntrpty_Cust_Id'] == '')]

#生成一份新数据，用于计算我行客户窗口期内累计转出转入总计
#miqie_flow = grouped_flow
#miqie_flow = miqie_flow[['Cust_No','Pty_Name','Cntrpty_Cust_Id','Cntrpty_Acct_Name','Cntrpty_Ind','Lj_Amt']]

#miqie_flow['Lj_Amt'] = abs(miqie_flow['Lj_Amt'])

#计算我行客户在窗口期内转出总计，转入总计
#grouped_flow = dep_flow_3.copy()
grouped_flow = grouped_flow[grouped_flow['Cust_No'] != grouped_flow['Cntrpty_Cust_Id']]#删除行内同名账户互转

out_flow = grouped_flow.loc[grouped_flow['Db_Cr_Ind'] == '1']
in_flow = grouped_flow.loc[grouped_flow['Db_Cr_Ind'] == '0']

in_Lj = in_flow['Lj_Amt'].groupby(in_flow['Cust_No']).sum()
out_Lj = out_flow['Lj_Amt'].groupby(out_flow['Cust_No']).sum()
sum_Lj = grouped_flow['Lj_Amt'].groupby(grouped_flow['Cust_No']).sum()

#生成字典,用于添加至输出汇总表中
dict_sum_Lj = dict(sum_Lj)
dict_in_Lj = dict(in_Lj)
dict_out_Lj = dict(out_Lj)

#累计转账总计
Lj_flow = grouped_flow[['Cust_No','Pty_Name']].drop_duplicates()

#行内客户列表，用于输出汇总表

#用来建立网络的数据
#当转账双方均为行内账户时，该条流水会有两笔记录  保留对手方为我行,同时交易方向为1的流水以及全部交易对手为行外的流水
dep_flow = grouped_flow.copy()
dep_flow = dep_flow[(dep_flow['Cntrpty_Cust_Id'] != '') & (dep_flow['Db_Cr_Ind'] == '1') |
                    ( dep_flow['Cntrpty_Cust_Id'] == '')]
#删除交易双方客户号相同的流水
dep_flow = dep_flow[dep_flow['Cust_No'] != dep_flow['Cntrpty_Cust_Id']]

#行内账户客户号字典
Cust_dict = dict(zip(dep_flow['Cust_No'],dep_flow['Pty_Name']))

#根据交易对手行外行内做分组
d1 = dep_flow.copy()
d2 = dep_flow.copy()
d1 = d1.loc[d1['Cntrpty_Cust_Id'] == ''] # 行外
d2 = d2.loc[d2['Cntrpty_Cust_Id'] != ''] #行内
acct_Name = d1['Cntrpty_Acct_Name']

##### 行外虚拟账户客户号字典名称 Cntrpty_dict

#给行外客户创建虚拟号码
Cntrpty_dict = dict(zip(acct_Name,range(len(acct_Name))))#虚拟客户号字典

d1['fake_Cn_number'] = d1['Cntrpty_Acct_Name'].map(Cntrpty_dict)

d2['fake_Cn_number'] = d2['Cntrpty_Cust_Id']
#把号码统一到一列，若为行内账户 则补满17位 用以区分
d2['fake_Cn_number'] = d2['fake_Cn_number'].str.zfill(17) 

HangWai = d1[['Cntrpty_Acct_Name','fake_Cn_number']].drop_duplicates() # 行外客户代号 


### 转到行外同名账户

#同名账户转出 转账金额超过一亿

hangwai_tongming = d1.loc[d1['Pty_Name'] == d1['Cntrpty_Acct_Name']]

hangwai_tongming = hangwai_tongming.loc[(hangwai_tongming['Lj_Amt'] > 100000000)|(hangwai_tongming['Lj_Amt'] < -100000000)].reset_index(drop = True)

#行外客户虚拟客户号字典转为excel


### 开始建立网络 计算数据

#用来建立网络的数据
zijin_flow = pd.concat([d1,d2]).reset_index(drop = True)

#调整转账金额为绝对值，调整金额大小 除以10000000
zijin_flow['adj_Lj_Amt'] = abs(zijin_flow['Lj_Amt'])
zijin_flow['adj_Lj_Amt'] = zijin_flow['adj_Lj_Amt'].map(lambda x: x/1000000)

#创建测试集 样本数量 2W
flow = zijin_flow#.sample(n = 2000)


#建立网络来计算几个度量值
import networkx as nx

G = nx.MultiDiGraph()

#添加带有方向的边信息
sample_1 = flow.loc[flow['Db_Cr_Ind'] == '1']
sample_0 = flow.loc[flow['Db_Cr_Ind'] == '0']

edge_1 = list(zip(list(sample_1['Cust_No']),list(sample_1['fake_Cn_number']),list(sample_1['adj_Lj_Amt'])))
edge_0 = list(zip(list(sample_0['fake_Cn_number']),list(sample_0['Cust_No']),list(sample_0['adj_Lj_Amt'])))

#向网络添加边,边的粗细 = 转账金额大小
G.add_weighted_edges_from(edge_1) 

G.add_weighted_edges_from(edge_0)

#生成网络图的代码

#edge_weight = [G.edges[c]['weight'] for c in G.edges]

#fig = plt.figure(figsize = (100,100),facecolor = 'white')
#pos = nx.spring_layout(G)
#nx.draw_networkx(G,pos,with_labels = False, node_size = 2)

### 生成客户相关信息表

#生成客户相关信息表
dlist = list(G.node)
dlen = len(dlist)
temp = []
deg_dict = nx.degree_centrality(G)#连接度：与其连接的点的个数占全部点的百分数bet_dict = nx.betweenness_centrality(G)#某两点之间最短线路会经过该点次数 百分数

close_dict = nx.closeness_centrality(G,normalized = True)#亲和度:到其他所有点的距离之和 

title = ['客户','转出客户数','转入客户数','连接度','亲和度']
for i in range(dlen):
    source = dlist[i]
    outputs = list(G.successors(source))
    inputs = list(G.predecessors(source))
    deg_value = deg_dict[source]
    close_value = close_dict[source]
    temp.append((source,len(outputs),len(inputs),deg_value, close_value)) #['客户','转出客户数','转入客户数','连接度','亲和度']
result = pd.DataFrame(temp,columns = title)


### 汇总表


#行内信息关联表
Cust = pd.merge(Lj_flow,result,left_on = 'Cust_No' ,right_on = '客户' )
Cust = Cust[['Cust_No','Pty_Name','转出客户数','转入客户数','连接度','亲和度']]
Cust = Cust.drop_duplicates()

Cust['累计转出金额'] = Cust['Cust_No'].replace(out_Lj)
Cust['累计转入金额'] = Cust['Cust_No'].replace(in_Lj)
Cust.loc[Cust['累计转出金额'] == Cust['Cust_No'],['累计转出金额']] = False
Cust.loc[Cust['累计转入金额'] == Cust['Cust_No'],['累计转入金额']] = False
Cust['动账总计'] = Cust['累计转入金额'] - Cust['累计转出金额']



#行外信息表
Cntrpty = pd.merge(HangWai,result,left_on = 'fake_Cn_number' ,right_on = '客户' )
Cntrpty = Cntrpty[['Cntrpty_Acct_Name','转出客户数','转入客户数','连接度','亲和度']]
Cntrpty = Cntrpty.drop_duplicates()
Cntrpty = Cntrpty.reset_index(drop = True)
#按照连接度排序
Cntrpty = Cntrpty.sort_values("连接度",ascending = False)

Cntrpty = Cntrpty.reset_index(drop = True)

#按照亲和度排序
Cust = Cust.sort_values("亲和度",ascending = False)
Cust = Cust.reset_index(drop = True)

#1.根据重要客户： 客户号为'VIP' 的客户找到与他有联系的客户
#2.部分数据的可视化

VIP = '00000000075899420'

sub_G = nx.MultiDiGraph()

sub_flow = flow.loc[flow['Cust_No'] == VIP]

#添加带有方向的边信息
sub_1 = sub_flow.loc[flow['Db_Cr_Ind'] == '1']
sub_0 = sub_flow.loc[flow['Db_Cr_Ind'] == '0']

sub_edge_1 = list(zip(list(sub_1['Cust_No']),list(sub_1['fake_Cn_number']),list(sub_1['adj_Lj_Amt'])))
sub_edge_0 = list(zip(list(sub_0['fake_Cn_number']),list(sub_0['Cust_No']),list(sub_0['adj_Lj_Amt'])))

#向网络添加边,边的粗细 = 转账金额大小
sub_G.add_weighted_edges_from(sub_edge_1) 

sub_G.add_weighted_edges_from(sub_edge_0)

fig = plt.figure(figsize = (25,25),facecolor = 'white')
pos = nx.shell_layout(sub_G)
nx.draw_networkx(sub_G,with_labels = False, node_size = 50,with_arrows = True)

hangwai_tongming.to_excel('/share/zhftt/行外同名账户转移.xlsx')

Cntrpty.to_excel('/share/zhftt/行外客户维度表.xlsx')
Cust.to_excel('/share/zhftt/行内客户维度信息汇总表.xlsx')


# In[ ]:



