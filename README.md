##1.运行
python interhat/main.py

##2.数据分布
data/parse/avazu/*.csv

##3.训练生成的模型文件
log/avazu下所有文件，暂时没用。

##4. 我们得到的可解释文件

performance/ml/*

  .1.csv : 第一次visit时，各个code所占的权重
  
  .2.csv : 第二次visit时，各个code所占的权重
  
  .3.csv : 第三次visit时，各个code所占的权重
  
  .4.csv : 三次visit所占的权重

注意，现在代码每次epoch都会产生以上四个文件。

##5. run_evaluation

main.py中的run_evaluation方法，看下那里调用。每次调用就是获取下此时，预测测试集时候的准确率和AUC。


##6. 多个模型微微修改说明

interhat/model.py          ==>就我们提出的模型

interhat/model_NOLSTMWITHVISIT.py  ==>就我们提出的模型去掉LSTM

interhat/model_NOTransformer.py   ==>就我们提出的模型去掉Transformer




