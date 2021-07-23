# An Interpretable Outcome Prediction Model Based on Electronic Health Records and Hierarchical Attention
   IoHAN, an interpretable outcome prediction model by leveraging attention mechanism. The main novelty of IoHAN is that it can pinpoint the fine-grained influence on the final
prediction result of each medical component by decomposing the attention weights hierarchically into hospital visits, medical variables, and interactions between medical variables. We evaluated IoHAN on MIMIC-III, a large real-world EHR dataset. The experiment results demonstrate that IoHAN can achieve higher prediction accuracy than state-ofthe-art outcome prediction models. In addition, the hierarchical decomposed attention weights can interpret the prediction results in a more natural and understandable way.
   
The paper can be visited at https://ieeexplore.ieee.org/abstract/document/9318015

# Requirement

1.Keras==2.1.2  
2.matplotlib==3.3.2  
3.numpy==1.19.2  
4.pandas==1.1.3  
5.Pillow==8.0.1  
6.PyYAML==5.3.1  
7.scikit-learn==0.22  
8.scipy==1.5.3  
9.sklearn==0.0  
10.tensorboard==1.8.0  
11.tensorflow==1.8.0  

   
(1)src/distance_cells/out1.txt  This file contains all the cell-id trajectories.

(2)src/distance_cells/out1_train.txt  This file contains the cell-id trajectories used for learning the cell-id embeddings.

(3)src/data/dl-data/couplet/vocabs  This file contains all the unique cell-ids.

(4)src/data/dl-data/couplet/train/in.txt  This file contains all the input sub-trajectories of the training samples.

(5)src/data/dl-data/couplet/train/out.txt  This file contains all the output sub-trajectories of the training samples.

(6)src/data/dl-data/couplet/test/intest.txt  This file contains all the input sub-trajectories of the testing samples.

(7)src/data/dl-data/couplet/test/outtest.txt  This file contains all the output sub-trajectories of the testing samples.

### Data format description

The basic component in all the data files is cell-id defined in the format of “C10000A20000”, where 10000 is the CellID (Cell Tower ID) and 20000 is the LAC (Location Area Code).

# Run the Project

### Run the command below to train the model:

      python couplet.py       
   
   The training results are in the src/data/dl-data/models
   
### Run the command below to test the model:

      python forqatest.py     
  
The input sub-trajectories can be configured in the code, with an example as follows.

```python
#input
qlist=['C8062A13844','C10365A22535','C10361A22535','C18524A22299','C10361A22535']
#prediction
res=inferTheStr(qlist)
#output
print(res)
```

The results are output in src/result.txt

