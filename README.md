# Cell-ID Trajectory Prediction Using Multi-Graph Embedding and Encoder-Decoder Network
   Trajectory prediction for mobile phone users is a cornerstone component to support many higher-level applications in LBSs (Location-Based Services). Most existing methods are designed based on the assumption that the explicit location information of the trajectories is available (e.g., GPS trajectories). However, collecting such kind of trajectories lays a heavy burden on the mobile phones and incurs privacy concerns. In this paper, we study the problem of trajectory prediction based on cell-id trajectories without explicit location information and propose a deep learning framework (called DeepCTP) to solve this problem. Specifically, we use a multi-graph embedding method to learn the latent spatial correlations between cell towers by exploiting handoff patterns. Then, we design a novel spatial-aware loss function for the encoder-decoder network to generate cell-id trajectory predictions.
   
The paper can be visited at https://ieeexplore.ieee.org/abstract/document/9318015

# Requirement

       1.python=3.6
       2.tensorflow-gpu=1.8.0
       3.keras=2.2.0

# Data Preparation
   
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

