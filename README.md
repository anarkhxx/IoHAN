IoHAN
=========================================

IoHAN, an interpretable outcome prediction model by leveraging attention mechanism. The main novelty of IoHAN is that it can pinpoint the fine-grained influence on the final 
  prediction result of each medical component by decomposing the attention weights hierarchically into hospital visits, medical variables, and interactions between medical variables. We evaluated IoHAN on MIMIC-III, a large real-world EHR dataset. The experiment results demonstrate that IoHAN can achieve higher prediction accuracy than state-ofthe-art outcome prediction models. In addition, the hierarchical decomposed attention weights can interpret the prediction results in a more natural and understandable way. 

#### Running IoHAN

**STEP 1: Fastest way to run IoHAN with MIMIC-III**  

1.You need ADMISSIONS.csv, DIAGNOSES_ICD.csv, PATIENTS.csv these three files which can be downloaded in (https://mimic.physionet.org/gettingstarted/access/), a publicly avaiable electronic health records collected from ICU patients over 11 years.

2.Put these three files in the preprocess folder and run process_mimic_modified.py. Next you will get eight files containing the main information.

3.Run processMyData.py, get training set and test set.

4.Then you can just run main.py.

**STEP 2: How to pretraining the code embedding useing TF-IDF**  

You only need to uncomment lines 103-110 and lines 136-143 of the processMyData.py file, and replace the code on lines 113 and 146 with sorted_result.Then re-run the processMyData.py file.

**STEP 3: How to pretraining the code embedding useing gram**

You only need to uncomment lines 17-63 and lines 161 of the modules.py file, and comment lines 162-165.Then re-run the modules.py file.





