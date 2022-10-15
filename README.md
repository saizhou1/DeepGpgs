## DeepGpgs
Protein arginine methylation is an important post-translational modification, which is related to protein functional diversity and pathological conditions such as cancer. For example, overexpression of arginine methylation has been observed in breast cancer and colon cancer, and the identification of methylation binding sites is beneficial to better understand the molecular function of proteins. However, the traditional experimental methods to identify methylation sites are not only expensive and laborious, but also have a long cycle. Nowadays, with the advent of the era of big data, deep learning-based methylation recognition is popular due to its accurate and fast prediction ability.

In this paper, we design a deep learning model DeepGpgs incorporating Gaussian prior and gated attention mechanism. First, we introduce a residual network channel to obtain the evolutionary information of proteins. The other channel consists of adaptive embedding and bidirectional long short-term memory network to form a context-shared coding layer. After that, the global information of the sequence is acquired through the multi-head attention mechanism based on gating, and Gaussian prior is injected into the sequence to assist the prediction of loci. Finally, we propose a weighted joint loss function to alleviate the false negative problem of modified sites.


## Requirement
hiddenlayer == 0.3  
pandas == 1.2.4  
scikit-learn == 0.24.0  
scipy == 1.7.2  
seaborn == 0.11.2  
torch == 1.4.0  
tqdm == 4.55.1  
matplotlib == 3.4.2  
numpy == 1.19.5  

## Run DeepGpgs for prediction
Prediction based on arginine methylation dataset:

```
python ./ptm_r.py\
  --train_file ./data/uniprot_r_train.fasta\
  --test_file ./data/uniprot_r_test.fasta\
  --save_path ./data/output/model.pth
```
To further verify the generalization performance of the DeepGpgs model, the Serine/threonine phosphorylation site dataset was further predicted as follows:

```
python ./ptm_st.py
```
  And the prediction of Y phosphorylation modification sites:

```
python ./ptm_y.py
```
