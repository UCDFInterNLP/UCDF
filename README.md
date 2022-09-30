# UCDF (User-defined Content Detection Framework)

UCDF is a set of framework to detect any kind of content that users have defined.
We conducted an exmperiment utilizing a source code, https://github.com/facebookresearch/DPR, from facebookresearch. We modified the source code to adjust the experiment in this paper. 

If you want to reproduce this work, you should install dependencies based on https://github.com/facebookresearch/DPR and additional libraries follow requirements.txt. 

### There are four steps to implement UCDF.
1. Training a retriever module based on encoder what you want to use, such as BERT or SimCSE in this experiment. (Before training a retriever module, dense embedding on all passages should be prepared.)

	- Using BERT as an encoder, this procedure is same as implemented in https://github.com/facebookresearch/DPR. (You can acceess to checkpoint of trained model and encoded passages. Also, NQ dataset for training a retriever is placed. Guidelines in the following URL tell us how to download ckpt, encoded passages and dataset.)
	- Using SimCSE or others as an encoder, you should modify files in conf folder and files in dpr folder. (You can access to checkpoint of trained model and encoded passages, generated in this experiment, in the following URL.  https://drive.google.com/drive/folders/14jbBKx-H5MPOMr-jFhGRClUd8zszs-IL?usp=sharing)


```bash
# genearte dense embedding on all the passages. 
bash generated_DE.sh

# training a retriever module
bash train_encoder.sh
```


2. After training a retriever module, we should construct a customized dataset which is used for training a binary classification task using the trained retriever module. We should prepare two kinds of dataset (positive sampels / negative samples)
	- Positive samples: positive queries-based
	- Negative samples: only random based, only negative queries-based, 50% random + 50% negative queries-based
	- The examples of customized dataset used for hate speech detection task in this paper can be downloaded in the following URL. 
	- hate speech detection task's retreived data: https://drive.google.com/file/d/1Im2xjPmkEetKBVwjDZ0gwBIIHv4CI6gw/view?usp=sharing
	- user-defined content detection task's retreived data: https://drive.google.com/file/d/1px0FvPg4OxQdDxZWE0tIi7uhbP1h9qV5/view?usp=sharing)

```bash
# you should select threshold strategy.
python build_customized_dataset.py
```

3. Fine-tuning a classifier with the customized datset
	- We utilzied a sentence encoder, same encoder type used in training retriever module for fine-tuning. Sentence encoder is fine-tuned with [CLS] token using BCE loss. Training time is different upto selecting building positive samples strategy (min/avg). 
	- materials related to this task is in 'fine_tuning' folder.
	- materials related to fine-tuning, such as checkpoint and dataset can be downloaded in the following URL.
	- hate speech test data: https://drive.google.com/drive/folders/1SFG3S7t6hDZoTNJeMvGLZMUuoxqW4WMo?usp=sharing
	- checkpoint: https://drive.google.com/file/d/1neuogXLuKbJBsFsc7lFJNNcSFh_SRIgX/view?usp=sharing
	- Customized test data: We can check customized test data which is used in User-content detection task (Section4.2)
	- South Korea test data, Relgion test data: https://drive.google.com/drive/folders/1lO_mhJctikmuFTiT5TaEOw1hP3Rn2-kk?usp=sharing

4. Evaluate the quality of UCDF.
	- We evalute the quality of UCDF with qualitative method and quantitative method. You can conduct quantitative experiment if there exists benchmark dataset. Since there are not a lot of benchmark datasets related to User-defined content (content-agnostic) in a real-world, you probably should implement using qualitative analysis.
	- materials related to this task is in 'fine_tuning' folder.
	- If you want to evaluate the quality of UCDF on other contents quantitatively, you should build your own customized test data. See (5. Customized test data).


```bash
# code related to fine-tuning and evaluate is in sample python file
# you should edit dataset and encoder type upto task
python fine_tuning.py
```

