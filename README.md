# salary_prediction

This repo tries to predict the salary for UK job postings. The data was downloaded from: https://www.kaggle.com/c/job-salary-prediction. 

The 'preprocessing' notebook is the main notebook where the data is preprocessed and then fed into a model.

At this point, only the JobSummary is used as input. NLP was performed on it and a fed into a word embedding layer with a pretrained weigths matrid from the GloVe embedding.

Various models such as MLPs and CNNs were trained and evaluated. 
