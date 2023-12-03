---

# **FactGuard - A Fake News Classifier based on NLP and ML**

---

## **Problem Description**

Fake news is information that is explicitly incorrect & misleading. It is used to harm a person or entity's reputation or to profit from advertising income. With the rise of social media and the sheer volume of people who use the internet, there is no shortage of online objects vying for our attention. The ease with which people may republish content & the lucrative nature of publishing on social media has allowed fake news to flourish. These days, fake news is considered a major threat to countries worldwide. Hence, it is of utmost importance we find a way to inhibit its spread if not stop it entirely. Fake news classifiers using NLP techniques with ML to examine word patterns and statistical correlations have been theorized by many. 

---

## **Objective**

As a beginner step, we wish to implement such a web application, which when provided with the title and body text from a news article, would try to determine if the article is truthful or fake and also the percentage chance it is truthful or fake.

---

## **Methodology**

We have used the below-given methodology for working on this application (do note that it is a high-level view and doesn't go into specifics):

1. **Gathering data:** Involves manual scraping of data using scripts from various sources or just using existing datasets.
2. **Cleaning & processing:** Crucial step which involves techniques like tokenization, stop-words removal, and so on. Data collected from different sources have their parameters normalized in this step.
3. **Exploratory data analysis:** The processed data is explored to gain an understanding of any patterns, trends, or anomalies that can be used to aid in modeling.
4. **Model selection & evaluation:** Various classification models are run over the data to decide which model is the best fit. Performance metrics are then optimized if feasible.
5. **Deployment:** The trained model is deployed online so as to turn it into a GUI with a dashboard for easy user access.

---

## **The Datasets**

### a. Primary Dataset

- *Link to the dataset:* <https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset>
- *Author:* **Clement Bisaillon**
- We are using v1 of the dataset that was uploaded on 26th March 2020.

#### **Some basic information regarding the dataset:**

- The data was originally collected by the University of Victoria ISOT Research Lab from real-world sources.
- The collection techniques and in-depth information about the dataset are available on [this link](https://www.uvic.ca/engineering/ece/isot/assets/docs/ISOT_Fake_News_Dataset_ReadMe.pdf), for those interested.
- Truthful articles were obtained by crawling articles from reuters.com (News website)
- Fake news articles were collected from different sources, usually unreliable websites that were flagged by Politifact (a fact-checking organization in the U.S.) and Wikipedia.
- The dataset contains different types of articles on different topics, however, the majority of articles focus on political and world news topics, between 2015 - 2018.

#### **Citations requested by dataset author:**

1. *Ahmed H, Traore I, Saad S. "Detecting opinion spams and fake news using text classification", Journal of Security and Privacy, Volume 1, Issue 1, Wiley, January/February 2018.*

2. *Ahmed H, Traore I, Saad S. (2017) "Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques. In: Traore I., Woungang I., Awad A. (eds) Intelligent, Secure, and Dependable Systems in Distributed and Cloud Environments. ISDDC 2017. Lecture Notes in Computer Science, vol 10618. Springer, Cham (pp. 127-138)."*

### b. Supplemental Dataset

- Link to the dataset: <https://www.kaggle.com/datasets/sameedhayat/guardian-news-dataset>
- Author: **Sameed Hayat**
- We are using v1 of the dataset that was uploaded on 2nd June 2019.

#### **Some basic information regarding the dataset:**

- We use a supplemental dataset because our primary dataset only contains real news stories from a single source (Reuters), which might lead to model overfitting despite our efforts to not do so.
- The dataset was gathered by Sameed Hayat from the news website, The Guardian.
- It contains over 52,000 real news articles on several topics, but the only ones used here are from the politics section (contains approx. 12,650 articles).

---

## **Downloading, cleaning & EDA**

- Downloading, cleaning, and exploratory data analysis of the primary dataset is done in the notebook named '***01 - Kaggle Dataset - DL, Clean & EDA.ipynb***', which can be found in the notebooks folder.
- Downloading and cleaning of the supplemental dataset is done in the notebook named, '***03 - Guardian Dataset - DL & Clean.ipynb***', which is also found in the notebooks folder.

---

## **Text Pre-Processing**

- Text pre-processing of the primary dataset is done in the notebook named '***02 - Kaggle Dataset - Text Preprocessing.ipynb***', which can be found in the notebooks folder.
- Text pre-processing of the supplemental dataset is done in the notebook named, '***03 - Guardian Dataset - DL & Clean.ipynb***', which is also found in the notebooks folder.

## **Selection of the Model**

- It was discovered throughout the EDA process that there is hardly any overlap between false and real news stories when comparing the percentage of capital letters in the story titles.
- On the basis of this distinction, a heuristic model was applied to the data in the notebook ***'04 - Heuristic Model (Percentage of Capital Letters in Title).ipynb'.***
- Despite having a 98% accuracy rate on this dataset, the heuristic looks unlikely to generalize well and may be easily beaten if it were to be deployed as a gatekeeper. This circumstance is comparable to the updates required for a spam filter.

---

- Following this, some machine learning models were utilized in an effort to find a more broadly applicable answer.
- The ***'05 - NB (BOW).ipynb'*** notebook contains multiple iterations of a Naive Bayes Classification model with bag of words (BOW) features, and the ***'06 - RF (BOW).ipynb'*** notebook contains several iterations of a Random Forest Classifier Model with BOW.
- Then, we implemented an extra model utilizing TF-IDF in the ***'07 - RF (TF-IDF).ipynb'*** notebook using the selected and normalized data that we discovered to have worked the best, and it was discovered that this model performed the best up until that time.

---

- The style of fake and true news in the dataset was straightforward to identify even with a simple bag of words model, in contrast to most scenarios where the machine learning models don't have the requisite accuracy, precision, or recall. We tried to generalize the models while maintaining high levels of accuracy, precision, and recall rather than trying to improve the models' accuracy or test different models.

- We were concerned that the model would overfit because the initial dataset only contained "actual" news from one source, Reuters. As a result, we tested the model that was chosen as the best using additional data. It was really disheartening that the outcome, which can be seen in the notebook ***'08 - Classify Story.ipynb',*** were only marginally better than random chance.

- The model was then retrained in the notebook ***'09 - RF (TF-IDF and Supplemental Dataset).ipynb'*** after we added part of this additional data into the training dataset. Though slightly less accurate, it could generalize better.

- The results of this new model's testing on the entire supplemental dataset, which can be seen in the notebook titled ***'10 - Classify Story (with Supplemental Dataset).ipynb',*** were satisfactory enough for us to feel confident in the model.

---

## **Conclusion**

- A random forest classifier that only took words from a stopwords list as input was the model that we discovered to be the most generalizable while still delivering an accuracy of 89% and f1 scores of 0.89 for both false and true results. The persons, locations, organizations, dates, jargon, and other situation-specific references were eliminated by focusing only on the stopwords, making the news more generalizable because the classifications would no longer be reliant on those removed aspects. The notebook ***'09 - RF (TF-IDF with Supplemental Dataset).ipynb'*** contains this model.

- While this model's accuracy and f1 scores are lower than those of the heuristic model, it provides a more generalized answer and is thus anticipated to perform well on news stories that are not included in the dataset. This is due to the fact that news articles shared on our social network may have been written by or shared from sources other than those in this dataset.

## **Future Work**

### **1. Acquire more labeled news stories to improve the model**

The origin of the news items categorized as true came from only two sources, Reuters and Guardian, while the origin of the news stories labelled as false is unknown, despite the fact that this dataset contains almost 35,000 news pieces with a balanced distribution across classes. A more reliable and generalizable model should be produced by including news items from more sources for both the true class and additional news stories that are false. It wasn't done before since gathering news stories and correctly classifying them as true or false requires a lot of time and labour.

### **2. Identify the origin of news stories**

The URL for each new story that is gathered and categorized, as well as the source URL for any existing stories in our dataset, should be collected. Knowing the news story's source ought to be useful information that can be added into the classification process.

### **3. Develop a more sophisticated (deep learning) model to use for fake news detection**

While the model created here did a fair job of categorizing news stories, detection should get harder if NLP advances, like the recent GPT-3.5. For instance, it would be challenging to spot if GPT-3.5 was instructed to create news articles in the manner of, say, a New York Times writer because our model is just analyzing the style of the articles and not their accuracy.

---

## **Deployed Model**

A live version of the model, that was deployed to Streamlit Cloud can be found at the link given below. Feel free to try it out!

[Link to live deployment](https://factguard.streamlit.app/)

---

## **Presentation**

A presentation is also included in the repository for those who are interested in the nitty-gritties. It is a .PDF file with the name ***'FactGuard Technical Presentation'.*** All steps that were involved in the building of this project along with in-depth information is chronicled in the same. Please do check it out. We hope you like our project!

---
