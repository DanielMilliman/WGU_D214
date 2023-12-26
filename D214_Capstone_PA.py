#!/usr/bin/env python
# coding: utf-8

# ## Research Question

# This Western Governors University Capstone Project aims to answer the research question: What role does the number of backers play in the probability of success for a Kickstarter project? In today's economy, the costs associated with a product launch are among the most significant barriers for those looking to enter the entrepreneurial marketplace. The crowdfunding company Kickstarter helps reduce these cost barriers by facilitating a platform for global fundraising that allows creators to secure financial support from backers interested in the proposed projects. If prospective Kickstarter entrepreneurs can gain insights into factors that lead to higher success rates for their product launches, they will have higher chances of success.  
# 
# This Capstone Project aims to perform a logistic regression analysis of historical Kickstarter data using the success rate as the dependent variable and the number of backers as the predictor variable. Using the analysis results, both creators and backers can tailor their Kickstarter strategies to make more informed decisions and achieve a higher likelihood of success.  
# 
# The null hypothesis of the project proposes that the number of backers does not affect a Kickstarter project's success. In other words, the quantity of backers does not play a decisive role in determining the project's outcome and implies that success or failure is independent of the number of backers. Conversely, the alternative hypothesis proposes that the number of backers significantly affects the likelihood of a Kickstarter project's success and suggests a meaningful relationship between the number of backers and the project's success. In summary, the research question explores the dynamics of Kickstarter success rates and is motivated by the practical implications for project stakeholders.  

# ## Data Collection

# The data set utilized for the Capstone project provided a comprehensive view of crowdfunding dynamics and is publicly available on the Kaggle platform. This data set featured 430,949 records, which included a diverse range of historical Kickstarter data, including funding goals, pertinent dates, the number of backers, and project success rates. The data types of the variables varied from categorical, discrete, and continuous, and the research question did not require any other outside data sources.  
# 
# One advantage of using Kickstarter as a data source was the amount of project information available. Kickstarter launched in 2009 and has amassed vast amounts of publicly available project data since then. This transparency enables access to a wide array of historical project information, which researchers can use to perform more extensive analyses of crowdfunding trends. Despite its advantages, one notable disadvantage of available Kickstarter data is the possibility of missing or inaccurate information. Although users can access project information from the Kickstarter platform, the company does not currently offer public API access or a formal library of downloadable data sets. Because of this, much of the available Kickstarter project data was accumulated through web scrapping or other data acquisition techniques performed by third parties. To overcome these challenges, an extensive data cleaning process was implemented to ensure the greatest amount of accuracy for the analysis. This process involved checking for outliers and duplicate values, as well as identifying and addressing entries with missing or unreliable information. The data was also continuously monitored throughout the modeling and analysis process to ensure the utmost accuracy.  
# 
# In summary, the data collection process encompassed leveraging a publicly available Kickstarter data set to answer the research question. After a rigorous data cleaning and validation process was implemented, the logistic regression analysis results were used to gain insights into crowdfunding trends on the Kickstarter platform.  

# ## Data Extraction and Preparation

# The primary programming language used for the data extraction and preparation process was Python. Python is a widely used language popular in data science and supports many different functions of the data analysis process, including machine learning, data visualization, predictive modeling, data mining, and many other tasks. One key area where Python excels is in its robust ecosystem of libraries that facilitate the data extraction and preparation process. Leveraging modules and packages such as NumPy, Pandas, Scikit-learn, or Statsmodels, Python can handle almost any aspect of the data analytics pipeline without requiring manual coding or resorting to other programming languages.  
# 
# However, with its many benefits, there are some drawbacks to utilizing Python for data extraction and preparation. One such drawback is the dependency on community-supported external libraries. Because Python is an open-source programming language, many libraries are generated by third parties. Although this third-party support fosters an environment of collaboration and innovation, it also introduces a level of dependency on the maintenance and support of those libraries. If a critical library becomes outdated or is no longer actively supported, it could impact the functionality of the projects relying on that library. Ways to circumvent this issue are ensuring all libraries are updated or utilizing an alternative library that performs a similar function. For example, Pandas can perform functions similar to NumPy, and TensorFlow can perform tasks similar to Scikit-learn.  

# ## Data Cleaning and Exploration

# Before using Python to perform any additional steps in the analysis process, essential libraries, such as Pandas, NumPy, Scikit-Learn, and other necessary modules and packages, were downloaded and installed within the JupyterLab programming environment. These add-ons are vital in data manipulation, creating and analyzing the logistic regression model, and other crucial machine-learning tasks.  

# In[1]:


# import necessary modules
import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from category_encoders.binary import BinaryEncoder as be
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
import statsmodels.api as sm
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

warnings.filterwarnings("ignore", category=FutureWarning)


# Once the necessary add-ons were imported into the programming environment, the notebook was configured to feature the maximum number of columns within the data frame. This adjustment is made because Pandas typically limits the default number of viewable columns. Upon executing the code snipped below, the data frame's maximum number of columns becomes visible. The expanded view allows for a comprehensive inspection of the data frame and facilitates the identification of accuracy issues or potential data integrity issues.  

# In[2]:


# Set the maximum number of rows and columns to display
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[3]:


# Reset the maximum number of columns to the default value
pd.reset_option('display.max_rows')


# In[4]:


# Read in data set
df = pd.read_csv("/Users/danielmilliman/Desktop/kickstarter.csv")
df.head(1000)


# Upon exploring the overall structure of the data frame, which included examining the number of columns, data types, and index range, the selected features are verified to ensure they possess the correct data types for the regression modeling tasks.

# In[5]:


# View the data shape
df.info()


# The next preprocessing steps included using the code snippets below to check the data frame for duplicate, null, or missing values. By appropriately identifying and handling any null or missing values, potential errors or biases can be avoided during future stages of the modeling and analysis process. If null or missing values are identified, Python's 'drop_duplicates' or 'dropna()' commands can address the inconsistencies.  

# In[6]:


#Check for duplicate values
df.duplicated()


# In[7]:


#Check for duplicate values
df.duplicated().sum()


# In[8]:


# Check for null values
df.isnull().sum()


# Once any duplicate or null values are addressed, the numerical features are inspected for outliers. Outliers can significantly affect statistical measures such as mean, variance, and standard deviation. If present, outliers can lead to inaccuracies in representing the spread of data or compromise the data's validity. Visualizing the numerical variables using a boxplot can identify potential outliers or abnormal values.  

# In[9]:


# Check for outliers
boxplot=sns.boxplot(x='backers_count',data=df)


# Using Seaborn, a count plot visualization for the 'binary_state' variable was also generated. Visualizing the distribution of classes within the target variable is crucial for making informed decisions during both preprocessing and model selection.  

# In[10]:


# Generate Countplot of Success Rate
sns.set_style("whitegrid")
sns.countplot(data=df, x= 'binary_state').set(title='Count of Success Rate')
plt.ylabel('Count')
plt.xlabel('')


# The next step in the preprocessing process includes generating summary statistics for the model's chosen features. Summary statistics give a general statistical overview of the variables and help guide subsequent preprocessing decisions.  This information helps to further identify and address outliers, skewed distributions, or other issues with data validity.  

# In[11]:


#Summary Statistics
df.backers_count.describe()


# In[12]:


#Summary Statistics
df.backers_count.value_counts()


# In[13]:


#Summary Statistics
df.binary_state.describe()


# In[14]:


#Summary Statistics
df.binary_state.value_counts()


# Next, the number of unique values for the 'binary_state' variable is displayed using the "df.unique()" command. This command is helpful for exploring the levels contained within a categorical variable and helps to verify data integrity.  

# In[15]:


# Find number of unique values for categorical values with object data type
df.binary_state.unique()


# Because 'binary_state' isn't necessarily an intuitive name for a column header, it will be renamed to 'success_rate' using the below code snippet. Panda's 'rename' command' allows columns or indexes to be easily changed on a mapping or by providing new names directly.  

# In[16]:


# Rename binary_state column to success_rate
df.rename(columns={'binary_state': 'success_rate'}, inplace=True)
df


# Since logistic regression and many other machine learning tasks use algorithms that only work with numerical inputs, binary encoding was used to change the 'success_rate' variable from categorical to numerical. The binary encoding technique works by representing categorical variables with two distinct levels (usually 0 and 1) by converting them into binary code. For variables with more than two levels, alternative encoding methods are available.  

# In[17]:


# Perform binary encoding on the success_rate variable
df['success_rate'] = df['success_rate'].apply(lambda x: 0 if x == 'failed' else 1)
df


# ## Create the Train and Test Split

# The next preprocessing task is to generate a training and test split. The purpose of creating training and test splits is to divide the data into two subsets—one for training the model and one for testing its performance. Within this process, the training set allows the model to learn patterns and relationships within the data, and the test set serves as an independent benchmark to assess how well the model handles the new, unseen data. The output below features a 70/30 training and test split, which is the industry standard for dividing data.  

# In[18]:


# Features
X = df[['backers_count']]

# Target variable
y = df['success_rate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# As with each of the previous preprocessing steps, it is vital to confirm that the code was executed successfully within the data frame. The code snippet below demonstrates the process of creating a subplot to verify the proper distribution of the training and test set variables.  

# In[19]:


#Generate subplots to view distribution of training and test sets
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
y_train.value_counts().plot(kind='bar', title='Training Set')

plt.subplot(1, 2, 2)
y_test.value_counts().plot(kind='bar', title='Testing Set')

plt.show()


# Printing the value counts of the training and test sets is also an essential preprocessing step. This task provides insights into the distribution of the variable's classes and helps identify potential issues with value counts or data validity. 

# In[20]:


#Print training and test set distributions 
print("Training Set - Distribution of Target Variable:")
print(y_train.value_counts(normalize=True))

print("\nTesting Set - Distribution of Target Variable:")
print(y_test.value_counts(normalize=True))


# Finally, the cleaned data set is saved to the desired path in Python using the 'to_csv' method, where the desired file path and name are specified as an argument.  

# In[21]:


# Save cleaned  as CSV file
df.to_csv(r'/Users/danielmilliman/Desktop/WGU/D214/df_cleaned.csv')  


# ## Create the Logistic Regression Model

# In[22]:


# Initialize the model
model = LogisticRegression()

# Fit the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
print(f'Confusion Matrix:\n{matrix}')


# In[23]:


#Generate logistic regression sumary
log_reg_model = sm.Logit(y_train, sm.add_constant(X_train)).fit()

# Print the summary
print(log_reg_model.summary())


# ## Analysis

# The data for the model was analyzed using the logistic regression summary output and classification report. The logistic regression summary provides a comprehensive and concise overview of the model's performance and the relationships between variables. The summary features outputs such as Interpretability, Variable Significance, Model Fit, Prediction Capability, Confidence Intervals, and Diagnostic Information. These metrics help researchers gain valuable insights into the data they are exploring. However, one disadvantage of using the logistic regression model's summary is the potential of biased or inaccurate data due to correlation among variables. This is because logistic regression assumes that the observations are independent of each other, and if not, it will cause data integrity issues. There are, however, ways to mitigate this issue through techniques such as correlation analysis, VIF, or domain knowledge.  
# 
# The second technique used to analyze the data was by using the classification report. This report displays a detailed evaluation of the model's performance and uses accuracy, precision, recall (sensitivity), F1-score, and confusion matrix outputs. One advantage of the classification report is that it provides a comprehensive evaluation of model performance in an easy-to-interpret format, making it a valuable resource for model analysis. The report has its limitations, though, because it can only perform classification tasks and typically doesn't work for regression or other machine-learning problems.  

# ## Data Summary and Implications

# The analysis results show that the model performed reasonably well and suggests a good fit to the data, which is backed up by the accuracy output of 77%. Furthermore, the precision metric, which measures the accuracy of positive predictions made by the model, accurately predicted unsuccessful projects 74% of the time and successful projects 85% of the time. The F-1 scores demonstrated similar results, with an output of 0.83 for unsuccessful projects and 0.65 for successful projects. Lastly, the model's P-value of 0.000 indicates that there is indeed a statistically significant relationship between the number of backers predicting the success rate.  
# 
# Based on these findings, there are valuable takeaways that both creators and backers can gain from this statistical analysis. For creators, it is recommended that they focus their crowdfunding efforts on achieving as many backers as possible. Creators can use the available Kickstarter data to benchmark their number of backers compared to previous projects. If they are in the lower quartile of the number of backers, they may need to adjust or revise their crowdfunding campaign. For those who may potentially back a Kickstarter campaign, it is recommended that they assess the number of backers before investing in a project. The investment may be deemed riskier if a project is in its early stages and has fewer backers. Conversely, campaigns with a high number of backers may prove to be a sounder investment for the backer and lead to a greater possibility of success for the campaign.   

# ## Resources

# Kerneler. (2019, July 26). Starter: 400,000 Kickstarter Projects a2c233e4-5. Kaggle. https://www.kaggle.com/code/kerneler/starter-400-000-kickstarter-projects-a2c233e4-5/input 
# 
# Capstone Proposal Form Part 2. (n.d.). Panopto. https://wgu.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=88a60be5-7728-46d8-b94f-acbe00051ed3 
# 
# Welcome to Python.org. (2023, November 23). Python.org. https://www.python.org/ 
# 
# Corbo, A. (2023, January 3). How is Python used in data science? Built In. https://builtin.com/data-science/python-data-science 
# 
# GeeksforGeeks. (2023, November 2). Disadvantages of Python. https://www.geeksforgeeks.org/disadvantages-of-python/ 

# In[ ]:




