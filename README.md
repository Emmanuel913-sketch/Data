# Project Business Bank Churners - Data

**Project Business Bank Churners - Data** They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction.

# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)


## Dataset Content
* The data is a bank Churners data that containd different ages of customers for the bank - The bank is disturbed with more and more customers leaving their credit card services. This dataset consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category. /workspace/Data/BankChurners.csv

* [alt text](image.png)


## Business Requirements
* A manager at the bank is disturbed with more and more customers leaving their credit card services. The manager wants to see a correlation between the customer ages and the number of customers leaving the credit card services. This correlation is needed for business planning, prediction and better decision making.


## Hypothesis and how to validate?
* Bank needs to have some insights on who is leaving their credit card service
* Data was retrieved and analysed
* No missing data was identified from the set
* The correlation between the customer ages and customers retained and those attrited
* The data will be validated using scatter plots to show the correlation
## Project Plan
* Down load data from Kaggle
* Save it in my local drive
* Open NoteBook in GitPod
* Write code to load and retrieve the data set
* Update the project progress on Kanban
* Update ReadMe file

High-level steps taken for the analysis.
* Using Isnull to check for any missing data
* Using scatter plot for correlation

Data management throughout the collection, processing, analysis and interpretation steps
* Download data from Kaggle
* Save it in my local drive
* Open NoteBook in GitPod to load data
* Write code to load and retrieve the data set
* Missing data search using Isnull
* No missing data was found
* A Correlation between Customer Age and Attrition Flag plotted
* Correlation found between Customer Age and Attrition Flag plotted - Customers yonger than the age of 51 years were more attricted ones and there are more existing customers from the age of 40 - 62 years old.

Why the research methodologies you used.
* With interest in business running and banking and credit agencies, I chose the BankChurners CSV file.
* With not much to deduce from the data set except for correlations between a few columns, only a Correlation was fit to be carried out to      
  understand the demographics of the Attrited & Existing customers vs Age. 
* The Scatter plot used as it is ideal for numerical correlations

## The rationale to map the business requirements to the Data Visualisations
*  With not much to deduce from the data set except for correlations between a few columns, only a Correlation scatter plot was fit to be carried out to understand the demographics of the Attrited & Existing customers vs Age of the customers. 
* The Scatter plot used as it is ideal for numerical correlations and it showed that there is a correlation between ages and the leaving customers and the staying ones.

## Analysis techniques used
* Read.csv
*Pandas
* Dataframe
* Query
* x, y scatter plot
* Seaborn plotting
* Plotly 

## Ethical considerations
* I made sure that the data is kept secure in a secure folder in the local drive.

## Unfixed Bugs
* No unfix bugs.

## Development Roadmap
* Setting up of Gitpod and Notebook in Gitpod
* Loading data into the Note book
* Analysing data
* Check for missing data
* Ploting the data graphically
* Drawing conclusions from the correlated columns


## Main Data Analysis Libraries
*  import os
current_dir = os.getcwd()
current_dir

os.chdir(os.path.dirname(current_dir))
print("/workspace/Data/BankChurners.csv")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

df = pd.read_csv('BankChurners.csv')

from sklearn.pipeline import Pipeline

filtered_df = df.query('Customer_Age > 20 & Customer_Age < 60 & Credit_Limit > 8500')
print(filtered_df.shape)  # Number of rows after filtering
print(filtered_df.head())  # Check the first few rows

from feature_engine.imputation import MeanMedianImputer
imputer = MeanMedianImputer(imputation_method='median',
                            variables=['Customer_Age' , 'UnitPrice'])
                            

fig, axes = plt.subplots(figsize=(8,8))

sns.scatterplot(data=df, x='Customer_Age', y='Attrition_Flag',)   # Seaborn code to draw a scatter plot
plt.title("Seaborn Plot!!!")
plt.xlabel('X-Axis: age ')
plt.legend(loc='upper left', title='Legend', frameon=False)
plt.show()   

import plotly.express as px

## Credits 

* Ihttps://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/data

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)


## Acknowledgements (optional)
* Thank you to Vasi and Neil and the classmates for providing the support in the project and learning.
