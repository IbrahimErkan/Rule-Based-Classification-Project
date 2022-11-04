# Lead Calculation with Rule-Based Classification

# BUSINESS PROBLEM:
# A game company wants to create new customer definitions based on level by using some features of its customers,
# create segments according to these new customer definitions and estimate how much the new customers can earn on average according to these segments.

# For example:
# It is desired to determine how much a 25-year-old male user from Turkey, who is an IOS user, can earn on average.


# Dataset Story:
# The Persona.csv dataset contains the prices of the products sold by an international game company and some demographic information of the users who buy these products.
# The data set consists of records created in each sales transaction. This means that the table is not deduplicated.
# In other words, a user with certain demographic characteristics may have made more than one purchase.

# Features:
# PRICE - Customer spend amount
# SOURCE - The type of device the customer is connecting to
# SEX - Customer's gender
# COUNTRY - Customer's country
# AGE - Customer's age



import numpy as np
import pandas as pd

dataframe = pd.read_csv("persona.csv")
df = dataframe.copy()

def check_df(dataframe, head=5, tail=5):
    print("################### Shape ###################")
    print(dataframe.shape)
    print("################### Types ###################")
    print(dataframe.dtypes)
    print("################### Head ###################")
    print(dataframe.head(head))
    print("################### Tail ###################")
    print(dataframe.tail(tail))
    print("################### NA ###################")
    print(dataframe.isnull().sum())
    print("################### Quantiles ###################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("################### Features Names ###################")
    print(dataframe.columns)

check_df(dataframe)


# Task1:
df["SOURCE"].value_counts()

df["PRICE"].unique()
df["PRICE"].nunique()

df["PRICE"].value_counts()
df["PRICE"].value_counts().sort_index()

df["COUNTRY"].value_counts()

df.groupby("COUNTRY")["PRICE"].agg("sum")

df.groupby("SOURCE")["PRICE"].agg("mean")

df.groupby(["COUNTRY", "SOURCE"])["PRICE"].agg("mean")


# TASK2:

# What are the average earnings in the breakdown of COUNTRY, SOURCE, SEX, AGE?
df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE" : "mean"}).round(2)


# TASK3:
# To better see the output in the previous question, apply the sort_values method according to PRICE in descending order.
# Save the output as agg_df.
agg_df =  df.groupby(["COUNTRY", "SOURCE", "SEX", "AGE"]).agg({"PRICE" : "mean"}).sort_values("PRICE", ascending = False)


# TASK4:  Convert the names in the index to variable names.
# All variables except PRICE in the output of the third question are index names. Convert these names to variable names.

agg_df = agg_df.reset_index()


# TASK5:
# Convert age variable to categorical variable and add it to agg_df.
# Convert the numeric variable age to a categorical variable.
# Construct the intervals convincingly.
# For example: '0_18', '19_23', '24_30', '31_40', '41_70'

# 1st way:
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], [0,18,23,30,40,66], labels=["0_18","19_23","24_30","31_40","41_66"])

# 2nd way:
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], [0,19,24,31,41,70], right=False)
agg_df["AGE_CAT"] = agg_df["AGE_CAT"].apply(lambda x: str(x.left)+"_"+str(x.right-1) if x.right != 70 else str(x.left)+"_"+str(agg_df["AGE"].max()))


# Task6:
# Define new level-based customers (personas) and add them as variables to the dataset. Name of new variable to add: customers_level_based
# You need to create the customers_level_based variable by combining the observations from the output from the previous question.
agg_df["customers_level_based"] = [(aggdfcol[0]+"_"+aggdfcol[1]+"_"+ aggdfcol[2]+"_"+aggdfcol[5]).upper() for aggdfcol in agg_df.values]
agg_dff = agg_df[["customers_level_based","PRICE"]].groupby("customers_level_based")["PRICE"].agg("mean").sort_values(ascending=False).reset_index()


# Task7:
# Segment new customers (personas). Divide new customers (Example: USA_ANDROID_MALE_0_18) into 4 segments according to PRICE.
# Add the segments to agg_df as a variable with the SEGMENT naming. Describe the segments (Group by segments and get the price mean, max, sum).
# Analyze the C segment (extract only the C segment from the dataset and analyze).
agg_dff["SEGMENT"] = pd.qcut(agg_dff["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_dff.groupby("SEGMENT")["PRICE"].agg(["mean", "max", "sum"])
agg_dff.SEGMENT.value_counts()
agg_dff[agg_dff["SEGMENT"]=="C"].describe()   #32.254 ile 34.072 arasında price değerlerine sahip
agg_dff[agg_dff["SEGMENT"]=="C"].sort_values("customers_level_based")


# Task8:
# Classify new customers by segment and estimate how much revenue they can generate.
# Which segment does a 33-year-old Turkish woman using ANDROID belong to and how much income is expected to earn on average?

new_user = "TUR_ANDROID_FEMALE_31_40"
agg_dff[agg_dff["customers_level_based"]==new_user]

# In which segment and on average how much income would a 35 year old French woman using iOS expect to earn?
new_user = "FRA_IOS_FEMALE_31_40"
agg_dff[agg_dff["customers_level_based"]==new_user]

