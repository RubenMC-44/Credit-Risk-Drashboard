import pandas as pd

##READING Data Frame
df = pd.read_csv('data.csv')

##Checking information about data frame and what we need to change
#------------df.info() #To Check the number of collummns and number of lines
#------------print(df["MonthlyIncome"].describe())#Information of the data 

##Difference between the risk and no risk
print(df['SeriousDlqin2yrs'].value_counts()) 

#----Removing nulls from the MONTHLY INCOME collumn-------------
##Chenking difference between mean and median
    #------------print(df['MonthlyIncome'].mean()) #Getting extreme values 
    #------------print(df['MonthlyIncome'].median())#Correct values that we can use to replace nulls 

df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df['MonthlyIncome'].median())#Replacement of nulls for median of monthly income. 
    #------------print(df["MonthlyIncome"].describe())#Information of the data 


#----Removing nulls from the NumberOfDependents collumn-------------
##Chenking difference between mean and median
    #------------print(df['NumberOfDependents'].mean()) #Getting extreme values 
    #------------print(df['NumberOfDependents'].median()) #The values es 0, so not a valid value
df["NumberOfDependents"] = df["NumberOfDependents"].fillna(df['NumberOfDependents'].mean())#Replacement of nulls for mean of NumberOfDependets. 
    #------------print(df["NumberOfDependents"].describe())#Information of the data 

print(df.isnull().sum()) #QUICK CHECK IF ALL THE NULLS ARE GONE

#-------------Saving clean Data Frame-------------
df.to_csv('clean_data.csv',index=False)


