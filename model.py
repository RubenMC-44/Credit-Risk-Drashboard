import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
df = pd.read_csv('clean_data.csv')

#----------Division of collumns Objetive and labels
X = df.drop(columns=['Unnamed: 0','SeriousDlqin2yrs'])
Y = df['SeriousDlqin2yrs']

x_train , x_test, y_train , y_test = train_test_split(X,Y, train_size= 0.80, random_state=42)
#print(x_train,x_test,y_train,y_test)

#Introducing StandarScaler--- Making out model more accurate
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#After the use- The recall went up to 0,65. Meaning that now we are going to detect even more risks.

#------LogistinRegression model-------------
model = LogisticRegression(class_weight='balanced',max_iter=1000) #We have and unbalanced data frame to much 0, so we have a lazy model who let pass the risk like nothing 
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
metrics_simple = {
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}
#----------------

#lets use a bigger model to try how the data frame response

#------Random forest RandomForestClassifier-------------
model_forest= RandomForestClassifier(class_weight="balanced")
model_forest.fit(x_train,y_train)

y_pred_forest = model_forest.predict(x_test)
print(classification_report(y_test, y_pred_forest))
#I am decide that use regression model is better, due to recall values, because is a better metric to work on this project
#although Random forest give better precisión with the risk, but we can not let so many risk alarms.  


# We use Model, beeing our train model, to keep it on the analysis. 
# WE save the model
joblib.dump(model, 'model.pkl')

# Charge the model
load_model = joblib.load('model.pkl')#The true model that we use to train the model.
joblib.dump(metrics_simple, 'metrics_simple.pkl') #Used to show the metric on the drashboard as an easy way

#Charging StandartScaler model 
joblib.dump(scaler, 'scaler.pkl')