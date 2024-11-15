'''CRISP-ML(Q)

a. Business & Data Understanding
   Machines which manufacturs the pumps. Unplanned machine downtime which is leading to loss of productivity
   
    i. Business Objective - Minimize the Unplanned machine Downtime
    ii. Business Constraint - Minimize maintenance cost;Maximize equipment efficiency

    Success Criteria:
    1. Business Success Criteria - Reduce the unplanned downtime by atleast 10%
    2. ML Success Criteria - Achieve an accuracy of atleast 96%
    3. Economic Success Criteria - Achieve a cost saving of at least $1M
    
    Data Collection -   Data collection may collected from IOT sensors, historical data etc.... to get the fuel pump machine Data has 2500 observations and 16 columns. 
    
    Metadata Description:
    Downtime = Type - this is output variable and has '2' classes - Machine failure & no machine failure
    '''
    # Load the requried Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
from sklearn.model_selection import train_test_split
# Hyperparameter optimization
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV



# load the dataset
machine_downtime = pd.read_csv(r"E:\project.1@360digiTMG\Data Set\Machine Downtime.csv")

# first 5 rows viewing
machine_downtime.head()

# viewing the columns
machine_downtime.columns  # 16 columns

######## EDA(Exploratory Data Analysis) ##########
# Describing the summary statistics( count,mean, median, q1,q3,min,max,std)
machine_downtime.describe()
machine_downtime.iloc[:, 3:-1].skew()  # all values in range of(-0.5 to 0.5), so all variables are normally distributed
machine_downtime.iloc[:, 3:-1].kurt()  # all values are less than 3 ,i.e Leptokurtic 
# Info and shape of a dataset
machine_downtime.info()  # 2500 entries and 16 columns, dtypes: float64(12), object(4)

######### Duplicate Values ##########
#checking the Duplicates
sum(machine_downtime.duplicated())  # No duplicate values

############ Missing Values ###########
# checking the Null values
machine_downtime.isnull().sum()   # There are null values

# dropping the 'date' columns which is interval data
machine_downtime.drop(['Date'],axis = 1, inplace  = True)

#Data split into Input and Output data
X = machine_downtime.iloc[:, :-1] #input data
Y = machine_downtime.iloc[:, -1]  #output data(Target variable)


### Handling the missing values , since i have outliers ,im going with median imputation, which less effected by extrem values
# Median Imputer

num_col = X.select_dtypes(exclude = ['object']).columns
cat_col = X.iloc[:, :-1].select_dtypes(include = ['object']).columns

# ### StandardScaler to convert the distribution of the data with mean = 0 and stdv = 1
num_pipeline = Pipeline(steps = [('impute', SimpleImputer(strategy = 'median')),
                                 ('scale', StandardScaler())])


# ### Encoding - One Hot Encoder to convert Categorical data to Numeric values

## Encoding
# Categorical features
encoding_pipeline = Pipeline([('label', OrdinalEncoder())])

# Creating a transformation of variable with ColumnTransformer()
preprocessor = ColumnTransformer(transformers = [('num', num_pipeline, num_col),
                                                 ('categorical', encoding_pipeline, cat_col)])

imp_enc_scale = preprocessor.fit(X)
X.isnull().sum()
# #### Save the imputation model using joblib
joblib.dump(imp_enc_scale, 'imp_enc_scale')

import os
os.getcwd()

cleandata = pd.DataFrame(imp_enc_scale.transform(X), 
                         columns = imp_enc_scale.get_feature_names_out())
cleandata

# if you get any error then update the scikit-learn library version & restart the kernel to fix it

# ### Outlier Analysis

# Multiple boxplots in a single visualization.
# Columns with larger scales affect other columns. 
# Below code ensures each column gets its own y-axis.

# pandas plot() function with parameters kind = 'box' and subplots = True

cleandata.iloc[:, 0:-2].plot(kind = 'box', subplots = True, sharey = False, figsize = (10, 6)) 
'''sharey True or 'all': x- or y-axis will be shared among all subplots.
False or 'none': each subplot x- or y-axis will be independent.'''
# Increase spacing between subplots
plt.subplots_adjust(wspace = 0.75) # ws is the width of the padding between subplots, as a fraction of the average Axes width.
plt.show()


winsor = Winsorizer(capping_method = 'iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail = 'both', # cap left, right or both tails 
                          fold = 1.5,
                          variables = list(cleandata.iloc[:, 0:-2].columns))

outlier = winsor.fit(cleandata[list(cleandata.iloc[:, 0:-2].columns)])

# Save the winsorizer model 
joblib.dump(outlier, 'winsor')

cleandata[list(cleandata.iloc[:,0:-2].columns)] = outlier.transform(cleandata[list(cleandata.iloc[:,0:-2].columns)])

########### Auto EDA ##############
########
# D-Tale
# pip install dtale
import dtale
d = dtale.show(machine_downtime)
d.open_browser()

# AutoEDA
import sweetviz
my_report = sweetviz.analyze(machine_downtime)
my_report.show_html('report.html')
#########

machine_downtime['Assembly_Line_No'].value_counts()

machine_downtime['Downtime'].value_counts()

machine_downtime['Machine_ID'].value_counts()

########## Zero Variance ###########
# Checking  is any variance is zero in row
machine_downtime.iloc[:, 3:-1].var() == 0  # NO variable has zero variance


# Split data into train and test with Stratified sample technique
# from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(cleandata , Y, test_size = 0.3, stratify = Y, random_state = 42)

''' Ready to go with Machine learning Model building '''
#### Linear model Support vector machine ########
from sklearn.svm import SVC
model1 = SVC(kernel="linear",max_iter=100000, gamma = 'auto')

model1.fit(X_train,Y_train)
test_pred = model1.predict(X_test)

linear_accuracy = np.mean(Y_test.values.flatten() == test_pred.values.flatten())
linear_accuracy  # --- 0.857% Accuracy

#### rgf Model
model2 = SVC(kernel="rbf",max_iter=150000)
model2.fit(X_train,Y_train)
rbf_pred=model2.predict(X_test)

rbf_accuracy = np.mean(Y_test.values.flatten() == rbf_pred)
rbf_accuracy  ## ---0.878% Accuracy

#### Poly Model
model3 = SVC(kernel="poly",max_iter=100000)
model3.fit(X_train,Y_train)
poly_pred = model3.predict(X_test)

poly_accuracy = np.mean(Y_test.values.flatten() == poly_pred)
poly_accuracy  ## --- 0.88% Accuracy

#### Sigmoid Model
model4 = SVC(kernel="sigmoid",max_iter=100000)
model4.fit(X_train,Y_train)
sigmoid_pred=model4.predict(X_test)

sig_accuracy = np.mean(Y_test.values.flatten() == sigmoid_pred)
sig_accuracy ## --- 0.80% Accuracy

 # cobimation of all models with accuracy
results = pd.DataFrame({"linear_model": linear_accuracy,"rbf_model": rbf_accuracy,"poly_accuracy":poly_accuracy,"sigmoid_accuracy":sig_accuracy},index=["Accuracy"])
results  ## -- Above all models RBF model giving the high Accuracy with 87% which is not enough for our Requirment.

######### Logistic regression ##########  --- Right fit with 85% accuracy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = LogisticRegression(random_state=42)  ## --- 85% Accuracy,Precisin,Recall,F1-score.

model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)

conf_matrix = confusion_matrix(Y_test, y_pred)
class_report = classification_report(Y_test, y_pred)

print("Accuracy:", accuracy)  ## --- 85% Accuracy,Precisin,Recall,F1-score.
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
#training accuracy
y_pred_train = model.predict(X_train)
accuracy_train = accuracy_score(Y_train, y_pred_train)

conf_matrix_t = confusion_matrix(Y_train, y_pred_train)
class_report_t = classification_report(Y_train, y_pred_train)


print("Accuracy:", accuracy_train)  ## --- 86% Accuracy,Precisin,Recall,F1-score.
print("Confusion Matrix:\n", conf_matrix_t)
print("Classification Report:\n", class_report_t)

#### GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


boost_clf = GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 1000, max_depth = 1)

boost_clf.fit(X_train, Y_train)


grad_pred = boost_clf.predict(X_test)
GB_report = classification_report(Y_test, grad_pred)

## Test scaore ----- 99% Accuracy,Precisin,Recall,F1-score
print(confusion_matrix(Y_test, grad_pred))
print(accuracy_score(Y_test, grad_pred))
print("Classification Report:\n", GB_report)

## Training score ----- 99% Accuracy,Precisin,Recall,F1-score
GB_report_train = classification_report(Y_train, boost_clf.predict(X_train))
print(confusion_matrix(Y_train, boost_clf.predict(X_train)))
print(accuracy_score(Y_train, boost_clf.predict(X_train)))
print("Classification Report:\n", GB_report_train)

pickle.dump(boost_clf, open('Gradiantboosting.pkl', 'wb'))


from sklearn.naive_bayes import GaussianNB
# Create a Categorical Naive Bayes model
clf = GaussianNB()

# Fit the model on the training data
clf.fit(X_train, Y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)
# Evaluate the accuracy
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy}")  ## --85% Accuracy

# Make predictions on the train data
y_pred_t = clf.predict(X_train)
# Evaluate the accuracy
accuracy = accuracy_score(Y_train, y_pred_t)
print(f"Accuracy: {accuracy}")  ## --85% Accuracy

from sklearn.model_selection import cross_validate

def cross_validation(model, _X, _y, _cv=5):
    
    '''Function to perform 5 Folds Cross-Validation
    Parameters
    ----------
    model: Python Class, default=None
          This is the machine learning algorithm to be used for training.
    _X: array
       This is the matrix of features.
    _y: array
       This is the target variable.
    _cv: int, default=5
      Determines the number of folds for cross-validation.
    Returns
    -------
    The function returns a dictionary containing the metrics 'accuracy', 'precision',
    'recall', 'f1' for both training set and validation set.
    '''
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                           X=_X,
                           y=_y,
                           cv=_cv,
                           scoring=_scoring,
                           return_train_score=True)

    return pd.DataFrame({"Training Accuracy scores": results['train_accuracy'],
          "Mean Training Accuracy": results['train_accuracy'].mean()*100,
          "Training Precision scores": results['train_precision'],
          "Mean Training Precision": results['train_precision'].mean(),
          "Training Recall scores": results['train_recall'],
          "Mean Training Recall": results['train_recall'].mean(),
          "Training F1 scores": results['train_f1'],
          "Mean Training F1 Score": results['train_f1'].mean(),
          "Validation Accuracy scores": results['test_accuracy'],
          "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
          "Validation Precision scores": results['test_precision'],
          "Mean Validation Precision": results['test_precision'].mean(),
          "Validation Recall scores": results['test_recall'],
          "Mean Validation Recall": results['test_recall'].mean(),
          "Validation F1 scores": results['test_f1'],
          "Mean Validation F1 Score": results['test_f1'].mean()
          })


Gboost_cv_scores = cross_validation(boost_clf, X_train, Y_train, 5)
Gboost_cv_scores

########## To see the train And validation data Performance #######
def plot_result(x_label, y_label, plot_title, train_data, val_data):
        # Set size of plot
        plt.figure(figsize=(12,6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        #ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='blue', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='red', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.show()


model_name = "Gradiant Boosting Classifier"

plot_result(model_name,
            "Accuracy",
            "Accuracy scores in 5 Folds",
            Gboost_cv_scores["Training Accuracy scores"],
            Gboost_cv_scores["Validation Accuracy scores"])

############  KNN #############
from sklearn.neighbors import KNeighborsClassifier
# Create a KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)  # You can adjust the number of neighbors (k) as needed
# Train the classifier
knn_classifier.fit(X_train, Y_train)
y_pred = knn_classifier.predict(X_test)
# Evaluate the model
knn_accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", knn_accuracy) #--86 % Accuracy
# Make predictions on the test set
y_pred = knn_classifier.predict(X_train)
# Evaluate the model
accuracy = accuracy_score(Y_train, y_pred)
print("Accuracy:", accuracy)  # -- 93%% Accuracy which is overfitted

################ Decision Tree ####################
from sklearn.tree import DecisionTreeClassifier as DT

### Decision Tree Model
model = DT(criterion = 'entropy')
model.fit(X_train, Y_train)

# Prediction on Test Data
preds = model.predict(X_test)
preds
# Prediction on Train Data
preds_train = model.predict(X_train)
preds_train

# Accuracy
print(accuracy_score(Y_test, preds)) # 0.98% accuracy
print(accuracy_score(Y_train, preds_train)) # 100% accuracy

pd.crosstab(Y_test, preds, rownames = ['Actual'], colnames = ['Predictions']) 


### Hyperparameter Optimization
# create a dictionary of all hyperparameters to be experimented
param_grid = { 'criterion':['gini', 'entropy'], 'max_depth': np.arange(3, 15)}

# Decision tree model
dtree_model = DT()

# GridsearchCV with cross-validation to perform experiments with parameters set
dtree_gscv = GridSearchCV(dtree_model, param_grid, cv = 5, scoring = 'accuracy', return_train_score = False, verbose = 1)


# Train
dtree_gscv.fit(X_train, Y_train)

# The best set of parameter values
dtree_gscv.best_params_


# Model with best parameter values
DT_best = dtree_gscv.best_estimator_
DT_best

# Prediction on Test Data
preds1 = DT_best.predict(X_test)
preds1
# Prediction on Training Data
preds1_train = DT_best.predict(X_train)
preds1_train

pd.crosstab(Y_test, preds, rownames = ['Actual'], colnames= ['Predictions']) 
# Accuracy
DT_accuracy = print(accuracy_score(Y_test, preds1))  ## -- 0.98%

DT_accuracy_train = print(accuracy_score(Y_train, preds_train)) ## --100%


### Random Forest Model -- over fit with train 100% and test 98%
from sklearn.ensemble import RandomForestClassifier

rf_Model = RandomForestClassifier()
rf1_model = rf_Model.fit(X_train, Y_train)
rf_pred = rf1_model.predict(X_train)

# Evaluating on training data
print(confusion_matrix(Y_train, rf_pred)) ##-- 100% accuracy
RF_accuracy =  print(accuracy_score(Y_train, rf_pred))
rf_pred = rf1_model.predict(X_test)

# Evaluating on training data
print(confusion_matrix(Y_test, rf_pred))  ## -- 98%
RF_accuracy_train = print(accuracy_score(Y_test, rf_pred))


# Define a list of models
models = [
    ('RandomForest', RandomForestClassifier()),
    ('GradientBoosting', GradientBoostingClassifier(learning_rate = 0.02, n_estimators = 1000, max_depth = 1)),
    ('Decision Tree', DT(criterion = 'entropy')),
    ('SVM', SVC(kernel="rbf",max_iter=150000)),
    ('LogisticRegression', LogisticRegression(random_state=42)),
    ('KNN',KNeighborsClassifier(n_neighbors=3)),
    ('GaussianNB',GaussianNB())
]

# Create an empty DataFrame to store results
results_df = pd.DataFrame(columns=['Model', 'Train Accuracy', 'Test Accuracy'])

# Iterate through the models
for model_name, model in models:
    # Train the model
    model.fit(X_train, Y_train)
    
    # Predict on train and test sets
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)
    
    # Calculate accuracy scores
    train_accuracy = accuracy_score(Y_train, Y_train_pred)
    test_accuracy = accuracy_score(Y_test, Y_test_pred)
    
    results_df = pd.concat([results_df, pd.DataFrame({'Model': [model_name], 'Train Accuracy': [train_accuracy], 'Test Accuracy': [test_accuracy]})], ignore_index=True)

# Display the results DataFrame
print(results_df)

# Note -- By inspecting the all models of Train Accuracy and Test Accuracy----- GradiantBoosting was built with 98% Accuracy in both (Train & Test-- Right fit).
 





