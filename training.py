# Love Saroha
# lovesaroha1994@gmail.com (email address)
# https://www.lovesaroha.com (website)
# https://github.com/lovesaroha  (github)

# Predicting salary using stack overflow developer survey data.
import pandas 
import numpy 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Data url (https://insights.stackoverflow.com/survey).

# Load data.
dataFrame =  pandas.read_csv("C:/Users/loves/Documents/survey_results_public.csv")

# Select columns.
dataFrame =  dataFrame[["Country", "EdLevel", "YearsCodePro", "Employment", "ConvertedComp"]]
dataFrame =  dataFrame.rename({"ConvertedComp": "Salary"}, axis=1)

# Select where salary is not null.
dataFrame =  dataFrame[dataFrame["Salary"].notnull()]
dataFrame =  dataFrame.dropna()
dataFrame.isnull().sum()

# Select with full employment and remove column.
dataFrame =  dataFrame[dataFrame["Employment"] == "Employed full-time"]
dataFrame =  dataFrame.drop("Employment", axis=1)

# Save only those country data where value is more than minimum.
def filterCountries(data, minimum):
    cmap = {}
    for i in range(len(data)):
        if data.values[i] >= minimum:
            cmap[data.index[i]] = data.index[i]
    return cmap

dataFrame['Country'] = dataFrame['Country'].map(filterCountries(dataFrame.Country.value_counts(), 500))
print(dataFrame['Country'].value_counts())

# Take salary data between 10000 and 300000.
dataFrame =  dataFrame[dataFrame["Salary"] <= 250000]
dataFrame =  dataFrame[dataFrame["Salary"] >= 10000]

# Change string values in experience column to float.
def toFloat(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)
dataFrame['YearsCodePro'] = dataFrame['YearsCodePro'].apply(toFloat)

# Change education values.
def setEducationValue(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelors'

dataFrame['EdLevel'] = dataFrame['EdLevel'].apply(setEducationValue)

# Change education values to int.
educationEncoder = LabelEncoder()
dataFrame['EdLevel'] = educationEncoder.fit_transform(dataFrame['EdLevel'])

# Change country values to int.
countryEncoder = LabelEncoder()
dataFrame['Country'] = countryEncoder.fit_transform(dataFrame['Country'])

# Set input and output.
inputdata = dataFrame.drop("Salary", axis=1)
output = dataFrame["Salary"]

# Train linear regression model.
linearRegressionModel = LinearRegression()
linearRegressionModel.fit(inputdata, output.values)

# Predict values.
prediction = linearRegressionModel.predict(inputdata)

# Show mean squared error.
print("Linear regression model error: ",numpy.sqrt(mean_squared_error(output, prediction)))

# Train decision tree regressor model.
decisionTreeRegressorModel = DecisionTreeRegressor(random_state=0)
decisionTreeRegressorModel.fit(inputdata, output.values)

# Predict values.
prediction = decisionTreeRegressorModel.predict(inputdata)

# Show mean squared error.
print("Decision tree regressor model error: ",numpy.sqrt(mean_squared_error(output, prediction)))

# Train random forest regressor model.
randomForestRegressorModel = RandomForestRegressor(random_state=0)
randomForestRegressorModel.fit(inputdata, output.values)

# Predict values.
prediction = randomForestRegressorModel.predict(inputdata)

# Show mean squared error.
print("Random forest regressor model error: ",numpy.sqrt(mean_squared_error(output, prediction)))

# Grid search cross validation.
regressor = DecisionTreeRegressor(random_state=0)
gs = GridSearchCV(regressor, {"max_depth": [None, 2,4,6,8,10,12]}, scoring='neg_mean_squared_error')
gs.fit(inputdata, output.values)
regressor = gs.best_estimator_
regressor.fit(inputdata, output.values)
# Predict values.
prediction = regressor.predict(inputdata)

# Show mean squared error.
print("Grid search CV model error: ",numpy.sqrt(mean_squared_error(output, prediction)))

# Predict on a given data.
country = "India"
education = "Bachelor’s degree"
experience = 10

# Convert values using encoder.
country = countryEncoder.transform([country])
education = educationEncoder.transform([education])
inputdata = numpy.array([[country, education, 10 ]])
inputdata = inputdata.astype(float)

prediction = regressor.predict(inputdata)
print("Estimated salary: ", prediction[0])