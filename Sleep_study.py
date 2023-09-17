#!/usr/bin/env python
# coding: utf-8

# # The Effects of Caffeine, Alcohol, and Exercise on Quality Sleep

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("C:/Users/Sankung/Downloads/Sleep_Efficiency.csv")


# In[3]:


df.head()


# In[4]:


df['Bedtime'] = pd.to_datetime(df['Bedtime'])
df['Wakeup time'] = pd.to_datetime(df['Wakeup time'])


# In[5]:


df.isnull().sum()


# In[6]:



# Perform one-hot encoding for "Smoking status"
smoking_status_encoded = pd.get_dummies(df['Smoking status'], prefix='Smoking')

# Concatenate the encoded columns with the original dataframe
df = pd.concat([df, smoking_status_encoded], axis=1)

# Remove the original "Smoking status" column
df.drop('Smoking status', axis=1, inplace=True)

# Print the encoded data
df.head()


# In[7]:


import pandas as pd
from sklearn.impute import KNNImputer

# Define the variables with missing values
variables_with_missing = ['Awakenings', 'Caffeine consumption', 'Alcohol consumption', 'Exercise frequency']

# Perform KNN imputation
imputer = KNNImputer(n_neighbors=5) 
imputed_df = imputer.fit_transform(df[variables_with_missing])

# Replace missing values with imputed values
df[variables_with_missing] = imputed_df

# Print the imputed data
df.head()


# In[8]:


df.isnull().sum()


# In[10]:


sns.pairplot(df,hue='Sleep efficiency',palette='Blues')


# In[11]:


# Calculate average deep sleep and light sleep percentages for men and women
gender_avg = df.groupby('Gender')['Deep sleep percentage', 'Light sleep percentage'].mean()

# Calculate average deep sleep and light sleep percentages for smokers and non-smokers
smoking_avg = df.groupby('Smoking_Yes')['Deep sleep percentage', 'Light sleep percentage'].mean()

# Display the results
print("Average deep sleep and light sleep percentages by gender:")
print(gender_avg)

print("\nAverage deep sleep and light sleep percentages by smoking status:")
print(smoking_avg)


# The data provided reveals some differences in average deep sleep and light sleep percentages based on gender and smoking status.
# 
# In terms of gender, the analysis shows that females had a slightly lower average deep sleep percentage compared to males (51.625% vs. 54.000%). Conversely, females had a slightly higher average light sleep percentage compared to males (25.1875% vs. 23.947368%). These findings suggest that there may be subtle variations in sleep patterns between genders, with males tending to have slightly more deep sleep and females experiencing slightly more light sleep on average.
# 
# Regarding smoking status, the data indicates that non-smokers (Smoking_Yes = 0) had a higher average deep sleep percentage compared to smokers (55.372483% vs. 47.88961%). On the other hand, smokers had a higher average light sleep percentage compared to non-smokers (29.337662% vs. 22.09396%). This suggests that smoking may have some influence on sleep patterns, with non-smokers tending to have more deep sleep and smokers exhibiting more light sleep on average. It's important to note that these observations are based on the provided data and further research is needed to establish a more comprehensive understanding of the relationship between smoking and sleep.
# 

# In[12]:


# Plot bar chart for average deep sleep and light sleep percentages by gender
gender_avg.plot(kind='bar', rot=0)
plt.title('Average Deep Sleep and Light Sleep Percentages by Gender')
plt.xlabel('Gender')
plt.ylabel('Percentage')
plt.legend()
plt.show()


# In[16]:


# Plot pie chart for average deep sleep and light sleep percentages by smoking status
fig, ax = plt.subplots()
ax.pie(smoking_deep_sleep, labels=smoking_labels, autopct='%1.1f%%', startangle=90, shadow=True)
ax.set_title('Average Deep Sleep Percentage by Smoking Status')
plt.show()

fig, ax = plt.subplots()
ax.pie(smoking_light_sleep, labels=smoking_labels, autopct='%1.1f%%', startangle=90, shadow=True)
ax.set_title('Average Light Sleep Percentage by Smoking Status')
plt.show()


# Age distribution 

# In[27]:


sns.kdeplot(data=df, x="Age",color="black",fill=True)
plt.xlabel("Age", color="black", fontsize=10)
plt.ylabel("count", color="black", fontsize=10)
plt.title("Age kdeplot", color="black",fontsize=10)
plt.show()


# Age is normally distributed with mean 40.

# In[30]:


sns.relplot(
    data=df, kind="line",
    x="Age", y="Sleep efficiency", style="Gender", color="black"
)
plt.show()


# Sleep efficiency tends to improve from childhood through adolescence, reaching its peak in early adulthood around the age of 19 or 20. However, after this peak, sleep efficiency gradually declines with age. Fluctuations in sleep efficiency may occur during adulthood, but it generally decreases over time. Individuals in their forties have been found to exhibit lower sleep efficiency, and females around the age of 57 have also reported decreased sleep efficiency. These trends align with observations from previous studies on sleep patterns.

# In[2]:


# Let's see if there is imbalance in the data


# In[32]:


df.Gender.value_counts()


# This dataset is well-balanced as it consists of 228 males and 224 females, ensuring a nearly equal representation of both genders.

# In[34]:


sns.countplot(data=df,x="Gender", color="yellow")
plt.xlabel("Female or Male", color="blue",fontsize=10)
plt.ylabel("Count", color="blue",fontsize=10)
plt.title("Male and Female", color="blue",fontsize=10)
plt.show()


# # Boxplot for smokers and non smokers

# In[38]:


# Adjust the code for the box plot
sns.boxplot(data=df, x="Smoking_Yes", y="Sleep efficiency", color="green")
plt.xlabel("Smoking Status", color="green", fontsize=10)
plt.ylabel("Sleep Efficiency", color="green", fontsize=10)
plt.title("Sleep Efficiency for Smokers and Non-Smokers", color="green", fontsize=10)
plt.show()


# On average, non-smokers tend to have better sleep efficiency compared to smokers. Additionally, the minimum sleep efficiency observed in smokers is lower than that of non-smokers.

# # Sleep duration

# In[39]:


sns.kdeplot(data=df, x="REM sleep percentage",color="blue",fill=True)
plt.xlabel("Sleep Duration", color="blue", fontsize=10)
plt.ylabel("frequency", color="blue", fontsize=10)
plt.title("Sleep duration kdeplot", color="blue",fontsize=10)
plt.show()


# In[45]:


# Create a scatter plot for Sleep Duration vs. Sleep Efficiency
sns.scatterplot(data=df, x="Sleep duration", y="Sleep efficiency", hue="Smoking_Yes", palette="Set1")
plt.xlabel("Sleep Duration")
plt.ylabel("Sleep Efficiency")
plt.title("Sleep Duration vs. Sleep Efficiency")
plt.show()


# There appears to be no clear relationship between sleep duration and sleep efficiency for both smokers and non-smokers.

# In[47]:


# Create a histogram for Age distribution
sns.histplot(data=df, x="Age", kde=True, color="purple")
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution")
plt.show()


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a bar plot for Exercise Frequency grouped by Smoking Status
sns.countplot(data=df, x="Exercise frequency", hue="Smoking_Yes", palette="Set3")
plt.xlabel("Exercise Frequency")
plt.ylabel("Count")
plt.title("Exercise Frequency Levels by Smoking Status")
plt.legend(title="Smoking Status", labels=["Non-smoker", "Smoker"])
plt.show()


# The bar plot clearly illustrates that non-smokers engage in a higher level of exercise compared to smokers. The graph displays the distribution of exercise frequency levels for both groups, with non-smokers having a noticeably higher count across various exercise levels. This indicates that non-smokers are more likely to participate in regular exercise activities, while smokers show a lower tendency to engage in consistent physical activity. Overall, the results from the graph strongly support the notion that non-smokers generally prioritize and engage in more exercise compared to smokers.

# In[51]:


# Create a box plot for Sleep Efficiency by Gender
sns.boxplot(data=df, x="Gender", y="Sleep efficiency", palette="Set2")
plt.xlabel("Gender")
plt.ylabel("Sleep Efficiency")
plt.title("Sleep Efficiency by Gender")
plt.show()


# In[60]:


# Create a count plot for Bedtime by Smoking Status
sns.countplot(data=df, x="Smoking_Yes", hue="Bedtime", palette="Set1")
plt.xlabel("Smoking Status")
plt.ylabel("Count")
plt.title("Bedtime by Smoking Status")
plt.legend(title="Bedtime")
plt.show()


# In[61]:


# Calculate the counts for each category
bedtime_counts = df.groupby(['Smoking_Yes', 'Bedtime']).size().reset_index(name='Count')

# Filter for the relevant smoking status
smoking_yes_count = bedtime_counts.loc[bedtime_counts['Smoking_Yes'] == 1, 'Count'].values[0]
smoking_no_count = bedtime_counts.loc[bedtime_counts['Smoking_Yes'] == 0, 'Count'].values[0]

# Create a pie chart
labels = ['Smoking Yes', 'Smoking No']
sizes = [smoking_yes_count, smoking_no_count]
colors = ['#FF7F50', '#6495ED']
explode = (0.1, 0)  # To highlight the "Smoking Yes" slice, you can adjust the explode values as needed

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Distribution of Bedtime by Smoking Status')
plt.show()


# In[65]:


sns.boxplot(data=df,x="Alcohol consumption",y="Sleep efficiency", color="green")
plt.title("the effect of drinking alcohol on sleep efficiency", color="blue",fontsize=25)
plt.show()


# In[72]:


sns.boxplot(data=df,x="Caffeine consumption",y="Sleep efficiency", color="red")
plt.title("Relationship between caffeine consumption and sleep", color="yellow",fontsize=10)
plt.show()


# In[70]:


sns.kdeplot(data=df, x="Caffeine consumption",color="gold",fill=True)
plt.xlabel("Caffeine Consumption", color="gold", fontsize=10)
plt.ylabel("frequency", color="orange", fontsize=10)
plt.title("Caffeine Consumption Kdeplot", color="brown",fontsize=10)
plt.show()


# In[73]:


# Filter the data for individuals who smoke or drink alcohol
smoke_drink_df = df[(df['Smoking_Yes'] == 1) | (df['Alcohol consumption'] == 1)]

# Create a bar plot for Awakenings by Smoking and Alcohol Consumption
sns.barplot(data=smoke_drink_df, x='Smoking_Yes', y='Awakenings', hue='Alcohol consumption', palette='Set2')
plt.xlabel('Smoking Status')
plt.ylabel('Awakenings')
plt.title('Awakenings for Smokers and Alcohol Consumers')
plt.legend(title='Alcohol Consumption')
plt.show()


# # ML MODELS

# In[11]:



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Select features and target variable
features = ['Age', 'REM sleep percentage', 'Deep sleep percentage', 'Light sleep percentage', 'Awakenings','Smoking_Yes', 'Exercise frequency']
target = 'Sleep efficiency'

X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # LINEAR REGRESSION

# In[12]:



# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = model.predict(X_train)

# Calculate R-squared for the training set
r2_train = r2_score(y_train, y_train_pred)

# Calculate RMSE for the training set
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))

# Make predictions on the test set
y_test_pred = model.predict(X_test)

# Calculate R-squared for the test set
r2_test = r2_score(y_test, y_test_pred)

# Calculate RMSE for the test set
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

print('Training set:')
print('R-squared:', r2_train)
print('RMSE:', rmse_train)
print('')

print('Test set:')
print('R-squared:', r2_test)
print('RMSE:', rmse_test)


# # LASSO

# In[28]:



from sklearn.linear_model import Lasso

# Initialize and train the Lasso model
lasso = Lasso(alpha=1.0)  # You can adjust the value of alpha as per your requirements
lasso.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = lasso.predict(X_train)

# Calculate metrics for the training set
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = mse_train ** 0.5
r2_train = r2_score(y_train, y_train_pred)

# Make predictions on the test set
y_test_pred = lasso.predict(X_test)

# Calculate metrics for the test set
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test_lasso = mse_test ** 0.5
r2_test = r2_score(y_test, y_test_pred)

# Print the summary for training set
print("Training set:")
print("RMSE:", rmse_train)
print("R-squared:", r2_train)
print("")

# Print the summary for test set
print("Test set:")
print("RMSE:", rmse_test_lasso)
print("R-squared:", r2_test)
print("")


# In[14]:


# Lasso
lasso_residuals = y_test - lasso.predict(X_test)
plt.scatter(y_test, lasso_residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Actual Sleep Duration')
plt.ylabel('Residuals')
plt.title('Lasso - Residual Plot')
plt.show()


# In[ ]:





# In[15]:


# Lasso
plt.hist(lasso_residuals, bins=20)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Lasso - Distribution of Residuals')
plt.show()


# # RIDGE

# In[29]:



from sklearn.linear_model import Ridge

# Initialize and train the Ridge model
ridge = Ridge(alpha=1.0)  # You can adjust the value of alpha as per your requirements
ridge.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = ridge.predict(X_train)

# Calculate metrics for the training set
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = mse_train ** 0.5
r2_train = r2_score(y_train, y_train_pred)

# Make predictions on the test set
y_test_pred = ridge.predict(X_test)

# Calculate metrics for the test set
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test_ridge = mse_test ** 0.5
r2_test = r2_score(y_test, y_test_pred)

# Print the summary for the training set
print("Training set:")
print("RMSE:", rmse_train)
print("R-squared:", r2_train)
print("")

# Print the summary for the test set
print("Test set:")
print("RMSE:", rmse_test_ridge)
print("R-squared:", r2_test)
print("")


# In[17]:


# Ridge
ridge_residuals = y_test - ridge.predict(X_test)
plt.scatter(y_test, ridge_residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Actual Sleep Duration')
plt.ylabel('Residuals')
plt.title('Ridge - Residual Plot')
plt.show()


# In[18]:


# Ridge
plt.hist(ridge_residuals, bins=20)
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Ridge - Distribution of Residuals')
plt.show()


# # SVR
# 

# In[30]:



from sklearn.svm import SVR


# Initialize and train the SVR model
svr = SVR(kernel='linear', C=1.0, epsilon=0.1)  # You can adjust the kernel, C, and epsilon parameters as per your requirements
svr.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = svr.predict(X_train)

# Calculate metrics for the training set
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = mse_train ** 0.5
r2_train = r2_score(y_train, y_train_pred)

# Make predictions on the test set
y_test_pred = svr.predict(X_test)

# Calculate metrics for the test set
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test_svr = mse_test ** 0.5
r2_test = r2_score(y_test, y_test_pred)

# Print the summary for the training set
print("Training set:")
print("RMSE:", rmse_train)
print("R-squared:", r2_train)
print("")

# Print the summary for the test set
print("Test set:")
print("RMSE:", rmse_test_svr)
print("R-squared:", r2_test)
print("")


# In[20]:


# SVR
svr_residuals = y_test - svr.predict(X_test)
plt.scatter(y_test, svr_residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Actual Sleep Duration')
plt.ylabel('Residuals')
plt.title('SVR - Residual Plot')
plt.show()


# In[31]:



from sklearn.neural_network import MLPRegressor


# Initialize and train the Neural Network model
nn = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', solver='adam', random_state=42)  
# You can adjust the hidden_layer_sizes, activation, solver, and other parameters as per your requirements
nn.fit(X_train, y_train)

# Make predictions on the training set
y_train_pred = nn.predict(X_train)

# Calculate metrics for the training set
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = mse_train ** 0.5
r2_train = r2_score(y_train, y_train_pred)

# Make predictions on the test set
y_test_pred = nn.predict(X_test)

# Calculate metrics for the test set
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test_nn = mse_test ** 0.5
r2_test = r2_score(y_test, y_test_pred)

# Print the summary for the training set
print("Training set:")
print("RMSE:", rmse_train)
print("R-squared:", r2_train)
print("")

# Print the summary for the test set
print("Test set:")
print("RMSE:", rmse_test_nn)
print("R-squared:", r2_test)
print("")


# In[22]:


# Neural Network
nn_residuals = y_test - nn.predict(X_test)
plt.scatter(y_test, nn_residuals)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Actual Sleep Duration')
plt.ylabel('Residuals')
plt.title('Neural Network - Residual Plot')
plt.show()


# In[23]:


import matplotlib.pyplot as plt

# Plotting the training set results
plt.scatter(y_train, y_train_pred, color='blue', label='Actual vs. Predicted (Training set)')
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
plt.xlabel('Actual Sleep Duration')
plt.ylabel('Predicted Sleep Duration')
plt.title('Neural Network - Training Set')
plt.legend()
plt.show()

# Plotting the test set results
plt.scatter(y_test, y_test_pred, color='green', label='Actual vs. Predicted (Test set)')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Sleep Duration')
plt.ylabel('Predicted Sleep Duration')
plt.title('Neural Network - Test Set')
plt.legend()
plt.show()


# In[26]:


from sklearn.ensemble import RandomForestRegressor
# Train the random forest model
rfr = RandomForestRegressor(n_estimators=12, max_depth=3, min_samples_leaf=3, max_features=2, random_state=42)
rfr.fit(X_train, y_train)

# Make predictions on the training and test sets
y_train_pred = rfr.predict(X_train)
y_test_pred = rfr.predict(X_test)

# Calculate R-squared and RMSE for the training and test sets
rfr_train_r2 = r2_score(y_train, y_train_pred)
rfr_test_r2 = r2_score(y_test, y_test_pred)
rfr_train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
rfr_test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Print the R-squared and RMSE values for the random forest model
print(f"Random forest training set R-squared: {rfr_train_r2:.3f}")
print(f"Random forest test set R-squared: {rfr_test_r2:.3f}")
print(f"Random forest training set RMSE: {rfr_train_rmse:.3f}")
print(f"Random forest test set RMSE: {rfr_test_rmse:.3f}")


# In[33]:


import matplotlib.pyplot as plt

# Create a bar graph of the RMSE values
rmse_values = [rmse_test, rmse_test_lasso, rmse_test_ridge, rmse_test_svr, rmse_test_nn, rfr_test_rmse]
models = ['Linear', 'Lasso', 'Ridge', 'SVR', 'Neural Net', 'Random Forest']

plt.bar(models, rmse_values)
plt.title('RMSE for Sleep Prediction Models')
plt.xlabel('Model')
plt.ylabel('RMSE')

# Rotate the x-axis labels by 45 degrees
plt.xticks(rotation=45)

plt.show()


# Ridge regression is the best model developed for a sleep prediction project achieved promising results. The model was trained using a dataset, and on the training set, it exhibited a low root mean squared error (RMSE) of 0.0603 and a high R-squared value of 0.7994. These metrics indicate that the model's predictions were generally close to the actual values, with around 79.9% of the variance in the target variable being explained by the independent variables. The model's performance was further validated on a test set, where it demonstrated a similar level of accuracy with an RMSE of 0.0624 and an R-squared value of 0.791. These results suggest that the regression model holds promise for accurately predicting sleep patterns, though it is essential to consider additional factors and domain-specific knowledge to comprehensively evaluate its performance.
