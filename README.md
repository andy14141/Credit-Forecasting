# Credit-Forecasting
Classifying whether or not a potential client should be granted a loan based on their personal features

# Summary 
This project involves analyzing credit data using a decision tree classifier. The project begins with importing the necessary libraries, including pandas for data manipulation and sklearn for machine learning. The dataset is read from a CSV file, and duplicates and missing values are removed.

The "BAD" column, representing loan repayment status, is converted from ordinal values to categorical values (0 as "REPAID" and 1 as "DEFAULT") for better understanding. The dataset is split into independent variables (x) and the dependent variable (y) for training and testing.

A decision tree classifier is created and fitted using the training data. The accuracy of the model is evaluated using the training and testing scores. To combat overfitting, the max_depth parameter of the decision tree is adjusted by looping through different values (1 to 10) and plotting the corresponding accuracy scores.

The optimal max_depth value is determined to be 8, where the model achieves a good balance between training and testing accuracy. A new decision tree classifier is created with max_depth set to 8, and its accuracy scores are calculated. The final decision tree is visualized with a depth of 3 for better interpretation.

# How to Use
1. Ensure that the required libraries (pandas, sklearn) are installed.
2. Prepare the credit data in a CSV file named "credit_data.csv" with appropriate column names.
3. Copy and run the provided code in a Python environment such as Jupyter Notebook or an IDE with Python support.
4. Adjust the max_depth parameter in the code if desired.
5. Analyze the training and testing scores to evaluate the model's performance.
6. Visualize the decision tree to replicate the model's decision-making process.

# Evaluation 
The project demonstrates the process of using a decision tree classifier to analyze credit data. The initial model shows signs of overfitting, as indicated by a perfect training score and a lower testing score. To address this, the project implements parameter tuning by adjusting the max_depth value.

The evaluation of different max_depth values reveals that a depth of 8 provides the best trade-off between training and testing accuracy. The final model achieves a training score of 0.891 and a testing score of 0.763, indicating reasonable performance in predicting loan repayment status.

The decision tree visualization helps understand the decision-making process of the model, although it is limited to a depth of 3 due to space constraints. Overall, the project showcases data preprocessing, model building, evaluation, and parameter tuning, providing insights into analyzing credit data using a decision tree classifier.
