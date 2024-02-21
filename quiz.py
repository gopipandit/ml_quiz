import streamlit as st
import numpy as np
#import plotly.graph_objects as go


st.subheader("Hello Learner!! This is a quiz for ML.")


# st.text("1. What is the main goal of linear regression?")



def get_quiz(i,question, answers,correct_answer):


    score = 0
    answer1 = st.radio(f"***Question {i+1}***: ***{question}***",
                    options= answers, index= None)

    if answer1 != None:
        if answer1 == correct_answer:
            st.success("Correct (1/1)")
            score += 1
        else:
            st.error("It is not a correct answer.")
    return score



# Define the questions, options, and correct answers
questions = [
    "What is the main goal of linear regression?",
    "Which of the following is not an assumption of linear regression?",
    "Logistic regression is used for:",
    "Decision trees are used for:",
    "Which of the following is a regularization technique used in linear regression?",
    "Random Forest is an ensemble learning method based on:",
    "Boosting models combine multiple weak learners to create a strong learner by:",
    "Bagging (Bootstrap Aggregating) is used to:",
    "Which of the following is a clustering algorithm?",
    "K-means clustering aims to:",
    "Which of the following is not a distance metric used in clustering?",
    "What is the silhouette score used for in clustering?",
    "Which statistical test is used to determine whether there is a significant difference between the means of two groups?",
    "What does p-value represent in hypothesis testing?",
    "Which of the following statements is true about the correlation coefficient?",
    "What is the purpose of cross-validation in machine learning?",
    "In machine learning, overfitting occurs when:",
    "Which of the following is not a method for handling missing data?",
    "Which of the following evaluation metrics is used for classification problems?",
    "What is the purpose of feature scaling in machine learning?",
    "Which of the following is not a dimensionality reduction technique?",
    "What does the term 'curse of dimensionality' refer to in machine learning?",
    "Which of the following is not a type of ensemble learning method?",
    "Which of the following statements is true about bias and variance in machine learning models?",
    "Which of the following is not a hyperparameter of the K-nearest neighbors (KNN) algorithm?",
    "What is the purpose of the elbow method in K-means clustering?",
    "Which of the following is not a disadvantage of the K-means clustering algorithm?",
    "What is the main advantage of hierarchical clustering over K-means clustering?",
    "Which of the following is not a step in the K-means clustering algorithm?",
    "Which of the following is not a type of regularization in machine learning?",
    "Which of the following is not a disadvantage of decision trees?",
    "Which of the following is not a type of boosting algorithm?",
    "Which of the following statements is true about feature importance in random forests?",
    "Which of the following is not a clustering algorithm?",
    "What is the main advantage of DBSCAN over K-means clustering?",
    "What is the main disadvantage of hierarchical clustering?",
    "Which of the following is not a hyperparameter of the Random Forest algorithm?",
    "Which of the following is not a step in the AdaBoost algorithm?",
    "Which of the following statements is true about the bias-variance tradeoff?",
    "Which of the following is not a type of cross-validation?",
    "What is the purpose of grid search in machine learning?",
    "Which of the following is not a disadvantage of PCA (Principal Component Analysis)?",
    "Which of the following is not a method for handling imbalanced datasets?",
    "Which of the following is not a type of kernel function used in Support Vector Machines (SVM)?",
    "Which of the following is not a measure of feature importance in decision trees?",
    "What is the main advantage of Lasso (L1) regularization over Ridge (L2) regularization?",
    "Which of the following is not a statistical test used for hypothesis testing?",
    "Which of the following is not a method for reducing overfitting in machine learning models?"
]

options = [
    ["Classification", "Regression", "Clustering", "Dimensionality reduction"],
    ["Homoscedasticity", "Multicollinearity", "Normality of residuals", "Independence of observations"],
    ["Regression", "Classification", "Clustering", "Dimensionality reduction"],
    ["Regression", "Classification", "Clustering", "Dimensionality reduction"],
    ["L1 Regularization", "L2 Regularization", "Elastic Net Regularization", "All of the above"],
    ["Decision trees", "Linear regression", "Logistic regression", "Support vector machines"],
    ["Training the models sequentially", "Training the models independently", "Taking the average of the models", "Selecting the best model"],
    ["Reduce variance", "Reduce bias", "Reduce both variance and bias", "Increase model complexity"],
    ["K-means", "Linear regression", "Logistic regression", "Decision tree"],
    ["Minimize within-cluster variance", "Maximize between-cluster variance", "Minimize both within-cluster and between-cluster variance", "Maximize within-cluster variance"],
    ["Euclidean distance", "Manhattan distance", "Minkowski distance", "Pearson correlation"],
    ["Measuring the compactness of clusters", "Measuring the separation between clusters", "Measuring the quality of clustering", "Measuring the density of clusters"],
    ["t-test", "ANOVA", "Chi-square test", "Mann-Whitney U test"],
    ["The probability of making a Type I error", "The probability of making a Type II error", "The probability of obtaining the observed results by chance, assuming the null hypothesis is true", "The probability of obtaining the observed results by chance, assuming the alternative hypothesis is true"],
    ["It ranges from -1 to 1", "It ranges from 0 to 1", "It ranges from -1 to 0", "It ranges from 0 to -1"],
    ["To train the model on multiple datasets", "To evaluate the performance of the model", "To select the best hyperparameters for the model", "All of the above"],
    ["The model performs well on the training data but poorly on unseen data", "The model performs poorly on the training data and unseen data", "The model performs well on the training data and unseen data", "The model has too few parameters"],
    ["Removing rows with missing values", "Imputing missing values with the mean", "Imputing missing values with the median", "Imputing missing values with random values"],
    ["Mean Absolute Error (MAE)", "Mean Squared Error (MSE)", "Accuracy", "R-squared"],
    ["To make the features easier to interpret", "To speed up the training process", "To ensure that all features have the same scale", "To reduce the dimensionality of the feature space"],
    ["Principal Component Analysis (PCA)", "t-Distributed Stochastic Neighbor Embedding (t-SNE)", "Independent Component Analysis (ICA)", "Linear Discriminant Analysis (LDA)"],
    ["The increase in dimensionality leads to sparsity in the data", "The increase in dimensionality leads to denser data", "The increase in dimensionality leads to increased computational complexity", "The increase in dimensionality leads to decreased computational complexity"],
    ["Random Forest", "AdaBoost", "Gradient Boosting", "All of the above"],
    ["High bias and low variance can lead to overfitting", "High bias and high variance can lead to overfitting", "Low bias and low variance can lead to overfitting", "Low bias and high variance can lead to overfitting"],
    ["Number of neighbors (K)", "Distance metric", "Learning rate", "Weight function"],
    ["To determine the optimal number of clusters", "To determine the optimal value of K", "To determine the optimal initialization of centroids", "To determine the optimal distance metric"],
    ["Sensitivity to outliers", "Dependence on the initial choice of centroids", "Inability to handle non-linearly separable data", "Difficulty in determining the number of clusters"],
    ["It does not require the number of clusters to be specified in advance", "It is faster and more efficient", "It can handle large datasets with high dimensionality", "It is less sensitive to the choice of initial centroids"],
    ["Initialization of centroids", "Assignment of data points to clusters", "Update of centroids", "Calculation of within-cluster variance"],
    ["Dropout Regularization", "L1 Regularization", "L2 Regularization", "Elastic Net Regularization"],
    ["They are prone to overfitting", "They cannot handle missing values", "They are computationally expensive", "They are not interpretable"],
    ["Random Forest", "Gradient Boosting", "Randomized Search", "XGBoost"],
    ["It is calculated based on the importance of features in individual trees", "It is calculated based on the correlation between features", "It is calculated based on the weights assigned to features in the model", "It is calculated based on the mutual information between features"],
    ["Linear regression", "K-means", "DBSCAN", "Hierarchical clustering"],
    ["It does not require the number of clusters to be specified in advance", "It is less sensitive to outliers", "It is computationally faster", "It can handle non-linearly separable clusters"],
    ["It is sensitive to outliers", "It cannot handle large datasets", "It has a higher computational complexity", "It requires the number of clusters to be specified in advance"],
    ["Number of trees", "Maximum depth of trees", "Learning rate", "Number of features to consider at each split"],
    ["Initialization of weights for data points", "Training of weak learners", "Combining the predictions of weak learners", "Updating the weights of data points"],
    ["Increasing model complexity always decreases bias and variance", "Increasing model complexity increases bias and decreases variance", "Decreasing model complexity increases bias and decreases variance", "Decreasing model complexity always decreases bias and variance"],
    ["Holdout validation", "K-fold cross-validation", "Leave-one-out cross-validation", "Random split cross-validation"],
    ["To search for the best hyperparameters of a model", "To search for the best features of a model", "To search for the best algorithm for a given problem", "To search for the best dataset for a given problem"],
    ["It is computationally expensive", "It can only be applied to numerical data", "It can be difficult to interpret the principal components", "It does not handle multicollinearity well"],
    ["Undersampling", "Oversampling", "SMOTE (Synthetic Minority Over-sampling Technique)", "Normalization"],
    ["Linear kernel", "Polynomial kernel", "Radial basis function (RBF) kernel", "Logistic kernel"],
    ["Gini impurity", "Entropy", "Information gain", "Standard deviation"],
    ["Lasso can select important features by shrinking some coefficients to zero", "Lasso is computationally faster", "Lasso is less sensitive to outliers", "Lasso is less prone to overfitting"],
    ["Student's t-test", "ANOVA", "Chi-square test", "F-test"],
    ["Feature engineering", "Regularization", "Bagging", "Early stopping"]
]

correct_answers = [
    "Regression", "Multicollinearity", "Classification", "Classification", "All of the above",
    "Decision trees", "Training the models sequentially", "Reduce variance", "K-means", "Minimize within-cluster variance",
    "Pearson correlation", "Measuring the quality of clustering", "t-test", "The probability of obtaining the observed results by chance, assuming the null hypothesis is true",
    "It ranges from -1 to 1", "All of the above", "The model performs well on the training data but poorly on unseen data", "Imputing missing values with random values",
    "Accuracy", "To ensure that all features have the same scale", "t-Distributed Stochastic Neighbor Embedding (t-SNE)",
    "The increase in dimensionality leads to increased computational complexity", "Randomized Search", "High bias and high variance can lead to overfitting",
    "Learning rate", "To determine the optimal value of K", "Inability to handle non-linearly separable data", "It does not require the number of clusters to be specified in advance",
    "Calculation of within-cluster variance", "Dropout Regularization", "They are prone to overfitting", "Random Forest",
    "It is calculated based on the importance of features in individual trees", "Linear regression", "It is less sensitive to outliers",
    "It has a higher computational complexity", "Number of trees", "Combining the predictions of weak learners", "Increasing model complexity increases bias and decreases variance",
    "Holdout validation", "To search for the best hyperparameters of a model", "It can be difficult to interpret the principal components",
    "Normalization", "Logistic kernel", "Standard deviation", "Lasso can select important features by shrinking some coefficients to zero",
    "F-test", "Feature engineering"
]

# Display the questions and options


# for i, question in enumerate(questions):
#     st.subheader(f"Question {i+1}: {question}")
#     option_selected = st.radio("", options[i], index=-1)  # Set default index to -1
#     if option_selected == correct_answers[i]:
#         st.write("Your answer is correct!")
#     else:
#         st.write(f"Sorry, the correct answer is: {correct_answers[i]}")


# start = 0
# end = len(correct_answers)
# #num_values = 10

# np.random.seed(42)
# random_values = np.random.choice(np.arange(start, end), size=end, replace=False)
# counter = 0
total_score = 0

for i in range(len(correct_answers)):
    total_score += get_quiz(i, questions[i], options[i], correct_answers[i])

if st.button("Submit"):
    st.write(f"Your total score is: {total_score}")

    total_questions = len(correct_answers)
    score_percentage = (total_score / total_questions) * 100

    # # Plot a gauge chart to visualize the score percentage
    # fig = go.Figure(go.Indicator(
    #     mode="gauge+number",
    #     value=score_percentage,
    #     title={'text': "Score Percentage"},
    #     gauge={'axis': {'range': [None, 100]},
    #         'bar': {'color': "darkblue"},
    #         'steps': [
    #             {'range': [0, 60], 'color': "red"},
    #             {'range': [60, 80], 'color': "orange"},
    #             {'range': [80, 100], 'color': "green"}]
    # }))

    # st.plotly_chart(fig)


