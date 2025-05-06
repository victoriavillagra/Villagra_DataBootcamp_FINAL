# Villagra_DataBootcamp_FINAL
**Data Bootcamp Final Project: Predicting Bankruptcy**

Maria Victoria Villagra

JF Koehler

May 12, 2025


**INTRODUCTION**

Bankruptcy prediction plays a critical role in assessing the financial health of companies nowadays and mitigating economic risk. This project aims to explore whether a company’s bankruptcy can be predicted using specific financial ratios, drawing inspiration from Ohlson’s O–Score model for bankruptcy prediction. The research focuses on six key indicators capturing a firm’s leverage, liquidity, and profitability, namely, _Debt Ratio_ (liabilities/total assets), _Working Capital to Total Assets_, _Current Liability to Assets_, _Net Income to Total Assets_; along with two binary distress measures, _Liability–Assets Flag_ and _Net Income Flag_. Using a dataset from the Taiwan Economic Journal (1999–2009), the analysis was approached as a predictive classification task in which predictive models including Logistic Regression, K–Nearest Neighbors (KNN), and Random Forest were compared against a baseline–majority class predictor. Despite the imbalanced nature of the dataset the analysis shows that machine learning models can meaningfully enhance recall and ranking performance. Among them, Random Forest was able to provide the best balance between precision and recall, offering the most effective tool for identifying companies at risk of insolvency. 

**DATA DESCRIPTION**

As previously mentioned, the dataset used in this project was obtained from the Taiwan Economic Journal, covering publicly listed companies from 1999 to 2009. It includes financial statement information for a wide range of firms, including a total of 95 financial ratios reflecting key aspects of corporate performance. The target variable is a binary indicator, ‘Bankrupt?’, which identifies if a company filed for bankruptcy or not. 

To build the predictive models, I selected a set of financial ratios inspired by the aforementioned Ohlson O score framework. The variables I chose reflect core dimensions of financial distress: leverage, liquidity, and profitability. Specifically, I included the Debt Ratio % to capture a firm’s financial leverage, Working Capital to Total Assets to assess short-term liquidity, and Current Liability to Assets to evaluate solvency. I also added two binary flags: one indicating whether a firm's liabilities exceed its assets (a sign of potential insolvency), and another reflecting whether the company reported negative income for two consecutive years. Finally, I included Net Income to Total Assets as a profitability measure. These ratios closely align with the Ohlson model's logic while leveraging variables readily available in the dataset.

A preliminary inspection revealed a significant class imbalance, with only about 3% of firms labeled as bankrupt. This imbalance has important implications for model evaluation and guided the use of metrics such as recall and ROC AUC rather than accuracy alone. Additionally, before analyzing the model, I calculated the correlation between the variables I had chosen as predictors in order to make sure it made sense to use them for my models (>0.8 ~ highly correlated.) The correlation heatmap (Figure 1) revealed that Debt Ratio % and Current Liability to Assets had a high correlation of 0.84, indicating potential multicollinearity. To reduce redundancy, especially in my logistic regression model, I opted to retain Debt Ratio % as my leverage proxy and substitute Current Liability to Assets for the Quick Ratio. The Quick Ratio (Acid Test) measures a firm’s immediate liquidity by comparing its most liquid assets to current liabilities. In the context of bankruptcy prediction, it helps capture short-term financial stress, which may precede insolvency. This complements leverage and profitability metrics by focusing on the firm’s ability to survive sudden cash obligations, making it a valuable addition to the models. In this second correlation heatmap (Figure 2), none of the correlations surpassed 0.8. As a result, I was able to determine that none of the predictors were highly correlated. 

_BOXPLOTS BY BANKRUPTCY CLASS_

In terms of further exploring the data and possible underlying relationships between the chosen variables, I first created boxplots by bankruptcy class. This allowed me to better analyze how each variable behaved across the two classes (healthy firms vs. bankrupt firms.) I was able to observe the following:

•	_Debt Ratio % (Figure 3):_ Bankrupt firms show a slightly higher median Debt Ratio than healthy firms, with limited overlap between boxplots. The tighter distribution and outliers reaching up to 0.6 suggest excessive leverage in distressed firms, reinforcing its role as a key bankruptcy indicator.

•	_Working Capital to Total Assets (Figure 3):_ Healthy firms generally have higher Working Capital Ratios, with less overlap between groups. Downward outliers among bankrupt firms suggest weaker liquidity buffers, supporting this variable’s relevance in identifying financial distress.

•	_Quick Ratio (Figure 4):_ Most firms, bankrupt or not, have a Quick Ratio of zero, though healthy firms show higher outliers (up to 8). This implies that while low liquidity is common, stronger Quick Asset ratios may be linked to financial stability.

•	_Liability-Assets Indicator (Figure 4):_ Nearly all firms report liabilities below assets, with rare outliers at 1.0 in both groups. While uncommon, these over-leveraged cases suggest the indicator may be relevant but not sufficient alone for prediction.

•	_Net Income to Total Assets (Figure 5):_ Healthy firms have higher profitability and less variability, while bankrupt firms show a wider spread and more low-end outliers. This supports profitability as a strong signal in bankruptcy prediction.

•	_Net Income Flag (Figure 5):_ Both bankrupt and healthy firms cluster at 1.0, indicating widespread consecutive net losses. Due to the lack of variation, it is likely that this feature offers little discrimination between the two classes in this dataset.

_HISTOGRAMS OF FEATURE DISTRIBUTIONS_

To understand the shape and variation of each feature, I plotted histograms (Figure 6) for the selected variables. Debt Ratio % and Working Capital to Total Assets both showed bell-shaped distributions. Debt Ratio % was slightly right skewed and centered around moderate leverage levels, while Working Capital to Total Assets clustered between 0.60 and 1.00, reflecting strong liquidity among firms. Net Income to Total Assets also followed a fairly normal distribution centered near 0.8, with a noticeable concentration just above that value, consistent with earlier profitability patterns. In contrast, Quick Ratio, Liability-Assets, and Net Income Flag each showed little to no variation, with nearly all firms reporting zero or fixed values. This suggests they may have limited usefulness as predictive features compared to the others, which exhibit more informative variation.

_SCATTERPLOTS BETWEEN SELECTED PAIRS_

Lastly, to further explore interactions between financial indicators, I created scatterplots for three key ratio combinations: Net Income to Total Assets vs. Debt Ratio % (Figure 7),  Net Income to Total Assets vs. Working Capital to Total Assets (Figure 8), and Working Capital to Total Assets vs. Debt Ratio % (Figure 9.) These visualizations helped examine how leverage, liquidity, and profitability relate to bankruptcy status. Across all plots, healthy firms tend to cluster tightly in favorable regions, while bankrupt firms show more spread and downward drift. In the Net Income to Total Assets vs. Debt Ratio % plot, bankrupt firms show slightly lower profitability despite moderate leverage. In the Net Income to Total Assets vs. Working Capital plot, healthy firms dominate the top-right quadrant, while bankrupt firms trail downward, reflecting weaker liquidity-profitability profiles. The third plot shows a subtle diagonal pattern among bankrupt firms, suggesting that higher leverage often coincides with lower liquidity. Overall, these relationships visually reinforce the predictive value of the selected financial ratios.

**MODELS & METHOD**

After completing the Exploratory Data Analysis, I moved on to the modeling stage with the goal of determining whether bankruptcy can be effectively predicted using specific financial ratios and supervised learning models. As mentioned earlier, this project draws inspiration from the Ohlson O-score model, a well-known logistic regression framework that predicts bankruptcy using financial ratios related to leverage, liquidity, and profitability. Guided by this framework and taking into consideration the results yielded by the EDA, I selected six ratios that capture these dimensions: Debt Ratio %, Working Capital to Total Assets, Quick Ratio, Liability-Assets Flag, Net Income to Total Assets, and Net Income Flag.

To assess performance meaningfully, I began with a baseline model that always predicts the majority class, non-bankrupt firms. Given the class imbalance, where bankruptcies account for only a small portion of the total, this approach achieved high accuracy but failed to identify any bankrupt companies, reinforcing its limitations as a predictive tool.

To improve on this, I trained and evaluated three supervised learning models: Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest. These models were selected for their increasing complexity and their potential to capture non-linear relationships that the baseline model overlooks. Logistic Regression closely mirrors the Ohlson model in both methodology and interpretability, making it a natural starting point. KNN and Random Forest were then used to explore whether more flexible, data-driven approaches could better detect firms at risk of bankruptcy.

**RESULTS & INTERPRETATION** 

_MODEL 1: Logistic Model_

The logistic regression model achieved an overall accuracy of ≈ 97%, which may initially appear impressive. However, given the highly imbalanced nature of the dataset, this metric is misleading. The model correctly classified 1,315 out of 1,320 non-bankrupt firms, but only 5 out of 44 bankrupt firms. This results in a recall of just 11% for the bankrupt class, meaning it failed to identify most firms at risk of bankruptcy.

The precision for the bankrupt class was 50%, indicating that when the model does predict bankruptcy, it's correct half the time, but this comes with very few actual positive predictions. The F1-score for class 1 (bankrupt) is low at 0.19, reflecting the imbalance between precision and recall. However, the ROC AUC score of 0.8937 is strong and suggests that the model has good ranking ability, meaning it can generally distinguish bankrupt from non-bankrupt firms in terms of predicted probability, even if it struggles with classification at the default threshold.

These results highlight that while logistic regression shows potential in ranking bankruptcy risk, further adjusting such as re-sampling the data, or trying out other models may help improve its ability to correctly flag distressed firms.

_MODEL 2: K–Nearest Neighbors (KNN) Model_

The KNN model achieved an overall accuracy of 96%, similar to logistic regression. It correctly classified 1,306 non-bankrupt firms and 9 bankrupt firms, which improved recall for the minority class from 11% in logistic regression to 20%. This indicates better sensitivity to bankruptcy cases. The precision for bankrupt firms dropped slightly to 39%, but the model is now identifying more at-risk firms, leading to a higher F1-score of 0.27 for class 1. However, it also misclassified 14 healthy firms bankrupt, slightly increasing false positives. The ROC AUC score of 0.8212 suggests that KNN can reasonably rank firms by their bankruptcy risk, though its performance in ranking is weaker than logistic regression.

Ultimately, KNN offers a better tradeoff between recall and precision than the baseline and logistic models but still struggles to consistently capture bankrupt firms. This highlights the potential value in exploring other model such as Random Forest. Even so, before moving on to the last model, I attempted to improve the KNN model’s performance by maximizing recall, using Grid Search to identify the optimal number of neighbors (k) for KNN. After tuning the number of neighbors (k) using Grid Search with recall as the scoring metric, the optimal value was found to be k = 1. This version of the KNN model achieved a recall of 34% for the bankrupt class, a significant improvement over both the baseline model and the KNN untuned version. It correctly identified 15 out of 44 bankrupt firms. However, the precision dropped to 29%, and 36 healthy firms were misclassified as bankrupt, increasing the number of false positives. The ROC AUC score also declined to 0.6568, indicating that the model’s ability to rank firms by risk worsened somewhat compared to previous versions. This underscores a classic tradeoff between recall and overall ranking performance. If the priority is to catch as many bankruptcies as possible, even at the risk of more false alarms, this tuned model might be appropriate. 

_MODEL 3: Random Forest Model_

The Random Forest model achieved a 97% overall accuracy, matching the performance of logistic regression and the baseline model. However, it significantly improved its detection of bankrupt firms, correctly identifying 9 out of 44 cases, resulting in a recall of 20%, similar to the tuned KNN model. Notably, it reached the highest precision for the bankrupt class at 64%, meaning when it predicts bankruptcy, it is correct nearly two-thirds of the time.

The F1-score for bankrupt firms was 0.31, the best balance so far between precision and recall. Additionally, the model achieved a strong ROC AUC score of 0.8614, confirming that it can rank firms by bankruptcy risk quite well. These results demonstrate that Random Forest is both effective at distinguishing between classes and confident in its predictions.

**CONCLUSION & NEXT STEPS** 

This project set out to explore whether bankruptcy could be accurately predicted using a set of financial ratios, guided by the structure of the Ohlson O score model. Using six carefully selected features related to leverage, liquidity, and profitability, I evaluated models, namely, Logistic Regression, K Nearest Neighbors, and Random Forest, against a baseline majority-class predictor.

All models outperformed the baseline in recall and ranking ability, with Random Forest emerging as the best overall performer, achieving the strongest balance between precision and recall, and the highest F1-score for the bankrupt class. Logistic Regression, while conceptually aligned with the Ohlson framework, struggled with recall at the default threshold, and KNN, although improved through tuning, still showed tradeoffs in precision and ranking.

Looking ahead, several steps could enhance the bankruptcy prediction process:

•	Addressing class imbalance through resampling techniques or cost-sensitive learning could improve model sensitivity to bankrupt firms.

•	Adding temporal context (e.g., lagged financial ratios or trends over time) might help capture early signs of distress.

In the final analysis, this research confirms that targeted financial ratios, when paired with the right classification models, can provide meaningful insights into corporate bankruptcy risk.




**Figures**


_Figure 1_

<img width="302" alt="image" src="https://github.com/user-attachments/assets/b0f5559a-129e-4101-ab48-492a757c2958" />



_Figure 2_

<img width="297" alt="image" src="https://github.com/user-attachments/assets/a13067ca-b4bf-464e-bbb5-5343b197251a" />



_Figure 3_

<img width="281" alt="image" src="https://github.com/user-attachments/assets/3ccd30fa-d630-463c-b467-b78b06d17bde" />



_Figure 4_

<img width="273" alt="image" src="https://github.com/user-attachments/assets/e886ef0a-c479-4b61-97f4-a6081fce3da4" />



_Figure 5_

<img width="276" alt="image" src="https://github.com/user-attachments/assets/d7f00f9c-4548-431d-a62e-75e024fab4c5" />



_Figure 6_

<img width="356" alt="image" src="https://github.com/user-attachments/assets/dc2596eb-64a6-4101-bdf7-df3b392ebcab" />



_Figure 7_

<img width="345" alt="image" src="https://github.com/user-attachments/assets/ac7fb3a4-7fbf-41ce-91e5-cf6908ba24df" />



_Figure 8_

<img width="354" alt="image" src="https://github.com/user-attachments/assets/2569a64e-ee60-4b75-9fb0-8255d742a563" />



_Figure 9_

<img width="332" alt="image" src="https://github.com/user-attachments/assets/48b97781-ddf8-4441-837e-ac9ecfa76267" />













