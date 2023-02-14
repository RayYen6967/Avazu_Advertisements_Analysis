# Avazu Advertisements Analysis
Online advertisements have significant influences on the survival of a business. This project aims to forecast the click through rate for evaluating ad performance and identifying potential customers.



Executive Summary

Online advertisements have a significant influence on the success of a business. Effective advertisements can help businesses establish long-term relationships with customers and target the right potential customers, resulting in repeat sales and a high conversion rate. Therefore, our project is aimed to forecast the click-through rate (CTR) for evaluating ad performance and identifying potential users. The process comprises four stages: gathering data, pre-processing, training and testing the model, and evaluation. We use the data present on Kaggle and then convert the raw data by partitioning, recategorizing, etc. After that, we use the cleaned data to train predictive models. Finally, experimental results reveal that our best performance model produces an accuracy of 65.2% and is good at identifying both clicked and non-clicked ads. Time is the most influential factor, followed by the types of device model, website and app in predicting click-through rate.

● Research Questions:

* Is it possible to predict whether an ad would be clicked by viewers based on historical data? Furthermore, is it possible to successfully identify both clicked ads and non-clicked ads?
* Is it possible to drive values from the data in an efficient way under constraints? 

● Data:

* Data source: Kaggle CTR prediction contest  
    (https://www.kaggle.com/competitions/avazu-ctr-prediction)
* Data information: 11 days worth of Avazu data 
* Sample size: 45006431 observations
* Variables: (23 in total)
1. Dependent variable: click (0/1 for non-click/click)
2. Independent variables: hour (format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC), id, banner_pos, site_id, site_domain, site_category, app_id, app_domain, app_category, device_id, device_ip, device_model, device_type, device_conn_type, C1, C14 – C21 (Anonymized categorical variables for privacy protection)


Detailed methodologies are included in the project report.
