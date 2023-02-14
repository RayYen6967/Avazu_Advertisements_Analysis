# Avazu Advertisements Analysis
Online advertisements have significant influences on the survival of a business. This project aims to forecast the click through rate for evaluating ad performance and identifying potential customers.

● Research Questions:

* Is it possible to predict whether an ad would be clicked by viewers based on historical data? Furthermore, is it possible to successfully identify both clicked ads and non-clicked ads?
* Is it possible to drive values from the data in an efficient way under constraints? 

● Data:

* Data source: Kaggle CTR prediction contest  
    (https://www.kaggle.com/competitions/avazu-ctr-prediction)
* Data information: 11 days worth of Avazu data 
* Sample size: 45006431 observations
* Variables:
1. Dependent variable: click (0/1 for non-click/click)
2. Independent variables: hour (format is YYMMDDHH, so 14091123 means 23:00 on Sept. 11, 2014 UTC), id, banner_pos, site_id, site_domain, site_category, app_id, app_domain, app_category, device_id, device_ip, device_model, device_type, device_conn_type, C1, C14 – C21 (Anonymized categorical variables for privacy protection)

-- Detailed methodologies are included in the project report.
