# User Action Prediction Model

## Description

  
This service predicts whether a user will perform a **target action** on the website of *SberAutoSubscription*. The target action refers to the user clicking one of the key buttons on the site: "Submit an Application" or "Request a Call". These actions are critical for the business, as they indicate a user's intention to initiate communication with the company. 

Once an application is submitted, a manager contacts the user to select an optimal car subscription plan. After successful verification, the user may receive a car either by picking it up in Moscow or via delivery to other cities across Russia. The goal of the prediction model is to assist the product team in improving the website's effectiveness by understanding user behavior better.

## Project Structure

- `data/csv/` — Source files containing data on sessions and hits, and also holidays list and city type dictionaries used at pipeline stage.
- `data/requests/` — .json-requests for model testing.
- `logs/catboost_info` — Some catboost logs.
- `media/finished_.mp3` — Sound file being played on pipeline completion.
- `src/main.py` — The main file containing the FastAPI code for serving the model.
- `src/pipeline.py` — The data processing pipeline creating the model.
- `models/user_action_prediction_pipeline.pkl` — Serialized machine learning model used for predictions.
- `notebooks/` — .ipynb-files used during model development.
- `environment.yml` — List of dependencies required to run the application.
- `README.md` — This guide for using the service.

## Endpoints 
The service provides three endpoints: 
1. **`/status`**: Returns the operational status of the service. 
2. **`/version`**: Provides the version of the currently deployed model. 
3. **`/predict`**: Predicts whether a user will perform a target action based on input session data.

## Why ROC-AUC?

We chose **ROC-AUC** as the primary evaluation metric for the model because it provides a balanced measure of classification performance. ROC-AUC considers both **true positive rate** and **false positive rate**, making it ideal for binary classification tasks where class imbalance is a potential issue. The metric helps ensure that the model is both sensitive to positive cases (users taking action) and does not generate excessive false positives (predicting action when none occurs).

In this case, we achieved a ROC-AUC score of 0.675, which surpasses the required threshold of 0.65.

## Feature Importance

Interpreting the importance of features is crucial for understanding how different variables influence the model's predictions. Below are some of the key features that the model considers most important:

- **utm_source**: The source of the traffic (e.g., Google, direct, social media) plays a significant role in user behavior.
- **device_category**: Whether the user is browsing from a mobile device, desktop, or tablet impacts engagement.
- **geo_country** and **geo_city**: Location data can significantly influence a user's likelihood of performing a target action.
- **visit_number**: The number of times a user has visited the site provides context on user familiarity and intent.

By understanding feature importance, the product team can make data-driven decisions to optimize the user journey on the website.

## Running the Application

### Installing Dependencies

To run the application, it is recommended to create a virtual environment using Conda and install all required libraries. If you are using `conda`, run the following command:
  
```bash
conda env create -f environment.yml
conda activate <your_environment_name>
```

### Starting the Service

To launch the FastAPI service, use the following command:

```bash
uvicorn src.main:app --reload
```

After starting the service, it will be available at `http://127.0.0.1:8000`.


## Usage

### Input Data

The model expects a JSON input containing user session data. Here’s an example of a request:

```
{
	"session_id":"2440182582781022068.1640107894.1640107894",
	"client_id":"568149281.1640107892",
	"visit_date":"2021-12-21",
	"visit_time":"20:31:34",
	"visit_number":1,
	"utm_source":"MvfHsxITijuriZxsqZqt",
	"utm_medium":"cpm",
	"utm_campaign":"FTjNLDyTrXaWYgZymFkV",
	"utm_adcontent":"xhoenQgDQsgfEPYNPwKO",
	"utm_keyword":"rcScqexpyNHshwBMMfPx",
	"device_category":"mobile",
	"device_os":"Android",
	"device_brand":"Samsung",
	"device_model":null,
	"device_screen_resolution":"412x892",
	"device_browser":"Samsung Internet",
	"geo_country":"Russia",
	"geo_city":"Moscow"
}
```

### Response

The response will contain the prediction: `0` — the user will not perform the target action, or `1` — the user will perform the target action.

Example response:
```
{
	"client_id":"568149281.1640107892",
	"prediction": 0
}
```


## Model Evaluation

The model was trained on user session data to predict whether a user will perform a target action on the website. It was evaluated using the **ROC-AUC** metric, achieving a score of 0.675, which indicates a reliable performance for this use case.

## Support

If you have any questions or issues using the model, feel free to reach out to the developer at a.o.zhuravlev@gmail.com.