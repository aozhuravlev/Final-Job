# User Action Prediction Model

## Description

  
This service predicts whether a user will perform a target action based on session data such as `utm_*`, `device_*`, and `geo_*` features. The model was trained using machine learning techniques (XGBoostClassifier) and achieved a ROC-AUC score of 0.675, exceeding the required threshold of 0.650.

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

The model was trained using user session data. The primary evaluation metric is ROC-AUC, which achieved a score of 0.675 on test data.

## Support

If you have any questions or issues using the model, feel free to reach out to the developer at a.o.zhuravlev@gmail.com.
