
# Table of Content

- [Background](#background)
- [Install](#install)
- [Usage](#usage)

# 1. Background

In this project, we request historical travel time data within Vic from ksqlDB, propcess these data for the prediction models and evaluate the results.

# 2. Install

This porject uses Python, Anaconda and PyTorch. Anaconda is a useful package control software. PyTorch is a easily used package to do research in neural network models. Go check them out If you don't have it locally installed.

In anconda terminal, you can input the python environment of this project called VicRoad.yaml.

```conda
conda env create -f VicRoad.yaml
```

Please check whether you have downloaded [PyTorch](https://pytorch.org/). It is noticed that the version of PyTorch is different on the different platform. Also, if you want to use GPU in this project, you should check the whether you have downloaded Nvidia CUDA Compiler (NVCC).

# 3. Program Structure

![image](file%20flow.png)

# 4. Usage

![image](code%20structure.png)

## 4.1 Entry

All the code you can run is in the directory *forecast_system*. The entry of program is a file named [main.py](forecast_system/main.py). Firthly, you need to define the address of kasqlDB and the table you want to request data. Then, you should define the region and the period you want to do research. Thirdly, you should define the forward steps you consider and backward steps you want to predict. Also, you can choose the duration of each point in time series. The original interval is 30 seconds, however, the travel time series does not have a significant variation in a short period. Therefore, you may want to aggregate original series. Also, we consider the temporal-spatial features in the data processing and the details you can check the report. Thirdly, you can import models from [sklearn](https://scikit-learn.org/stable/) and evaluate models with multiple metrics.

You can run the program directly by the following command in the terminal.

```bash
python main.py
```

**It is noticed that the neural network ([nn_coach.py](forecast_system/nn_library/nn_coach.py)) models have not import in main.py since it is still in the testing process.**

## 4.2 Model Builder

[model_builder.py](forecast_system/model_builder.py) is the most important part in this program, which is the control cneter. It will request raw travel data from [data_harvester.py](forecast_system/data_harvester.py) and calculates the spatial features from [road_network.py](forecast_system/road_network.py). Then, it will do feature selection from line 103 to line 113 based on the [data_processor.py](forecast_system/data_processor.py). Finally, the model training and evaluation are done by calling [forecaster.py](forecast_system/forecaster.py) and [evaluator.py](forecast_system/evaluator.py).

## 4.3 Data Haverster

As for the process of requesting travel time data from ksqlDB, you can check [data_harvester.py](forecast_system/data_harvester.py). It converts the raw Json data into DataFrame of [Pandas](https://pandas.pydata.org/) after droping duplications and fill missing values with linear interpolation.

## 4.4 Road Network

In the [road_network.py](forecast_system/road_network.py), we match spatial data of links with travel time series. Also, you can calculate POI density with different buffer distances in this file. 

## 4.5 Data Processor

In the [data_processor.py](forecast_system/data_processor.py), you can aggregate data with 30-second interval by calculating mean value with other duration of intervals. Then, it can roll the time series with the format of multiple forward steps and multiple backward steps to create X and Y for machine learning models. In addition, you can call the functions in this file to add other spatial-temporal features.

## 4.6 Forecaster

In the [forecaster.py](forecast_system/forecaster.py), it can normalize and split, and train the model from [sklearn](https://scikit-learn.org/stable/). The prediction results will serve for the following evaluation.

## 4.7 Evaluator

In the [evaluator.py](forecast_system/evaluator.py), it will calculate the MAPE, RMSE, MAE of each link based on the true travel times and prediction resutls.

