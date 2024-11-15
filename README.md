# Scene-GCN: A Time-Series Prediction method in Complex Monitoring Environments Through Spatial–Temporal Knowledge Graph (ST-GK)
## Requirements
- Ubuntu == 20.04
- tensorflow == 1.15
- scipy
- numpy
- matplotlib
- pandas
- math
## The overall framework of our model is shown below:
![Fig  1 Spatial–temporal forecasting modeling framework](https://github.com/user-attachments/assets/37ac7620-d84b-4215-b3bd-5553b56122ec)
## Data
The data files are available in `ST-GK-Tensorflow/ST-GKdata`. The results for `ST-GK-Tensorflow/ST-GKdata/3611817550_adj.xlsx` need to be obtained using `ST-GK-Tensorflow/Gauss.ipynb`.
## Program operation
After running `ST-GK-Tensorflow/Gauss.ipynb`, adjust the parameters in `ST-GK-Tensorflow/ST-GK_L2_loss_min_test_rmse.ipynb`. Then, execute this file to perform ST-GK training and testing. The test results will be saved in the generated `out` folder, including evaluation metrics, prediction results, true values, and other results that are helpful for observing during parameter adjustments for training.

Our baselines included:
1. Autoregressive Integrated Moving Average model (ARIMA)
2. Support Vector Regression model (SVR)
3. Gated Recurrent Unit model (GRU)
4. Attention Temporal Graph Convolutional Network for Traffic Forecasting (A3T-GCN)
5. A Temporal Graph Convolutional Network for Traffic Prediction (T-GCN)

The python implementation of the ARIMA/SVR/A3T-GCN model is in the baseline folder; the GRU model and the T-GCN model are in gru.py and tgcn.py, respectively.

## Results
### The results in Figure 8 were obtained using ST-GK-Tensorflow/ST-GK_L2 loss min test_rmse.ipynb, where the epoch has been fixed to 2500, and the values of gru_units were selected from [8,16,32,64,100,128].

![图片1](https://github.com/user-attachments/assets/490edc53-d75f-416b-89b3-931d00a941f0)
* Fig. 8 Comparison of the model’s predicted performance on the test set with different numbers of hidden units.

* The table corresponding to Figure 8 is shown below:

![image](https://github.com/user-attachments/assets/76e59359-2a5a-4c36-9f71-eb43c266c6e3)


### The results of Figs. 9-12 were obtained using the data in the ST-GK-Tensorflow/data_compare folder by running ST-GK-Tensorflow/plot.ipynb as well as ST-GK-Tensorflow/plotcut.ipynb, respectively. and ST-GK-Tensorflow/plotcut.ipynb respectively.

![image](https://github.com/user-attachments/assets/d385375f-e075-4789-8276-626dfccdd661)
  - Fig. 9 Comparison between model forecasting with embedded static attributes and forecasting with no attributes

![image](https://github.com/user-attachments/assets/6e3d5976-731d-4823-a216-d1b791d3bccb)
  - Fig. 10 Comparison between model forecasting with embedded dynamic attributes and forecasting with no attributes

![image](https://github.com/user-attachments/assets/be1f7f6f-9e39-4f93-b325-206efb35b26d)
  - Fig. 11 Comparison between model forecasting with embedded static and dynamic attributes and forecasting with no attributes

![image](https://github.com/user-attachments/assets/a10eac20-83c8-4cec-b301-1ecd1295deac)
  - Fig. 12 Comparison of embedding different state attributes for forecasting
