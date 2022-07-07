# MLFlow Documentation

# Steps:

1. Create an Experiment in the mlflow portal
    
    ![Create experiment](doc/1.png)
    
    For this experiment we will name it:
    
    ```markdown
    Experiment Name: test-sklearn-autolog
    Artifact Location: s3://mlflow-volkno/test-sklearn
    ```
    
2. Change to mlflowtest repo. Make sure you are using PIP as your package manager and that mlflow is installed. 
    
    ```bash
    pip3 install mlflow
    cd mlflowtest
    ```
    
3. Run lower quality model to see poor performance

```bash
python3 sklearn_autolog.py lower_quality
```

If we now check Mlflow, the following information is stored

![Screen Shot 2022-07-06 at 10.13.21 PM.png](doc/Screen_Shot_2022-07-06_at_10.13.21_PM.png)

![Screen Shot 2022-07-06 at 10.14.14 PM.png](doc/Screen_Shot_2022-07-06_at_10.14.14_PM.png)

![Screen Shot 2022-07-06 at 10.14.00 PM.png](doc/Screen_Shot_2022-07-06_at_10.14.00_PM.png)

![Screen Shot 2022-07-06 at 10.14.29 PM.png](doc/Screen_Shot_2022-07-06_at_10.14.29_PM.png)

1. Run on higher quality model. This model uses one-hot encoding on categorical variables instead of dropping them. this should result in a better fit.

```bash
python3 sklearn_autolog.py higherquality
```

Mlflow shows the following data in the experiment tab. As seen below, training mse and val_mse are lower on the “higher quality” model. 

![Screen Shot 2022-07-06 at 10.25.07 PM.png](doc/Screen_Shot_2022-07-06_at_10.25.07_PM.png)