# Projet Fluke : Federated & Decentralized Learning

Thanks to the original gitlab repository we create our own repository, to better understand how works the fluke package. 

We achieve all the tasks step by step, by adding each config file in the order of our tasks.

---

## Install Fluke and the underlying Python environment : 

To configure the environment `fluke310` :

1.  **Create environment** : 
    ```bash
    python -m venv fluke310
    ```
2.  **Activate environment** :
    * **Windows** : 
    ```bash
    fluke310/Scripts/activate
    ```

    * **Linux/macOS** : 
    ```bash
    source fluke310/bin/activate
    ```

---

## Run experiment with MNIST dataset

* **IID** :

    ```bash
    fluke federation config/exp-iid.yaml config/fedavg.yaml
    ```
     **Results :** `/runs/fluke_mnist_iid`


* **non IID** :

    ```bash
    fluke federation config/exp-non-iid.yaml config/fedavg.yaml
    ```
     **Results :** `/runs/fluke_mnist_non_iid`


## Use Fluke functionalities to save global metrics, and local metrics by client :
    
All the metrics are saved in the `runs` folder after an experiment.

## Plot the performance of the global FL model, and the clients local models : 

Performance plots of the IID and non-IID experiment are detailed in the `4_plot_performance.ipynb` notebook.

- **Results** : `/plots/performance_iid.png` and `/plots/performance_non_iid.png`


## Apply a method that handles clients data heterogeneity (SCAFFOLD), and compare its performance with FedAvg :

- First run the experiment with the SCAFFOLD method :

```bash
fluke federation config/exp-non-iid.yaml config/scaffold.yaml
```

Then the performance comparison is done in the `4_plot_performance.ipynb` notebook.

- Results : `/plots/comparison_fedavg_scaffold.png`

##  Deploy a decentralized FL system without any server : 
- Followings files are required :<br> 
`fluke_package/fluke/run.py`<br>
`fluke_package/fluke/algorithms/decentralized.py` <br>
`fluke_package/config/decentralized_fedavg.yaml`

- Then run : 

```bash
fluke decentralized config/exp.yaml config/decentralized_fedavg.yaml
```
- Results : `/runs/fluke_mnist_non_iid_decentralized`

## Integrate a tabular dataset with different model architectures (M1, M2, M3)

We implement the tabular dataset **Adult**.

- Followings files are required :<br> 
 `fluke/data/datasets.py` <br>
 `fluke/nets.py`

-  Architectures : <br>
 **M1** : Adult_LogReg (Logistic Regression) <br>
 **M2** : Adult_SVM (Support Vector Machine) <br>
 **M3** : Adult_MLP (Multi-Layer Perceptron)

### Commands and results : 

| Model | Type | Command | Results |
| :--- | :--- | :--- | :--- |
| **M1** | Centralized | `fluke centralized config/exp-adult.yaml config/fedavg-adult.yaml` | `runs/adult_LR_centralized` |
| | Federation | `fluke federation config/exp-adult.yaml config/fedavg-adult.yaml` | `runs/adult_LR_fedavg` |
| | Decentralized | `fluke decentralized config/exp-adult.yaml config/decentralized_fedavg_adult.yaml` | `runs/adult_LR_decentralized` |
| **M2** | Centralized | `fluke centralized config/exp-adult.yaml config/fedavg-adult-svm.yaml` | `runs/adult_SVM_centralized` |
| | Federation | `fluke federation config/exp-adult.yaml config/fedavg-adult-svm.yaml` | `runs/adult_SVM_fedavg` |
| | Decentralized | `fluke decentralized config/exp-adult.yaml config/decentralized_fedavg_adult_svm.yaml` | `runs/adult_SVM_decentralized` |
| **M3** | Centralized | `fluke centralized config/exp-adult.yaml config/fedavg-adult-mlp.yaml` | `runs/adult_MLP_centralized` |
| | Federation | `fluke federation config/exp-adult.yaml config/fedavg-adult-mlp.yaml` | `runs/adult_MLP_fedavg` |
| | Decentralized | `fluke decentralized config/exp-adult.yaml config/fedavg-adult-mlp-decentralized.yaml` | `runs/adult_MLP_decentralized` |


## Evaluate the performance of the different models

The graphs and comparative analyses are available in the `9_plot_performance.ipynb` notebook.

## Test different FL hyperparameters


**Hyperparameters** tested :

- Client ratio : 20, 40, 60, 80, 100
- Local epochs : 5, 10, 20, 30

**To run :**

Modify `eligible_perc` in the exp-non-iid.yaml to change the client ratio. <br>
Modify `local_epochs` in the fedavg.yaml to change the local epochs of each client. 

**Results :**

The graphs and comparative analyses are available in the `10_plot_performance.ipynb` notebook.





