# Project Overview: Federated Learning Benchmark

This study aims to evaluate the performance and robustness of Federated Learning through three distinct architectural models: Logistic Regression (LR), Multi-Layer Perceptron (MLP), and Support Vector Machine (SVM).

The objective is to analyze how these models handle the real-world challenges of digital healthcare by subjecting them to three progressive experimental scenarios:


## Data presentation
The data came from the UCI website:
https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

There are 22 featrues including the target. Here is the feature list:
| Feature                  | Description                                                                                                                                                                                                 |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ID                       | Patient ID                                                                                                                                                                                                 |
| **Target**: Diabetes_binary          | 0 = no diabetes, 1 = prediabetes or diabetes (Target)                                                                                                                                                     |
| HighBP                   | 0 = no high BP, 1 = high BP                                                                                                                                                                               |
| HighChol                 | 0 = no high cholesterol, 1 = high cholesterol                                                                                                                                                             |
| CholCheck                | 0 = no cholesterol check in 5 years, 1 = yes cholesterol check in 5 years                                                                                                                              |
| BMI                      | Body Mass Index                                                                                                                                                                                            |
| Smoker                   | Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes] 0 = no, 1 = yes                                                                                               |
| Stroke                   | (Ever told) you had a stroke. 0 = no, 1 = yes                                                                                                                                                              |
| HeartDiseaseorAttack     | coronary heart disease (CHD) or myocardial infarction (MI) 0 = no, 1 = yes                                                                                                                              |
| PhysActivity             | physical activity in past 30 days - not including job 0 = no, 1 = yes                                                                                                                                    |
| Fruits                   | Consume Fruit 1 or more times per day 0 = no, 1 = yes                                                                                                                                                     |
| Veggies                  | Consume Vegetables 1 or more times per day 0 = no, 1 = yes                                                                                                                                                |
| HvyAlcoholConsump        | Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) 0 = no, 1 = yes                                                                          |
| AnyHealthcare            | Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc. 0 = no, 1 = yes                                                                                       |
| NoDocbcCost              | Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? 0 = no, 1 = yes                                                                                   |
| GenHlth                  | Would you say that in general your health is: scale 1-5 (1 = excellent, 2 = very good, 3 = good, 4 = fair, 5 = poor)                                                                                     |
| MentHlth                 | Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? scale 1-30 days   |
| PhysHlth                 | Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? scale 1-30 days           |
| DiffWalk                 | Do you have serious difficulty walking or climbing stairs? 0 = no, 1 = yes                                                                                                                                |
| Sex - **Sensitive attribute**                     | 0 = female, 1 = male                                                                                                                                                                                       |
| Age                      | 13-level age category (_AGEG5YR see codebook) 1 = 18-24, 9 = 60-64, 13 = 80 or older                                                                                                                       |
| Education                | Education level (EDUCA see codebook) scale 1-6 (1 = Never attended school or only kindergarten, 6 = College 4 years or more)                                                                             |
| Income                   | Income scale (INCOME2 see codebook) scale 1-8 (1 = less than $10,000, 5 = less than $35,000, 8 = $75,000 or more)                                                                                           |

<br>
<br>


## Requirements

Some python package are required to run the project, here are the cmd to execute to be able to run the entire project:

- 'python -m venv .venv'    to create a virtual environement, ensure that the python version is 3.10
- 'pip install -e .'        inside the fluke package folder
- 'pip install ucimlrepo'
- 'pip install flwr'
- 'pip install -U "flwr[simulation]"'
- 'pip install opacus' 



## Experimental Scenarios

- 1. IID Baseline (Uniform): Establishing reference performance under ideal and balanced data distribution conditions.
- 2. Heterogeneity Study ($\beta$): Analyzing model resilience against data fragmentation and imbalance across clients (Non-IID).
- 3. Differential Privacy (DP-FedAvg): Evaluating the trade-off between protecting patient privacy (via Gaussian noise injection) and diagnostic accuracy.
- 4. Vertical Federated Learning, with and without Differencial Privacy

By conducting this benchmark, we measure the impact of each configuration on global utility, prediction fairness, and associated communication costs.


<br>

## **To run all the scenario :** 
```bash 
python main_cmd.py
```
<br>

##  Evaluation Framework & Metrics

To assess the performance of each model across our three scenarios (IID, Non-IID, and Privacy), we monitor three distinct categories of metrics:

### 1. Utility Metrics

These metrics measure the diagnostic performance of the global model on the testing set.
- Accuracy: The percentage of correct predictions. While useful, it can be misleading due to the class imbalance (86% healthy).
- F1-Score (Macro): The harmonic mean of Precision and Recall. This is our primary metric, as it treats both classes (Diabetic vs. Healthy) with equal importance.
- Precision: Measures the model's ability to avoid false positives (predicting diabetes when the patient is healthy).
- Recall: Measures the model's ability to detect all diabetic cases (avoiding false negatives).

### 2. Fairness Metrics

These metrics evaluate whether the model treats patients equitably, specifically comparing outcomes between Male and Female groups.

- Statistical Parity Difference (SDP): Measures the difference in the probability of being predicted as "Diabetic" between the two groups. <br> $SDP = |P(\hat{Y}=1 | \text{Sex}=1) - P(\hat{Y}=1 | \text{Sex}=0)|$

- Equal Opportunity Difference (EOD): Measures the difference in True Positive Rates between the groups. It ensures the model is equally good at detecting diabetes for both sexes. <br> $EOD = |P(\hat{Y}=1 | Y=1, \text{Sex}=1) - P(\hat{Y}=1 | Y=1, \text{Sex}=0)|$

### 3. Cost Metrics

These metrics quantify the resources required to train the model in a federated environment.

- Training Time: The total wall-clock time required to complete all communication rounds. It reflects the computational complexity of the model (e.g., MLP vs. LR).

- Communication Cost: The cumulative volume of data (in MB) exchanged between the clients and the server. This is critical for assessing the feasibility of the model on bandwidth-constrained devices.

<br>
<br>
<br>

## Experimental Scenario 1: IID Baseline (Uniform)

This scenario serves as the **baseline** for our federated learning study. The objective is to evaluate model performance under ideal distribution conditions before introducing heterogeneity or privacy constraints.

### Scenario Description
In this configuration, the **Diabetes** dataset is distributed in an **IID** (Independent and Identically Distributed) manner among participants.

* **Number of clients:** 10
* **Distribution:** Uniform (each client possesses a representative sample of the global population).
* **Class Proportions:** Approximately 86% Class `0` (Healthy) and 14% Class `1` (Diabetes) per client.
* **Aggregation Algorithm:** `FedAvg` (Federated Averaging).

### Evaluated Models
Three architectures of varying complexity are compared to identify the best trade-off between utility and cost:

| Model | Type | Description |
| :--- | :--- | :--- |
| **LR** | Linear | Logistic Regression (21 inputs, 2 outputs). A lightweight model used to establish the performance baseline. |
| **MLP** | Non-linear | Multi-Layer Perceptron. Capable of capturing complex relationships between health indicators. |
| **SVM** | Maximum Margin | Support Vector Machine. Evaluates the robustness of class separation in a federated setting. |

<br>

## **Results and metrics of this scenario :** ``scenario_1.ipynb``

<br>
<br>
<br>

## Experimental Scenario 2: Heterogeneity Study ($\beta$)

The objective is to evaluate model robustness when faced with data fragmentation. We utilize three levels of concentration based on the Dirichlet distribution:

| Parameter | Difficulty Level | Statistical Description |
| :--- | :--- | :--- |
| **$\beta = 0.5$** | **High** | Strong heterogeneity: some clients become "experts" in a single class. |
| **$\beta = 5$** | **Medium** | Moderate heterogeneity: imbalance is present but manageable. |
| **$\beta = 100$** | **None** | Quasi-IID: uniform data distribution across the 10 clients. |

<br>

## **Results and metrics of this scenario :** ``scenario_2.ipynb``

<br>
<br>
<br>

## Experimental Scenario 3: Differential Privacy (DP-FedAvg)

This scenario introduces **Differential Privacy (DP)** into the learning protocol. The objective is to mathematically guarantee that the participation of a specific patient in the dataset cannot be inferred by observing the updates of the global model.

### The DP Mechanism
For each client, two critical operations are added during local training before sending the gradients:
1. **L2-Norm Clipping:** Gradients are capped by a threshold ($C=1.0$) to limit the maximum influence of any single individual.
2. **Noise Addition:** Gaussian noise, proportional to the $\sigma$ parameter, is injected into the aggregated gradients.

### Noise Levels and Protection
We test three values of the `noise_multiplier` ($\sigma$) to measure the trade-off between utility, fairness, and privacy:

| Parameter ($\sigma$) | Privacy Level | Theoretical Impact | Risk to the Model |
| :--- | :--- | :--- | :--- |
| **0.1** | **Low** | Symbolic protection. Maximum utility. | Low risk of performance degradation. |
| **0.5** | **Standard** | Balanced trade-off used in research. | Appearance of fluctuations in convergence. |
| **1.2** | **High** | Robust protection. | Risk of F1-Score collapse (Signal drowned in noise). |

<br>

## **Results and metrics of this scenario :** ``scenario_3.ipynb``




## Experimental Scenario 4: Vertical Federated Learning (with and without Differencial Privacy)

This scenario introduce the concept of Vertical Federated Learning (VFL). Instead of split our datasets by sampling batch of rows among clients like in Horizontal Federated Learning, the sampling is done now on features. It means that all clients have different features.


### Architecture explaination
- dataset.py : load the dataset into a pandas.Dataframe
- split.py : Data partioning (features) among clients, formated as pytorch Tensor.
- models.py : define the local models (clients) and the server one.
- server.py : server implementation - loss calculation and training based on the embeddings received from the clients.
- startegy.py : define the strategy used to round coordination between clients and server.
- metrics.py : metrics update.

<br>
<br>

The models used for every client is the same, a simple MLP. For the serveur part we also used a MLP to aggregate gradient and embeddings.


<br>

Each client train its model locally before sending their embeddings to the server. 
Then the server aggregates embeddings (concatenation), calculate the loss and update the model. 
The updates are sent back to the clients.


<br>
<br>

We decide to create 4 clients with the following features:
- client1: ["HighBP","HighChol", "CholCheck"],
- client2: ["Stroke","HeartDiseaseorAttack","PhysActivity", "AnyHealthcare", "DiffWalk", "GenHlth", "PhysHlth", "MentHlth"],
- client3: ["Fruits","Veggies","HvyAlcoholConsump"], 
- client4: ["Sex","Smoker", "Age", "Education", "Income", "BMI", "NoDocbcCost"]     # sensitive informations


<br>
<br>

Note that this architecture (clients only share the embeddings), allow to keep data privacy and ensure confidentiality.
To improve the data confidentiality it is possible to add differential privacy.
Then, to assess the models (with and without DP), it is possible to use fairness metrics like EOD and SDP.
Note that our results are very bad with Vertical Federated Learning, in particular if we focus on TP, TN, FP, and FN.
It become worse with Differencial Privacy implementation.

<br>
<br>

Here the differential privacy mechanism can be described as follow:
- Clipping of client's embeddings.
- Add of Gaussian noise to these embeddings before transmission to the server.

### Run instructions
To be able to run the trainings, execute:
- python .\run_experiment_vfl.py
- python .\run_experiment_vfl_dp.py 
    - Note that you should modify the config.yml file (vfl_part/config/config.yml) to test different configurations of differential privacy.


## **Results and metrics of this scenario :** ``vlf_part/analysis.ipynb``
