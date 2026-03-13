# Project Overview: Federated Learning Benchmark

This study aims to evaluate the performance and robustness of Federated Learning through three distinct architectural models: Logistic Regression (LR), Multi-Layer Perceptron (MLP), and Support Vector Machine (SVM).

The objective is to analyze how these models handle the real-world challenges of digital healthcare by subjecting them to three progressive experimental scenarios:

## Experimental Scenarios

- 1. IID Baseline (Uniform): Establishing reference performance under ideal and balanced data distribution conditions.
- 2. Heterogeneity Study ($\beta$): Analyzing model resilience against data fragmentation and imbalance across clients (Non-IID).
- 3. Differential Privacy (DP-FedAvg): Evaluating the trade-off between protecting patient privacy (via Gaussian noise injection) and diagnostic accuracy.

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

