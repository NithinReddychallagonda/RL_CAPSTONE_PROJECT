
# An Efficient Framework for Crop Recommendation Under Dynamic Field Conditions Using Deep Reinforcement Learning

<p align="center">
  <b>A Comparative Study of DQN, PPO, A2C, and QR-DQN</b>
</p>

## 1. Project Overview

Agriculture plays an important role in food production and economic development. Selecting a suitable crop based on soil and environmental conditions is an important challenge for improving agricultural productivity.

Traditional machine learning approaches generally learn the relationship between soil parameters and crop labels. This project explores a different approach by applying Deep Reinforcement Learning (DRL) to crop recommendation.

The proposed framework uses **Deep Q-Network (DQN)** as the primary algorithm and compares it with three baseline reinforcement learning algorithms:

- Proximal Policy Optimization (PPO)
- Advantage Actor-Critic (A2C)
- Quantile Regression Deep Q-Network (QR-DQN)

All four algorithms are trained and evaluated using the same crop recommendation dataset, feature representation, and custom Gymnasium environment.

The main objective is to investigate how different reinforcement learning approaches learn crop selection decisions and compare their performance using accuracy, macro F1-score, reward, convergence, sample efficiency, and inference time.

## 2. Problem Statement

Crop recommendation requires analysing multiple agricultural parameters, including soil nutrients and environmental conditions.

A suitable recommendation system should be able to understand the relationship between these parameters and the corresponding crop categories.

The main problem addressed in this project is:

**How can Deep Reinforcement Learning algorithms be applied and comparatively evaluated for crop recommendation using soil and environmental information?**

To investigate this problem, the project formulates crop recommendation as a one-step reinforcement learning task in which an agent receives an agricultural observation, selects a crop, and receives a reward based on whether its prediction matches the actual crop label.

## 3. Project Objectives

The major objectives are:

- Develop a reinforcement learning-based crop recommendation framework.
- Implement a custom environment using Gymnasium.
- Apply DQN as the main reinforcement learning model.
- Implement PPO, A2C, and QR-DQN as baseline models.
- Train all four algorithms using a consistent dataset and experimental setup.
- Compare the performance of the algorithms using multiple evaluation metrics.
- Analyse training behaviour, convergence, sample efficiency, and prediction performance.
- Identify the strengths and limitations of different reinforcement learning approaches for crop recommendation.

## 4. Dataset Description

### Dataset Source

The project uses the publicly available Crop Recommendation Dataset from Kaggle.

**Dataset:** atharvaingle/crop-recommendation-dataset

**Kaggle Link:**
https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset

The dataset is downloaded directly into the notebook using the `kagglehub` library. Manual downloading and uploading are not required.

### Input Features

The dataset contains seven agricultural input features.

| Feature | Description |
|---|---|
| Nitrogen (N) | Nitrogen content in the soil |
| Phosphorus (P) | Phosphorus content in the soil |
| Potassium (K) | Potassium content in the soil |
| Temperature | Temperature of the environment |
| Humidity | Relative humidity |
| pH | Soil pH value |
| Rainfall | Rainfall measurement |

### Target Variable

The target variable is the recommended crop label.

The categorical crop labels are converted into numerical class indices using `LabelEncoder`.

The action space is created based on the available crop classes in the dataset.

### Dataset Preprocessing

The following preprocessing steps are performed:

1. Download the dataset directly from Kaggle.
2. Load the CSV file using Pandas.
3. Rename the columns for better readability.
4. Remove records containing missing values in the selected features or target.
5. Remove duplicate records.
6. Encode the crop labels using LabelEncoder.
7. Separate the input features and target labels.
8. Split the dataset into training, validation, and testing sets.
9. Normalize the input features using MinMaxScaler.

### Data Splitting

The dataset is divided using stratified sampling:

| Dataset | Percentage |
|---|---:|
| Training | 60% |
| Validation | 20% |
| Testing | 20% |

The training set is used for model learning.

The validation set is used to evaluate the models during training and select the best checkpoint.

The test set remains separate and is used only for final evaluation.

Feature normalization is performed using a scaler fitted only on the training data to avoid information leakage from the validation and test sets.

## 5. Proposed Methodology

The proposed framework follows a reinforcement learning workflow.

Unlike a conventional supervised classification pipeline, the crop recommendation task is represented through an agent and an environment.

The agent observes the agricultural input features and selects one crop from the available crop classes.

The environment evaluates the selected action against the actual crop label and returns a reward.

The agent uses this feedback during training to improve its crop selection policy.

### 5.1 Reinforcement Learning Components

**Agent:**

The reinforcement learning algorithm responsible for selecting a crop.

**Environment:**

A custom Gymnasium environment that provides agricultural observations and evaluates crop selection actions.

**State:**

The seven normalized input features:

- Nitrogen
- Phosphorus
- Potassium
- Temperature
- Humidity
- pH
- Rainfall

**Action:**

The selected crop class from the available crop categories.

**Reward:**

The environment provides a reward based on the correctness of the selected crop.

### 5.2 Reward Function

The project uses a simple binary reward function:

| Condition | Reward |
|---|---:|
| Correct crop selection | +1 |
| Incorrect crop selection | 0 |

The reward is calculated by comparing the selected action with the actual crop label.

A correct decision receives a positive reward, while an incorrect decision receives zero reward.

Each episode contains only one crop selection. Therefore, the average episode reward corresponds to the crop-label accuracy.

This reward design allows the four reinforcement learning algorithms to be evaluated under the same decision-making objective.

## 6. Complete Project Flow

The overall project follows these stages:

```text
             START
               |
               v
       Download Kaggle Dataset
               |
               v
        Load Crop CSV File
               |
               v
        Data Preprocessing
               |
               v
     Remove Missing and Duplicate Rows
               |
               v
       Encode Crop Categories
               |
               v
       Train / Validation / Test Split
               |
               v
        Feature Normalization
               |
               v
       Create Gymnasium Environment
               |
               v
       Define State and Action Spaces
               |
               v
          Define Reward
               |
               v
       Initialize Four RL Algorithms
               |
       -----------------------------
       |       |       |           |
       v       v       v           v
      DQN     PPO     A2C        QR-DQN
       |       |       |           |
       -----------------------------
               |
               v
       Train Each RL Algorithm
               |
               v
       Evaluate on Validation Data
               |
               v
       Save Best Validation Models
               |
               v
       Evaluate on Unseen Test Data
               |
               v
       Calculate Evaluation Metrics
               |
               v
       Generate Comparison Graphs
               |
               v
       Compare Four RL Algorithms
               |
               v
             END
```

## 7. Reinforcement Learning Algorithms

Four reinforcement learning algorithms are implemented in this project.

### 7.1 Deep Q-Network (DQN)

DQN is the primary model used in the project.

It combines Q-learning with a deep neural network to estimate the expected value of available actions.

The model receives the current agricultural state and estimates Q-values for the available crop actions. The action with the highest estimated Q-value can be selected during deterministic prediction.

The implementation uses:

- Experience replay buffer
- Target network
- Epsilon-greedy exploration
- Neural network architecture with two hidden layers
- Mini-batch training

The DQN configuration uses:

| Parameter | Value |
|---|---:|
| Learning rate | 0.0003 |
| Buffer size | 50,000 |
| Learning starts | 2,000 |
| Batch size | 256 |
| Training frequency | 4 |
| Gradient steps | 2 |
| Exploration fraction | 0.20 |
| Final exploration epsilon | 0.02 |
| Target update interval | 1,000 |
| Hidden layers | 256, 256 |

### 7.2 Proximal Policy Optimization (PPO)

PPO is a policy-gradient reinforcement learning algorithm.

It improves the policy through controlled updates using a clipped objective function. This helps prevent excessively large policy changes during training.

PPO is included as a baseline to compare policy-based learning against the value-based DQN approach.

### 7.3 Advantage Actor-Critic (A2C)

A2C is an actor-critic algorithm that combines two components:

- Actor: Selects actions based on the learned policy.
- Critic: Estimates the value of the current state.

The advantage estimate helps guide policy updates.

A2C is included to investigate how actor-critic learning performs in the crop recommendation environment.

### 7.4 Quantile Regression Deep Q-Network (QR-DQN)

QR-DQN is a distributional extension of DQN.

Instead of estimating only a single expected return, it learns quantiles representing the distribution of possible returns.

This provides another value-based learning approach for comparison with standard DQN.

The implementation uses 50 quantiles and includes experience replay and a target network.

## 8. Model Training Strategy

Each algorithm is trained independently using the same training data and custom environment.

The notebook uses the following training configuration:

| Algorithm | Total Training Steps |
|---|---:|
| DQN | 150,000 |
| PPO | 150,000 |
| A2C | 150,000 |
| QR-DQN | 150,000 |

The models are evaluated at regular training checkpoints.

Validation accuracy is calculated every 5,000 training steps.

The best-performing validation checkpoint is saved for each algorithm.

The selected model is then used for final evaluation on the untouched test dataset.

A random seed of 42 is used for reproducibility.

## 9. Model Evaluation Metrics

The four algorithms are evaluated using multiple performance measures.

### Accuracy

Measures the percentage of correctly recommended crops.

### Macro F1-Score

Calculates the F1-score independently for each crop category and averages the results equally across classes.

### Average Cumulative Reward

Measures the average reward received per one-step episode.

### Reward Standard Deviation

Measures the variation in rewards across the evaluation records.

### Convergence Speed

Measures the earliest validation checkpoint from which the validation accuracy remains above the selected threshold through the end of training.

### Sample Efficiency

Measures the first training checkpoint at which the validation accuracy reaches the selected threshold.

### Inference Time

Measures the average time required to make one crop recommendation.

### Training Time

Measures the time taken to train each reinforcement learning algorithm.

The final evaluation uses an 85% validation accuracy threshold for convergence and sample-efficiency calculations.

## 10. Experimental Results

The following values are taken from the notebook's recorded final evaluation.

### 10.1 Performance Comparison

| Algorithm | Test Accuracy | Macro F1 |
|---|---:|---:|
| DQN | 99.32% | 0.9932 |
| PPO | 92.50% | 0.9078 |
| A2C | 87.27% | 0.8420 |
| QR-DQN | 98.86% | 0.9886 |

### 10.2 Additional Evaluation Results

| Metric | DQN | PPO | A2C | QR-DQN |
|---|---:|---:|---:|---:|
| Average Reward | 0.9932 | 0.9250 | 0.8727 | 0.9886 |
| Reward Std. | 0.0824 | 0.2637 | 0.3337 | 0.1061 |
| Convergence Steps | 5,000 | 60,000 | 105,000 | 5,000 |
| Sample Efficiency | 5,000 | 60,000 | 105,000 | 5,000 |
| Inference Time (ms) | 1.0137 | 1.5212 | 0.7278 | 0.4142 |
| Training Time (seconds) | 280.02 | 221.40 | 231.41 | 300.71 |

### 10.3 Results Analysis

Based on the recorded test results:

- DQN achieved the highest test accuracy of 99.32%.
- QR-DQN achieved a closely comparable test accuracy of 98.86%.
- PPO achieved 92.50% test accuracy.
- A2C achieved 87.27% test accuracy.
- DQN and QR-DQN reached the selected validation threshold earlier than PPO and A2C.
- QR-DQN recorded the lowest inference time among the four models.

These results describe the experiment conducted using the selected dataset, environment, hyperparameters, and random seed. They should not be interpreted as guaranteed performance on real agricultural fields.

## 11. Result Visualizations

The notebook generates ten graphs to analyse training and validation behaviour.

The generated figures include:

1. DQN validation accuracy learning curve.
2. Training success-rate comparison across four algorithms.
3. DQN training reward histogram.
4. DQN training reward box plot.
5. Training steps versus mean reward scatter plot.
6. Validation accuracy heatmap for four algorithms.
7. DQN training reward violin plot.
8. DQN training reward empirical cumulative distribution.
9. DQN positive and zero training reward distribution.
10. DQN cumulative training reward area chart.

All figures are generated using training rollout rewards and/or validation checkpoints. The held-out test results are not used to generate these training and validation graphs.

## 12. Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Graph generation |
| Seaborn | Heatmap visualization |
| Scikit-learn | Data splitting, encoding, normalization and evaluation metrics |
| KaggleHub | Direct Kaggle dataset download |
| Gymnasium | Custom reinforcement learning environment |
| Stable-Baselines3 | DQN, PPO and A2C implementation |
| SB3-Contrib | QR-DQN implementation |
| Jupyter Notebook | Development and experimentation |
| Google Colab | Cloud-based execution environment |

## 13. Project Structure

```text
Crop-Recommendation-Using-Deep-RL/
│
├── RL_Capstone_project.ipynb
│
├── README.md
│
├── requirements.txt
│
└── rl_research_results/
    │
    ├── DQN_best_validation.zip
    ├── PPO_best_validation.zip
    ├── A2C_best_validation.zip
    ├── QR_DQN_best_validation.zip
    │
    ├── final_8_metrics_and_accuracy.csv
    │
    ├── DQN_training_checkpoints.csv
    ├── PPO_training_checkpoints.csv
    ├── A2C_training_checkpoints.csv
    ├── QR-DQN_training_checkpoints.csv
    │
    ├── train_feature_ranges.csv
    ├── train_crop_ranges.csv
    │
    └── Generated Graphs
```

The results directory is created during notebook execution. The saved model and CSV files can be added to the repository if required.

## 14. How to Run the Project

### Option 1: Google Colab

1. Open Google Colab.
2. Upload or open the project notebook.
3. Run the cells in sequence.
4. Allow the notebook to install the required libraries.
5. The dataset will be downloaded directly using KaggleHub.
6. Execute the training cells for all four algorithms.
7. View the evaluation results and generated graphs.

### Option 2: Run Locally

Clone the repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
```

Move into the project directory:

```bash
cd your-repository-name
```

Install the dependencies:

```bash
pip install "stable-baselines3[extra]>=2.3,<3" "sb3-contrib>=2.3,<3" "gymnasium>=0.29" kagglehub scikit-learn seaborn pandas numpy matplotlib
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebook and execute the cells in order.

KaggleHub may require Kaggle authentication depending on your execution environment.

## 15. Research Contributions

The project focuses on the comparative investigation of different reinforcement learning approaches for crop recommendation.

The main contributions are:

- A custom Gymnasium environment for crop-label decision-making.
- Implementation of DQN as the main reinforcement learning model.
- Comparative implementation of PPO, A2C, and QR-DQN.
- A common experimental setup for comparing four different RL approaches.
- Validation-based model checkpoint selection.
- Evaluation using accuracy, macro F1-score, reward, convergence, sample efficiency, inference time, and training time.
- Training and validation visualizations for analysing model behaviour.

The comparison provides an experimental understanding of how value-based, policy-based, actor-critic, and distributional value-learning methods behave in this crop recommendation task.

## 16. Limitations

The current implementation has some limitations:

- The environment uses a one-step crop-label decision rather than a multi-season crop rotation simulation.
- Soil nutrient values are input features and are not dynamically updated after crop selection.
- The reward is based on matching the actual crop label.
- The experiment uses a single dataset and a fixed random seed.
- Results may vary with different datasets, seeds, and hyperparameters.
- High test accuracy on this dataset does not establish real-world agricultural effectiveness.

These limitations provide opportunities for further research.

## 17. Future Scope

The framework can be extended in several ways:

- Develop a multi-season agricultural environment.
- Introduce crop-specific soil nutrient consumption and replenishment.
- Include previous crop history and crop rotation constraints.
- Integrate weather forecasts and seasonal information.
- Use real-time soil sensor measurements.
- Explore more advanced reward functions.
- Evaluate the models across multiple datasets and random seeds.
- Validate the recommendations with agricultural experts and field experiments.

## 18. Conclusion

This project explores the application of Deep Reinforcement Learning to crop recommendation under agricultural input conditions.

DQN is implemented as the primary model, while PPO, A2C, and QR-DQN are used as baseline algorithms for comparative evaluation.

The experimental results show that DQN achieved the highest test accuracy among the four implemented algorithms, while QR-DQN also demonstrated strong performance.

The project provides a practical comparison of different reinforcement learning strategies and establishes a foundation for future research involving more realistic dynamic agricultural environments.


## Acknowledgement

This project was developed as part of academic research to explore the application of Artificial Intelligence and Deep Reinforcement Learning in agricultural decision-making.

If you find this project useful, consider giving the repository a ⭐ star.

