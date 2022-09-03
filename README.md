# Adversarial Disentangle Bias for Recommender System（ADS）
## 1. Abstract
User behavior data like rating and click has been widely used to build a personalizing model for recommender systems. However, many unflattering factors (e.g., popularity, ranking position, users' selection) that significantly influence users' behaviors have always been ignored, crucially affecting the performance of the learned recommendation model. Existing work on unbiased recommendation mainly manages the bias from the perspective of the sample's granularity, such as sample-reweighting (e.g., inverse propensity score, learning to reweight) and creating pseudo samples (e.g., imputation model, doubly robust). However, rare work studies the problem from the model's perspective, and a bias-insensitive model that accounts for the mixed biases is even lacking. Towards this research gap, in this paper, we propose a novel adversarial disentangling bias framework, decoupling the bias attributes from the learned representations. It is achieved through an adversarial game. Notably, a bias-identifier tries to predict the bias from the learned representation, and an attacker disturbs the representation, aiming to deteriorate the identifier's performance. The adversarial bias disentanglement is robust towards multi-bias and eliminates the sophisticated configuration of sample reweighting methods. Finally, extensive experiments on the benchmark and synthetic datasets have demonstrated the effectiveness of the proposed approach against a wide range of recommendation unbiasing methods.

## 2. Overall framework
<img src='https://user-images.githubusercontent.com/31196524/151699502-ac6b2484-274e-4074-8ee9-9bbe43cf69af.png' width="80%">

## 3. Runtime Environment

* System:Linux dell-PowerEdge-R730

* CPU: Intel(R) Xeon(R) CPU E5-2630 v4 @ 2.20GHz

* CPU-Memory:25G

* GPU:NVIDIA Corporation GV100 [TITAN V] (rev a1)

* GPU-Memory:12G

* Pytorch: 1.7.0

* CUDA:10.1

## 4. Usage
### 4.1 Run ADS_S, ADS_Pos (selection bias/ position bias)
Go the ads/ directory and run the following command:

- For dataset Yahoo!R3:

```shell
python ADS_S.py --dataset yahooR3
```

- For dataset Coat:

```shell
python ADS_S.py --dataset coat
```

- For dataset Simulation:

```shell
python ADS_Pos.py --dataset simulation
```
### 4.2 Run ADS_P (popularity)
Go the ads_p/ directory 

(1) Configure some training parameters through the **test.yaml**.

```shell
Set **mode = 1**, **norm = 1e-4**.
```

(2) Run the following command:

```shell
python run_recbole.py --dataset ml-1m --model BPR --config_files test.yaml
```

