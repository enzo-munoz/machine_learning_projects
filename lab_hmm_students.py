#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install hmmlearn')


# In[4]:


import numpy as np
import pandas as pd
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import nltk
from sklearn.model_selection import train_test_split
import pprint, time

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[6]:


from IPython.core.display import Image, display


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [6,6]
plt.rcParams['font.size'] = 12


# ## Degree in Data Science and Engineering, group 96
# ## Machine Learning 2
# ### Fall 2024
# 
# &nbsp;
# &nbsp;
# &nbsp;
# # Lab 8. Hidden Markov Models
# 
# &nbsp;
# &nbsp;
# &nbsp;
# 
# **Jose Manuel de Frutos Porras and David Martínez Rubio**
# 
# **Adapted from a notebook by Ignacio Peis**
# 
# Dept. of Signal Processing and Communications
# 
# &nbsp;
# &nbsp;
# &nbsp;
# 
# 

# # 1. Introduction
# 
# &nbsp;
# 
# **Markov Models** are one of the simplest ways to treat sequential data, by relaxing the i.i.d. (independent and identically distributed) assumption. The joint distribution for a sequence of observations is given by:
# $$ p(\mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_T) = p(\mathbf{x}_{1:T}) = \prod_{t=2}^T p(\mathbf{x}_t | \mathbf{x}_1, ..., \mathbf{x}_{t-1}) $$
# where we assume that each observation depends on all previous observations, which can be expressed by means of the product rule.
# This expression can be relaxed to obtain simpler models by establishing dependencies only with the previous observation (*first-order Markov chain*) or the two previous ones (*second order Markov chain*).
# 
# Suppose we wish to build a model for sequences that is not limited by the Markov assumption to any order and yet that can be specified using a limited number of free parameters. We can achieve this by introducing additional latent variables to permit a rich class of models to be constructed out ot simple components, as we did with mixture distributions. For each observation $\textbf{x}_t$ , we introduce a corresponding latent variable $\textbf{z}_t$, which might be of different type or dimensionality to the observed variable. The joint distribution for this model is given by:
# $$ p(\mathbf{x}_{1:T}, \mathbf{z}_{1:T}) = p(\mathbf{z}_1 ) \left[ \prod_{t=2}^T p(\mathbf{z}_t|\mathbf{z}_{t-1}  ) \right] \prod_{t=1}^T p(\mathbf{x}_t |\mathbf{z}_t) $$
# If this latent variable is continous, we call the model a *state space model*. In contrast, if the latent variable is discrete, we have a **Hidden Markov Model** and is denoted as **state**, $\mathbf{z}_t \equiv s_t$.
# 
# 
# 

# # 2. Hidden Markov Models
# 
# &nbsp;
# 
# <img src='http://www.tsc.uc3m.es/~ipeis/ML2/hmm.png' width=400 />
# 
# &nbsp;
# 
# A Hidden Markov Model (HMM) consists of a discrete-time, discrete-state Markov chain (first-order Markov discrete model), with hidden states $s_t \in \{ 1, ..., L \}$, plus an **observation** model $p(\mathbf{y}_t | s_t)$. The joint distribution has the form:
# $$ p(\mathbf{y}_{1:T}, s_{1:T}) = p(s_1 ) \left[ \prod_{t=2}^T p(s_t|s_{t-1}  ) \right] \prod_{t=1}^T p(\mathbf{y}_t |s_t) $$
# The probabilities of each observation $p(\mathbf{y}_t |s_t)$ given the state is called **emission**. If the data is continuous, it follows a Gaussian distribution:
# $$ p(\mathbf{y}_t |s_t) = \mathcal{N} (\mathbf{y}_t | \boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i) $$
# Hence, for each observation $\mathbf{y}_t$, we have a set of $L$ possible Gaussians, depending on the state $s_t$ that generated the sample. In other terms, a HMM can be seen as a sequential GMM, where the component of each sample depends on the previous one.
# The probabilities of the discrete hidden states are defined by the transition matrix, $\mathbf{A}$. Each of its elements $a_{ij}$ defines the probability of state $j$ given that the previous state was $i$. Further, we need a probability for the initial states (as they do not have previous states). This probabilities are called $\boldsymbol{\pi}$.
# 
# - $ S = \{ s_1, s_2, ..., s_T : s_t \in 1, ..., L \}  $: hidden state sequence.
# - $ Y = \{ \mathbf{y}_1, \mathbf{y}_2, ..., \mathbf{y}_T : \mathbf{y}_t \in \mathbb{R}^M \}  $: observed continuous sequence.
# - $ \mathbf{A} = \{ a_{ij}: a_{ij} = p(s_{t+1}=j | s_t = i \} $: state transition probabilities.
# - $ \mathbf{B} = \{ b_{i}: p_{b_i}(\mathbf{y}_t) = p(\mathbf{y}_t | s_t = i \} $: observation emission probabilities.
# - $ \boldsymbol{\pi} = \{ \pi_i: \pi_i = p(s_1=i) \} $: initial state probability distribution.
# - $ \boldsymbol{\theta} = \{ \mathbf{A}, \mathbf{B}, \boldsymbol{\pi} \} $: model parameters.
# 

# ## 2.1. Inference in HMMs
# 
# In graphical models, given the joint distribution, we can perform probabilistic inference. This refers to the task of estimating unknown quantities from known quantities. i.e., the distributions for latent variables given the observations. Provided the HMM model, several inference scenarios might be needed to be solved.
# 
# ### 2.1.1. The Forwards Algorithm
# 
# The forwards algorithm is used with the aim at obtaining the evidence of a sequence, given the sequence $Y$ and the parameters $\theta$:
# $$p(Y | \theta) = \sum_S p(Y, S | \theta)$$
# As $S$ is hidden (we do not observe it), we have to infer its probabilities. This is solved by means of the **Forwards algorithm**, which compute the filtered marginals $p(s_t=j | \mathbf{y}_{1:t})$ recursively:
# $$ \alpha_t(i) = p(s_t=i | \mathbf{y}_{1:t}) $$
# 
# $$ \alpha_1(i) = \pi_i p_{b_i}(\mathbf{y}_1) $$
# $$ \alpha_t(i) = \left( \sum_{j=1}^L \alpha_{t-1}(j) a_{ji} \right) p_{b_i}(\mathbf{y}_t) $$
# 
# ### 2.1.2. The Forwards-backwards Algorithm
# 
# The goal is to estimate the probability for state $s_t$ at time $t$, given a sequence of observations $Y$ and the parameters of the model $\theta$. For that purpose, we compute the smoothed marginals, $p(s_t=j | \mathbf{y}_{1:T})$ using offline inference (we know future observations for each time $t$):
# $$ \gamma_t(j) = p(s_t | y_{1:T}) = \alpha_t(j) \beta_t(j) $$
# where:
# $$ \beta_T(i) = 1 $$
# $$ \beta_t(i) = \sum_j a_{ij} P_{b_j} (\mathbf{y}_{t+1}) \beta_{t+1}(j)  $$
# 
# ### 2.1.3. The Viterbi Algorithm
# 
# 
# In this case, the objective is obtaining the optimal entire sequence of states given the sequence of observations $Y$ and the parameters $\theta$:
# $$ S^* = \underset{S}{argmax} \, p(S |Y, \theta) $$
# 

# ## 2.2. Learning in HMMs: EM by The Baum-Welch algorithm
# 
# In graphical models, we denote as *learning* the computation of the parameters that explain a given set of observations. In HMMs, the parameters $\boldsymbol{\theta} = \{ \boldsymbol{\pi}, \textbf{A}, \textbf{B} \} $ can be obtained by using a special version of the EM algorithm, The Baum-Welch algorithm. The process is the same that for GMMs, with the difference that now we consider the time dependencies.
# 
# ### 2.2.1. E-step
# The expected complete data log likelihood is given by:
# 
# $$ Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{old}) = \sum_{i=1}^L \left( \mathbb{E}[N_i^1] \right) \log{\pi_i} + \\
# \sum_{i=1}^L \sum_{j=1}^L \mathbb{E} [N_{ij}] \log{a_{ij}} + \\
# \sum_{n=1}^N \sum_{t=1}^{T_n} \sum_{i=1}^L p(s_t=i | \mathbf{y}_t, \boldsymbol{\theta}^{old}) \log{p(\mathbf{y}_{n, t} | b_i)}
# $$
# 
# where the expected counts are given by:
# $$ \mathbb{E}[N_i^1] = \sum_{n=1}^N p(s_{n1}=i | \mathbf{y}_n, \boldsymbol{\theta}^{old}) = \sum_{n=1}^N \gamma_{n, 1}(i) \\
# \mathbb{E}[N_{ij}] = \sum_{n=1}^N \sum_{t=2}^{T_n} p(s_{n,t-1} = i, s_{n,t} = j | \mathbf{y}_n, \boldsymbol{\theta}^{old}))  = \sum_{n=1}^N \xi_{n,t}(i,j) \\
# \mathbb{E}[N_i] = \sum_{n=1}^N \sum_{t=1}^{T_n} p(s_{n,t} = i | s_{n,t} = j | \mathbf{y}_n, \boldsymbol{\theta}^{old}))
# $$
# 
# This function can be expressed in terms of the **smoothed node and edge marginals**:
# $$ \gamma_{n, t}(i) = p(s_t=i | \mathbf{y}_{n, 1:T_n}, \boldsymbol{\theta}) \\
# \xi_{n, t}(i, j)  = p(s_{t-1} = i, s_t=j | \mathbf{y}_{n, 1:T_n}, \boldsymbol{\theta}) = \alpha_t(i) a_{ij} P_{b_j}(\mathbf{y}_{t+1}) \beta_{t+1}(j)
# $$
# 
# These are the two variables that must be computed during the E-step, as we did with the *responsibilities* in the EM algorithm for GMMs.
# 
# ### 2.2.2. M-step
# Given all these computed variables, we can update the parameters $\boldsymbol{\theta} = \{ \boldsymbol{\pi}, \textbf{A}, \textbf{B} \} $ using:
# $$ \hat{\pi}_i= \frac{\mathbb{E}[N_i^1]}{N} $$
# $$ \hat{a}_{ij}= \frac{\mathbb{E}[N_{ij}]}{\sum_{j'} \mathbb{E}[N_{ij'}]} $$
# The emission model will depend on the type of the data. For a Gaussian emission, the parameters for each component will be:
# $$ \hat{\boldsymbol{\mu}}_i = \frac{\mathbb{E}[\bar{\mathbf{y}}_i]}{\mathbb{E}[N_i]} \qquad
# \hat{\boldsymbol{\Sigma}}_i = \frac{ \mathbb{E}[(\overline{yy})_i^T] - \mathbb{E}[N_i]\hat{\boldsymbol{\mu}}_i \hat{\boldsymbol{\mu}}_i^T }{ \mathbb{E}[N_i] }
# $$
# where the expected sufficient statistics are
# $$ \mathbb{E}[\bar{y}_i] = \sum_{n=1}^N \sum_{t=1}^{T_n} \gamma_{n,t}(i) \mathbf{y}_{n,t} $$
# $$ \mathbb{E}[\overline{yy}_i^T] = \sum_{n=1}^N \sum_{t=1}^{T_n} \gamma_{n,t}(i) \mathbf{y}_{n,t} \mathbf{y}_{n,t}^T $$
# 
# By the other hands, if the observations are discrete, the emissions follow a Mutinouilli model, and the parameters are a matrix $\mathbf{B}$ where:
# $$ \hat{b}_{im} = \frac{\mathbb{E} [M_{im}] }{\mathbb{E} [N_{i}]}$$
# where:
# $$ \mathbb{E} [M_{im}] = \sum_{n=1}^{N} \sum_{t=1}^{T_n} \gamma_{n,t}(i) \mathbb{I}(y_{n,t}=m) = \sum_{n=1}^{N} \sum_{t:y_{n, t}=l} \gamma_{n,t} (i)$$
# and $m$ correspond to the discrete observed varaible.

# ## 2.3. HMMs with fully observed data
# 
# There is another approach for HMMs where we assume that the states are also observed. In this case, inference of the states is not required, as we actually know the values of $S$. We can directly compute the Maximum Likelihood Estimation for $\mathbf{A}$ and $\boldsymbol{\pi}$ with the expressions:
# 
# $$ \hat{a}_{ij}= \frac{\mathbb{E}[N_{ij}]}{\sum_{j'} \mathbb{E}[N_{ij'}]} $$
# $$ \hat{\pi}_i= \frac{\mathbb{E}[N_i^1]}{N} $$
# 
# The transitions $\hat{a}_{ij}$ are obtained as the proportion of observed transitions from state $i$ to $j$ with respect to all the states $i$. The $\hat{\pi}_i$ initial probabilities are simply the proportion of states $i$ with respect to all the observed states.
# 
# In the second part of this notebook we will fit a HMM with fully observed discrete data. For the discrete case, the emissions are given by:
# $$ \hat{b}_{im} = \frac{N_{im}^X}{N_i} \qquad N_{im}^X = \sum_{n=1}^{N} \sum_{t=1}^{T_n}  \mathbb{I}(s_{n,t}=i, x_{n,t}=m)$$
# The emission $\hat{b}_{im}$ is basically the proportion of times a word $m$ appears with tag $i$ with respect to all the words with tag $i$.

# ## 2.4. HMMs in hmmlearn
# 
# [hmmlearn](https://hmmlearn.readthedocs.io/en/latest/api.html) is a python package that implements the HMM model and its principal applications of inference and learning, and follows scikit-learn API as close as possible, but adapted to sequence data. Its easy interface allows to create HMM models and optimize their parameters using the Baum-Welch algorithm in a few lines of code.
# 
# An example of the definition of a HMM, and sampling sequences from the model, is included below. For this model, observations $\textbf{y}_t$ are bidimensional, and we use $L=3$ states.

# In[8]:


np.random.seed(42)

# Number of states
L=3

# Define the HMM model
model = hmm.GaussianHMM(n_components=L, covariance_type="full")

# SETTING THE PARAMETERS
# Prior probabilities Pi
Pi = np.array([0.6, 0.3, 0.1])
model.startprob_ = Pi

# Transition matrix A
A = np.array([[0.7, 0.2, 0.1],
              [0.3, 0.5, 0.2],
              [0.3, 0.3, 0.4]])
model.transmat_ = A

# Parameters of the state-condition Gaussian densities p(yt|st)
mus = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]])
Sigmas = np.tile(np.identity(2), (3, 1, 1))
model.means_ = mus
model.covars_ = Sigmas

# Obtain a sequence of 100 samples
Y, S = model.sample(100)

f, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
ax[0].plot(Y[:, 0], label=r'$y_0$')
ax[0].plot(Y[:, 1], label=r'$y_1$')
ax[0].plot(model.means_[S][:, 0], ':', color='tab:blue', label=r'$\mu_{k0}$')
ax[0].plot(model.means_[S][:, 1], ':', color='tab:orange', label=r'$\mu_{k1}$')
ax[0].grid(alpha=0.4)
ax[0].legend(loc='best')
ax[0].set_title('samples')
ax[1].plot(S, '-o', color='tab:red')
plt.yticks([0, 1, 2], ['State 0', 'State 1', 'State 2'])
ax[1].grid(alpha=0.4)
ax[1].set_title('states')
plt.show()


# # 3. Experiments
# 
# ## 3.1. Human Activity Recognition (HAR)
# 
# In Human Activity Recognition, the states $s_t$ can be used to represent activities or gestures and the observations $\mathbf{y}_t$ to features extracted from video or sensors signals. Each activity defines a distribution for the inertial sensors. The signals of an accelerometer will change depending on whether the patient is lying or running, for example.
# 
# The [DaLiAc (Daily Life Activities) database](https://www.mad.tf.fau.de/research/activitynet/daliac-daily-life-activities/) consists of data from 19 subjects (8 female and 11 male, age 26 ± 8 years, height 177 ± 11 cm, weight 75.2 ± 14.2 kg, mean ± standard deviation (SD)) that performed 13 daily life activities.
# 
# Four sensors were used for data acquisition. Each sensor node was equipped with a triaxial accelerometer and a triaxial gyroscope. Data were sampled with 204.8 Hz Hz and were stored on SD card. The sensor nodes were placed on the left ankle, the right hip, the chest, and the right ankle.
# 

# In[9]:


data = pd.read_csv('HAR_data.csv', index_col=0)
activities = ['Sitting', 'Lying', 'Standing', 'Washing Dishes', 'Vacuuming', 'Sweeping', 'Walking',
              'Ascending stairs', 'Descending stairs', 'Treadmill running',
             'Bicycling on ergometer (50W)', 'Bicycling on ergometer (100W)', 'Rope jumping']
labels = data.iloc[:, -1].values
print(data.shape)
data.head()


# In[10]:


fs = 204.8
f, ax = plt.subplots(1, 2, figsize=(16,4), sharey=True)
t = np.arange(data.shape[0])/fs / 60
ax[0].plot(t,  data.iloc[:, 0:3])
ax[0].set_xlabel('min')
ax[0].grid(alpha=0.4)
ax[0].set_title('Whole sequence')
ax[0].set_ylabel(r'$m/s^2$')
ax[1].plot(t[:300]*60, data.iloc[:300, 0:3])
ax[1].set_xlabel('sec')
ax[1].grid(alpha=0.4)
ax[1].set_title('Segment with 300 observations')
plt.suptitle('Accelerometer data')

f, ax = plt.subplots(1, 2, figsize=(16,4), sharey=True)
ax[0].plot(np.arange(data.shape[0])/fs / 60,  data.iloc[:, 3:6])
ax[0].set_xlabel('min')
ax[0].grid(alpha=0.4)
ax[0].set_title('Whole sequence')
ax[0].set_ylabel(r'$deg/s$')
ax[1].plot(t[:300]*60, data.iloc[:300, 3:6])
ax[1].set_xlabel('sec')
ax[1].grid(alpha=0.4)
ax[1].set_title('Segment with 300 observations')
plt.suptitle('Gyroscope data')

plt.show()


# ### 3.1.1. Data Preprocessing
# 
# The sampling frequency is $f_s=204.8 Hz$. As it has no sense to guess the performed activity each $1/f_s=4.8 ms$, in HAR, segmentation of the data is performed using windows of $W$ samples. Thus, a segment of 5 seconds correspond to a window of 1024 samples. For each window, we extract the mean and standard deviation as features.

# In[12]:


# Normalize data
data_n = data.iloc[:, :-2]
scaler = StandardScaler().fit(data_n)
data_n = scaler.transform(data_n)


fs = 204.8  # 204.8 Hz
Twind = 5 # 5 seconds
Wwind = int(fs * Twind)
nsegments = np.ceil(len(data_n) / Wwind).astype(int)

X = []
acts = []
for s in range(nsegments):
    # Extract features for each segment
    segment = data_n[s*Wwind:(s+1)*Wwind, :]
    mean = np.mean(segment, axis=0)
    std = np.std(segment, axis=0)
    feat = np.concatenate((mean, std))
    X.append(feat)

    # Labels
    l = labels[s*Wwind:(s+1)*Wwind]
    counts = np.unique(l, return_counts=True)
    label = counts[0][np.argmax(counts[1])]
    acts.append(label)

X = np.stack(X)
acts = np.stack(acts) # range 0:K-1

print(X.shape)

f, ax = plt.subplots(1, 2, figsize=(16,4), sharey=True)
t = np.arange(X.shape[0])*Wwind/fs / 60
ax[0].plot(t, X[:, :3])
ax[0].set_xlabel('min')
ax[0].grid(alpha=0.4)
ax[0].set_title('Preprocessed accelerometer')
ax[1].plot(t, X[:, 6:9])
ax[1].set_xlabel('min')
ax[1].grid(alpha=0.4)
ax[1].set_title('Preprocessed gyroscope');


# ### 3.1.2. Learning the parameters of the HMM
# 
# **TASK1: Using hmmlearn, create a Gaussian HMM model and fit it to the preprocessed data stored in $X$ from the DaLiAc database. Using plt.imshow(), show the transition matrix $A$ as an image, in order to visualize better its elements. Use plt.xticks and plt.yticks to include the activities as labels.**

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

# Assuming X and activities are already defined from previous cells

# Create a Gaussian HMM model with 13 components (for 13 activities)
model = hmm.GaussianHMM(n_components=13, covariance_type="diag")

# Fit the model to the preprocessed data X
model.fit(X)

# Get the transition matrix A
A = model.transmat_

# Visualize the transition matrix
plt.figure(figsize=(8, 8))
plt.imshow(A, cmap="viridis")  # Use a colormap for better visualization
plt.colorbar()  # Add a colorbar to interpret the values
plt.xticks(np.arange(len(activities)), activities, rotation=90)
plt.yticks(np.arange(len(activities)), activities)
plt.title("Transition Matrix (A)")
plt.xlabel("Next State")
plt.ylabel("Current State")
plt.show()


# ## Questions
# 
# **Q1. Decribe the transition matrix you have obtained, and provide with your intuitions about it.**
# - Your answer.
# 
# **Q2. Is there any state for which the probability of transitioning to other state is considerable? Why?**
# - Your answer.
# 

# ### 3.1.2. Inference in HMM: MAP estimation of the states.
# 
# **TASK2: Use the corresponding method of hmmlearn to run the Viterbi algorithm that gives you the most probable sequence of states given the signal $X$. Plot the sequence of activities and the decoded states of the HMM. Use the given list of activities for the yticks in the plot.**
# 
# *Note: the states of the HMM does not have to be aligned with the labels. For example, state 4 might be associated to activity 1. Use the provided function to align the states with the activities.*

# In[14]:


def align_states(acts, states, activities=activities):
    """Align each hidden state with its corresponding class."""

    # M_ci show how many points of class c has been associated to state i
    L = len(activities)
    M = confusion_matrix(acts, states)
    plt.figure()
    plt.imshow(M)
    plt.xticks(np.arange(L), ['state '+str(i) for i in range(L)], rotation=90)
    plt.yticks(np.arange(L), activities)
    align = np.argmax(M, axis=0)
    aligned_states = states.copy()
    for s in range(13):
        aligned_states[states==s] = align[s]
    return aligned_states

import numpy as np
import matplotlib.pyplot as plt

# Assuming X, activities, and the trained model are already defined from previous cells

# Run the Viterbi algorithm to decode the most likely sequence of states
decoded_states = model.predict(X)

# Align the decoded states with the activity labels
aligned_states = align_states(acts, decoded_states, activities)

# Plot the sequence of activities and the decoded states
plt.figure(figsize=(12, 6))
plt.plot(acts, label="True Activities", marker="o", linestyle="-")
plt.plot(aligned_states, label="Decoded States", marker="x", linestyle="--")
plt.xlabel("Time Segment")
plt.ylabel("Activity/State")
plt.yticks(np.arange(len(activities)), activities)  # Set y-axis ticks to activity labels
plt.legend()
plt.title("Activity Recognition with HMM")
plt.grid(True)
plt.show()



# ## Questions
# 
# **Q3. Why the state index does not coincide with the class index?**
# - Your answer.
# 
# **Q4. Is there any set of activities that the model hardly distinguishes?**
# - Your answer.
# 

# ## 3.2. Part-Of-Speech (POS) Tagging with Hidden Markov Models
# 
# &nbsp;
# 
# <img src='http://www.tsc.uc3m.es/~ipeis/ML2/POS.png' width=500 />
# 
# &nbsp;
# 
# In Natural Language Processing (NLP), associating each word in a sentence with a proper POS (part of speech) is known as POS tagging or POS annotation. POS tags are also known as word classes, morphological classes, or lexical tags. A HMM can be fitted to text data, using the states $S$ for modeling the tags. Hence, for this part, we will fit a **HMM with fully observed data**.
# 
# In this experiment, we will use a Multinomial HMM. Features are the appearance of each word in a vocabulary conformed by words from a big database of sentences. Hence, the emission parameters are a big matrix $\mathbf{B}$ with dimensions $L\times M$, where $L$ is the number of states (number of tags) and $M$ is the size of the vocabulary (number of different words in the database).
# 
# The dataset for this experiment is the Treebank database, that can be easily download from the [nltk package for NLP in python](https://www.nltk.org/). As the dataset is tagged, we do not have to perform inference, we can just obtain the model parameters by applying the "Fully Observed Data" expressions of section 2.3. The dataset consist on a total of 3914 sentences, and each sentence $n$ is composed by $T_n$ words with their respective tags.
# 

# In[15]:


# nltk datasets will be downloaded in "Users/user/nltk_data".

#download the treebank corpus from nltk
nltk.download('treebank')

#download the universal tagset from nltk
nltk.download('universal_tagset')

# reading the Treebank tagged sentences
# data contains a list with sentences
# each sentence is a list of tuples with a word and the corresponding tag
data = list(nltk.corpus.treebank.tagged_sents(tagset='universal'))
print('Number of sentences: ' + str(len(data)))

#print each word with its respective tag for first two sentences
print('First two sentences:')
for sent in data[:2]:
    for tuple in sent:
        print(tuple)


# We are going to split the dataset in a 80% of the sentences for training, and 20% for test. All the sentences in these subsets will be concatenated in <code>train_words</code> and <code>test_words</code>, respectively. We are going to build a vocabulary <code>vocab</code> that will contain every word in the train set. The variable <code>tags</code> will contain the $L$ possible tags, which will be out states.

# In[18]:


# split data into training and validation set in the ratio 80:20
train_sentences,test_sentences =train_test_split(data,train_size=0.80,test_size=0.20,random_state = 101)

# join all the sentences in train and test sets
train_words = [ tup for sent in train_sentences for tup in sent ]
test_words = [ tup for sent in test_sentences for tup in sent ]
print('Train words: ' + str(len(train_words)))
print('Test words: ' + str(len(test_words)))

# check some of the tagged words.
print('\nFirst 20 training words: ')
print(train_words[:20])

# use set datatype to check how many unique tags are present in training data
tags = {tag for word,tag in train_words}
L = len(tags)
print('\nNumber of tags: ' + str(L))
print(tags)

# check total words in vocabulary
vocab = {word for word,tag in train_words}
print('\nNumber of words in the vocabulary: ' + str(len(vocab)))


# ## 3.2.1. HMM with fully observed data
# 
# 
# **TASK3: Build the functions for obtaining the emission and transition probabilities, using expressions of section 2.3.**

# In[19]:


import nltk

def emission(word, tag, train_bag=train_words):
    """Probability B_im that a tag i emits a word m, computed from train_bag"""
    tag_list = [pair for pair in train_bag if pair[1] == tag]
    count_tag = len(tag_list)  # N_i: Count of words with the given tag
    word_tag_list = [pair[0] for pair in tag_list if pair[0] == word]
    count_word_tag = len(word_tag_list)  # N_im: Count of the word with the given tag

    B_im = count_word_tag / count_tag if count_tag else 0  # Emission probability

    return B_im

def transition(tag1, tag2, train_bag=train_words):
    """Probability A_ij of state j (tag2) given that previous state was i (tag1)"""
    tags = [pair[1] for pair in train_bag]
    count_tag1 = len([t for t in tags if t == tag1])  # N_i: Count of tag1 occurrences
    count_tag1_tag2 = 0
    for i in range(len(tags) - 1):
        if tags[i] == tag1 and tags[i + 1] == tag2:
            count_tag1_tag2 += 1  # N_ij: Count of tag2 after tag1

    A_ij = count_tag1_tag2 / count_tag1 if count_tag1 else 0  # Transition probability

    return A_ij

def transition_matrix(tags, train_bag=train_words):
    """Build transition matrix A of dimensions LxL from train_bag"""
    tags = list(tags)  # Convert tags to a list
    A = [[transition(tag1, tag2, train_bag) for tag2 in tags] for tag1 in tags]
    A = np.array(A)
    A = pd.DataFrame(A, columns=tags, index=tags)
    A['.'] = [0] * len(tags)  # Add a row for the initial state ('.')
    A.loc['.'] = [0] * len(tags) + [1]  # Initialize the initial state row
    A.loc['.', '.'] = 0  # Set initial state probability to 0 (will be calculated later)

    return A

def Viterbi(words, transition_matrix, train_bag = train_words):
    """ Obtain the most probable sequence of tags given a list of words"""
    state = []
    T = list(set([pair[1] for pair in train_bag]))

    for key, word in enumerate(words):
        #initialise list of probability column for a given observation
        p = []
        for tag in T:
            if key == 0:
                transition_p = transition_matrix.loc['.', tag]
            else:
                transition_p = transition_matrix.loc[state[-1], tag]

            # compute emission and state probabilities
            emission_p = emission(words[key], tag)
            state_probability = emission_p * transition_p
            p.append(state_probability)

        pmax = max(p)
        # getting state for which probability is maximum
        state_max = T[p.index(pmax)]
        state.append(state_max)

    return list(zip(words, state))


# In[20]:


# convert the matrix to a df for better readability
#the table is same as the transition table shown in section 3 of article
A = transition_matrix(tags)
A = pd.DataFrame(A, columns = list(tags), index=list(tags))
plt.figure(figsize=(8, 8))
plt.imshow(A)
plt.xticks(np.arange(len(tags)), list(tags))
plt.yticks(np.arange(len(tags)), list(tags))
display(A)


# #### **TASK4: Choosing 10 random sequences from the test set, use the Viterbi function to estimate their tags. Compute the accuracy as the proportion of corrected tagged words.**

# In[ ]:


#############
#           #
# YOUR CODE #
#           #
#############


# ## 3.2.2. Sampling sentences
# 
# We know how to obtain the parameters of our model when we have fully observed data. As we have a generative model, we can obtain samples from it. In this section, you will build a Multinomial HMM object of hmmlearn and sample sentences.
# 

# **TASK5: Obtain the emission matrix $\mathbf{B}$ matrix using the emission function. You should calculate the probability of each word in <code>vocab</code> for each tag.**
# 
# *Note: $\mathbf{B}$ will be a huge matrix. This computation might take a while (up to half an hour).*

# In[ ]:


#############
#           #
# YOUR CODE #
#           #
#############


# **TASK6: Compute the parameters $\boldsymbol{\pi}$ as the proportion of the tags in the train set.**

# In[ ]:


#############
#           #
# YOUR CODE #
#           #
#############


# **TASK7: Build a Multinomial HMM object with the computed parameters and print 30 words sampled from it.**

# In[ ]:


from hmmlearn.hmm import MultinomialHMM

#############
#           #
# YOUR CODE #
#           #
#############


# ## Questions
# 
# **Q5. Do the sampled sequences follow a correct grammatical structure?**
# - Your answer.
# 
# **Q6. Do the sampled sequences have a properly semantic content? Why?**
# - Your answer.
# 

# ## References
# 
# [1]. Murphy, K. P. (2012). Machine learning: a probabilistic perspective. MIT press.
# 
# [2]. Bishop, C. M. (2006). Pattern recognition and machine learning. springer.
# 
# 
