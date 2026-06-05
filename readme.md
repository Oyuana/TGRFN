# TGRFN: Temporal Graph Relational Fake News Detection for Early Rumor Identification

## Introduction

This repository contains the official implementation of **TGRFN**, a temporal graph-based fake news detection framework designed for early-stage rumor identification on social media platforms.

Unlike conventional graph neural network approaches that rely on static propagation structures, TGRFN explicitly models:

* Temporal evolution of information propagation;
* Dynamic contribution decay of propagation nodes;
* Multi-relational interactions among news, posts, and users;
* Early-stage fake news detection under sparse propagation conditions.

The framework is evaluated on three widely used benchmark datasets:

* GossipCop
* PolitiFact
* PHEME

---

## Framework Overview

The overall pipeline of TGRFN is:

```text
News Collection
      │
      ▼
Feature Extraction
(Text + Visual Features)
      │
      ▼
News / Post / User Node Construction
      │
      ▼
Propagation Graph Generation
      │
      ▼
Random Walk with Restart (RWR)
      │
      ▼
Temporal Graph Relational Modeling
      │
      ▼
Fake News Detection
```

---

## Project Structure

```text
TGRFN/
│
├── data/
│   ├── raw datasets
│   ├── processed_data/
│   ├── rwr_results/
│   ├── feature extraction scripts
│   ├── graph construction scripts
│   └── dataset loaders
│
├── fakenewsnet_code/
│   ├── news content collection
│   ├── tweet collection
│   ├── retweet collection
│   ├── user profile collection
│   └── Twitter API resource management
│
├── models/
│   ├── train_and_evaluation/
│   │   ├── GossipCop.py
│   │   ├── PolitiFact.py
│   │   ├── Pheme.py
│   │   ├── ablation variants
│   │   └── early-detection experiments
│   │
│   ├── pre-trained/
│   │   ├── gossipcop.rar
│   │   ├── politifact.rar
│   │   └── pheme.rar
│   │
│   └── ablation/
│
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Dataset Preparation

### FakeNewsNet

Download the original FakeNewsNet dataset:

https://github.com/KaiDMML/FakeNewsNet

The repository provides data collection scripts under:

```text
fakenewsnet_code/
```

including:

* news_content_collection.py
* tweet_collection.py
* retweet_collection.py
* user_profile_collection.py

---

### PHEME

Download the PHEME dataset from the official source and place it under:

```text
data/PHEME/
```

---

## Feature Extraction

### Text Features

Generate textual embeddings:

```bash
python data/embed_text_gossipcop.py
python data/embed_text_politifact.py
python data/embed_text_pheme.py
```

---

### Visual Features

Extract image representations:

```bash
python data/visual_feature_extractor.py
```

---

## Graph Construction

Generate heterogeneous propagation graphs:

```bash
python data/generate_graph_gossipcop.py
python data/generate_graph_politifact.py
python data/generate_graph_pheme.py
```

The generated graph contains three node types:

* News
* Post
* User

and multiple relation types among them.

---

## Random Walk Sampling

Construct propagation subgraphs using Random Walk with Restart (RWR):

```bash
python data/rwr_gossipcop.py
python data/rwr_pheme_politifact.py
```

Generated files are stored under:

```text
data/rwr_results/
```

---

## Training

### GossipCop

```bash
python models/train_and_evaluation/GossipCop.py
```

### PolitiFact

```bash
python models/train_and_evaluation/PolitiFact.py
```

### PHEME

```bash
python models/train_and_evaluation/Pheme.py
```

---

## Ablation Studies

The repository includes multiple ablation settings:

### Temporal Module Removal

```bash
python models/train_and_evaluation/*-time.py
```

### Relation Module Removal

```bash
python models/train_and_evaluation/*-relation.py
```

### Attention Module Removal

```bash
python models/train_and_evaluation/*-atten.py
```

---

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate tgrfn
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

---

## Pre-trained Resources

Preprocessed resources are provided in:

```text
models/pre-trained/
├── gossipcop.rar
├── politifact.rar
└── pheme.rar
```

Please extract them before running experiments.


