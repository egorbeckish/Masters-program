import streamlit as st
from sklearn import datasets
from sklearn.decomposition import PCA, FactorAnalysis, KernelPCA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
import time
import numpy as np
import matplotlib.pyplot as plt


DECOMPOSITION = {
	"PCA": PCA(),
	"Factor Analysis": FactorAnalysis(),
	"KernelPCA": KernelPCA()
}

MANIFOLD = {
	"t-SNE": TSNE(random_state=42),
	"MDS": MDS(init="random", n_init=1, random_state=42),
	"Isomap": Isomap()
}

DATASET = {
	"Breast Cancer": datasets.load_breast_cancer(),
	"Diabetes": datasets.load_diabetes(),
	"Digits": datasets.load_digits(),
	"Iris": datasets.load_iris(),
	"Wine": datasets.load_wine()
}
