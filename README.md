# AI-Powered Drug Combination Therapy Platform

A complete application for predicting drug synergy using **Graph Neural Networks (GNNs)**, with all data stored in a **PostgreSQL database** instead of CSV files.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🎯 Overview

This platform leverages state-of-the-art **Heterogeneous Graph Neural Networks** to predict synergistic drug combinations for cancer treatment. The system analyzes complex relationships between drugs, biological targets, and diseases stored in a relational database.

### Key Features

- ✅ **Database-Driven**: All data in PostgreSQL (no CSV files after ETL)
- ✅ **Graph Neural Networks**: Heterogeneous GNN with drug, target, and disease nodes
- ✅ **Safety Checking**: Automatic detection of harmful drug interactions
- ✅ **Uncertainty Quantification**: Confidence scores using MC Dropout
- ✅ **REST API**: FastAPI backend with comprehensive endpoints
- ✅ **Interactive Dashboard**: Streamlit-based UI with molecular visualization
- ✅ **Docker Support**: Containerized deployment with docker-compose
- ✅ **Comprehensive Tests**: Unit tests for all major components

---

## 🏗️ Architecture