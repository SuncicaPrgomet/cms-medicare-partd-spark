# CMS Medicare Part D Prescriber Analysis

Personal project analyzing large-scale U.S. healthcare claims data end-to-end on Databricks — from API-based ingestion through distributed data engineering, exploratory analysis, Spark ML classification, and a generative AI layer that turns aggregated results into a plain-language executive summary.

## Dataset

Medicare Part D Prescribers – by Provider and Drug (CMS.gov) — a publicly available dataset of U.S. prescriber-level drug claims, with no credentialing required. Fields include prescriber specialty, state, brand/generic drug name, total claims, total beneficiaries, day supply, and total drug cost.

## Why PySpark on Databricks instead of pandas?

The full dataset is tens of millions of rows (~4 GB as a raw CSV) — too large to comfortably load or manipulate with pandas on a single machine, and too large to upload directly through Databricks Community Edition's UI (~2 GB limit). Instead, this project:

* Pulls data directly from the CMS Data API in paginated batches of 5,000 rows via `requests`
* Builds a Spark DataFrame straight from the collected records — no intermediate pandas step, no manual file upload
* Uses distributed Spark operations (not pandas) for every subsequent step: cleaning, aggregation, joins, and ML

## What this project demonstrates

### 1. Data ingestion & cleaning

* Paginated REST API ingestion directly into Spark
* Schema correction (CMS returns most fields as strings; numeric columns are explicitly cast to `long`/`double`)
* Missing-value auditing across all columns (nulls and blank strings)

### 2. Exploratory analysis with distributed aggregations

* Prescribing volume and cost broken down by specialty, drug, and state
* Provider-level aggregation with derived metrics

### 3. Spark internals

* Inspected physical execution plans (`.explain("formatted")`) to compare narrow vs. wide transformations
* Explicit `repartition()` / `coalesce()` to control shuffle behavior

### 4. Spark SQL, window functions, and joins

* Registered a temp view and queried it with Spark SQL
* Ranked providers within each specialty using window functions
* Built a provider lookup table and joined it back to claim-level data

### 5. Spark ML classification

* Engineered provider-level features and defined a high-cost prescriber label (top 10% of total drug cost, via `approxQuantile` to avoid a full sort)
* Trained a Logistic Regression pipeline (`VectorAssembler` + `LogisticRegression`)
* Evaluated with AUC, accuracy, precision, recall, F1, a confusion matrix, and a threshold sweep to show the precision–recall trade-off on this imbalanced problem

### 6. Generative AI layer

* Fed aggregated Spark results (top specialties, drugs, states) into Databricks' native `ai_query()` SQL function
* Generated a natural-language executive summary suitable for a non-technical, healthcare-analytics audience
* Explicitly prompted the model to avoid unsupported clinical or causal claims

## Tech stack

Python · PySpark · Databricks (Photon execution engine) · Spark SQL · Spark ML · Databricks `ai_query()` (Llama 3.3 70B) · CMS Data API

## Key takeaways

* A REST API with pagination is a practical way to bring large public datasets into Spark without manually handling a multi-GB file.
* Reading Spark's physical execution plans, rather than only using the DataFrame API, helps with understanding and controlling shuffle behavior at scale.
* Spark can support an end-to-end workflow spanning data ingestion, cleaning, distributed analysis, feature engineering, and machine learning.
* Generative AI is useful here as a presentation layer on top of rigorously computed aggregates, rather than as a replacement for the underlying statistical analysis.
