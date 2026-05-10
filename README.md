# Overview

This project builds a lightweight hybrid log analysis pipeline using Regex-based classification for deterministic patterns
Sentence embeddings + clustering for grouping unknown logs
ML classification for semantic categorization of logs

The current implementation uses:

`all-MiniLM-L6-v2` from sentence-transformers
LogisticRegression from scikit-learn

This setup was intentionally chosen because it is:

- Lightweight
- Fast on CPU
- Easy to train locally
- Low memory usage
- Good enough for most internal tooling / observability pipelines

## Project Structure
```
project/
│
├── training/
│   ├── dataset/
│   │   └── synthetic_logs.csv
│   │
│   ├── regex_patterns.json
│   │
│   ├── train.py
│   ├── clustering.py
│   ├── classifier.py
│   └── config.py
│
├── models/
│   └── log_classifier.joblib
│
├── requirements.txt
└── README.md
```

## Installation

1. Create Virtual Environment
`python -m venv venv`

Activate:

- Mac/Linux
`source venv/bin/activate`

- Windows
`venv\Scripts\activate`
2. Install Dependencies
`pip install -r requirements.txt`

Dataset Format

Expected CSV structure:

```csv
log_message,target_label,source
"User User123 logged in.","User Action","Portal"
"Backup completed successfully.","System Notification","Infra"
```



Column	Description :

| Column | Description |
|---|---|
| log_message | Raw log text |
| target_label | Final classification label |
| source | System/source generating logs |
`

Regex Pattern Configuration

Regexes are externalized into:
`configs/regex_patterns.yaml`

Example:

```{
  "User Action": [
    "User User\\d+ logged (in|out).",
    "Account with ID .* created by .*"
  ],
  "System Notification": [
    "Backup completed successfully.",
    "System updated to version .*"
  ]
}
```
This allows:

- Adding new patterns without code changes
- Easy experimentation
- Cleaner maintenance 

## Training the Model
Main Training Entry Point

Run:

`python train.py`

This script:

- Loads dataset
- Applies regex classification
- Filters already-classified logs
- Generates embeddings
- Trains classifier
- Evaluates accuracy
- Saves trained model


## Output Model

Trained model gets saved at:

`models/log_classifier.joblib`

Load later using:

```
import joblib

clf = joblib.load("models/log_classifier.joblib")
```


## Current embedding model:

### SentenceTransformer('all-MiniLM-L6-v2')

Chosen because:

| Feature | Benefit |
|---|---|
| Small (~80MB) | Fast downloads |
| CPU friendly | No GPU required |
| Fast inference | Good for real-time systems |
| Strong semantic quality | Good enough for log similarity |
| Production proven | Widely used |

Very suitable for:

- Internal tooling
- Real-time pipelines
- Low-cost deployments
- Local development
- Edge/serverless systems

## Alternative Embedding Models

Depending on scale and accuracy needs:

| Model | Pros | Cons |
|---|---|---|
| all-MiniLM-L6-v2 | Very fast/lightweight | Slightly lower accuracy |
| all-mpnet-base-v2 | Better semantic quality | Heavier/slower |
| BAAI/bge-small-en | Strong retrieval quality | Slightly larger |
| BAAI/bge-base-en | Excellent embeddings | More RAM usage |
| intfloat/e5-small-v2 | Great for search/classification | Slightly slower |
| intfloat/e5-base-v2 | Very accurate | Heavy |
| gte-small | Efficient modern embedding model | Less common ecosystem |

## How To Switch Models

Update in config:

`EMBEDDING_MODEL = "all-MiniLM-L6-v2"`

Example:

`EMBEDDING_MODEL = "all-mpnet-base-v2"`

Then retrain:

`python training/train.py`

Important:

- Embedding dimensions may change
- Always retrain classifier after switching embedding model
- Larger models improve quality but increase:
  - latency
  - RAM
  - disk usage

Recommended Production Improvements
1. Persist Embeddings : Avoid recomputing embeddings repeatedly.
   > Possible options:
        FAISS, 
        ChromaDB, 
        Qdrant, 
        Pinecone.
2. Add Confidence Thresholds
Low confidence predictions can:

- fallback to clustering
- fallback to regex
- trigger human review

3. Incremental Regex Learning : Cluster unknown logs periodically and generate:

- new regex candidates
- new labels
- anomaly alerts
4. Use Better Classifiers

Current: LogisticRegression

Can explore:

- XGBoost
- LightGBM
- RandomForest
- LinearSVC
5. Add Drift Detection

Monitor:

- new log distributions
- embedding drift
- unseen patterns

Useful for long-running systems.

## Example Prediction Flow
```
Incoming Log
      ↓
Regex Match?
      ↓ yes
Return Regex Label
      ↓ no
Generate Embedding
      ↓
ML Classification
      ↓
Low Confidence?
      ↓ yes
Cluster / Human Review
```

## Future Enhancements

Possible upgrades:

- Online learning
- Streaming Kafka ingestion
- Vector DB integration
- Real-time anomaly detection
- OpenTelemetry support
- Log summarization using local LLMs
- Ollama integration for local semantic analysis

## TODO : 
- Add LLM model for predictions for unknown statuses
- Add processign pipeline code which will take logs and classify them.