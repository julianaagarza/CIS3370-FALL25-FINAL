```mermaid
graph LR
    A[CIC-IDS-2018 Dataset] -->|Raw CSVs| B(Preprocessing & Cleaning)
    B -->|Feature Extraction| C{ML Model}
    C -->|Benign| E[JSON Generation]
    C -->|Malicious| E
    E -->|Ingestion| F((Elasticsearch))
    F -->|Visualization| G[Kibana Dashboard]
```
