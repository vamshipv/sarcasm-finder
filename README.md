"""# Python Data Screening & Curation Pipeline

This repository contains the data engineering pipeline designed to filter, clean, and deduplicate raw Python source code datasets (e.g., **StarCoder**, **CodeParrot**) for training **Small Language Models (SLMs)**.

The goal of this pipeline is to maximize information density and ensure that the model trains exclusively on high-quality, syntactically valid, and logically unique human-written Python code that fits within a standard 512-token context window.

---

## Pipeline Overview

The pipeline processes data through four consecutive, isolated stages:

```mermaid
graph TD
    A[Raw Data Ingestion] --> B[Stage 1: Signal-to-Noise Filter]
    B -->|Removes legal headers, filters boilerplate| C[Stage 2: Syntax & Exact Deduplication]
    C -->|Validates AST parse, eliminates identical files| D[Stage 3: MinHash Semantic Deduplication]
    D -->|Elimates near-duplicates >85% similarity| E[High-Density Dataset Ready for Training]
