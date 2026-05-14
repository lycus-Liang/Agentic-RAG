# Omni-Modal Agentic RAG Architecture

This project has two main paths: an offline indexing pipeline that builds the Qdrant collection, and an online RAG workflow that plans, retrieves, and generates answers.

![Omni-Modal Agentic RAG Architecture](./architecture.svg)

```mermaid
flowchart TD
    U[User] --> UI[Streamlit UI / API / CLI]
    UI --> WF[LangGraph Workflow]

    CFG[configs/config.yaml] --> WF
    PROMPTS[configs/prompts.yaml] --> WF

    WF -->|use_agentic_loop=true| PLAN[plan_query_node]
    PLAN --> TC[tool_call_agent_node]
    TC -->|native tool call| RT[Retrieval Tools]
    RT -->|retrieve_text / retrieve_vision / retrieve_hybrid + search_style| SEARCH[Searcher]

    SEARCH --> ENC[QueryEncoder]
    ENC -->|BGE dense/sparse| VEC1[Text Vectors]
    ENC -->|optional HyDE, config capped| HYDE[HyDE Expansion]
    ENC -->|ColQwen2 / ColPali| VEC2[Vision Vectors]

    VEC1 --> QD[Qdrant]
    VEC2 --> QD
    HYDE --> VEC1

    QD -->|RRF Fusion| FUSION[Recall Results]
    FUSION -->|optional config reranker| RERANK[Reranker]
    RERANK --> GEN[generate_node]
    FUSION --> GEN

    GEN --> LLM[Generation LLM]
    LLM --> UI

    WF -->|use_agentic_loop=false| RET[retrieve_node]
    RET --> SEARCH

    INGEST[Offline Build Pipeline] --> PARSE[PDF / JSON Parsing]
    PARSE --> CHUNK[Chunks + Crops + Proxy Descriptions]
    CHUNK --> EMBED[Text / Vision Embeddings]
    EMBED --> QD
```

## Retrieval Control

The agent chooses a modality-level tool and an intent-level retrieval style:

- `retrieve_text`, `retrieve_vision`, `retrieve_hybrid`
- `search_style`: `auto`, `exact`, `semantic`, or `expanded`

The agent does not directly control backend switches such as `use_dense`, `use_sparse`, `use_hyde`, or `use_reranker`. Those remain capped by `configs/config.yaml`.

## Search Style Mapping

- `auto`: use the configured dense, sparse, HyDE, vision, and reranker settings.
- `exact`: disable HyDE and prefer sparse lexical matching; fall back to dense only if sparse is disabled.
- `semantic`: disable HyDE and use configured text semantic/lexical routes.
- `expanded`: allow HyDE only when `use_hyde` is enabled; otherwise degrade to semantic behavior.

All active recall routes are fused through Qdrant RRF before optional reranking.
