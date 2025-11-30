# MÍMIR: Cognitive Diagnostic & Incident-Response Agent

An end-to-end, production-grade LLM system for DevOps, SRE, and cloud-infrastructure reasoning.

---

## 📌 Project Overview

**Mímir** is an advanced LLM-driven cognitive workflow agent designed for incident response, DevOps troubleshooting, and cloud/SRE diagnostics. It's engineered as a real-world, industry-standard AI assistant, not a toy chatbot.

It integrates several advanced AI and software engineering components:

- **RAG (Retrieval-Augmented Generation)**
- **PEFT Fine-Tuning (LoRA/QLoRA)**
- **Structured Chain-of-Thought reasoning**
- **Quantization-aware Serving (vLLM / TGI)**
- **Incident-specific Evaluation & Ablations**
- **Guardrails, safety layers, and multi-turn memory**
- **Production-quality engineering workflows**

The core goal is to analyze logs, alerts, and metrics, retrieve relevant documentation (SOPs, runbooks, docs), and reason step-by-step to produce actionable root-cause analyses and suggest safe mitigation steps consistent with SRE best practices.

---

## 💻 Core Use Case: Incident Assistance

Mímir assists engineers by providing structured diagnostics during incidents, including but not limited to:

- Service downtime
- Pod crash loops
- Failing deployments
- Memory/CPU saturation
- Authentication/configuration errors
- Networking issues and database latency spikes
- Container-level faults and alert rule anomalies
- CI/CD pipeline errors

---

## 🧠 Key Capabilities

### 1. Retrieval-Augmented Generation (RAG)

Enables grounded reasoning by fetching relevant context from a knowledge base.

- **Custom Ingestion**: Kubernetes docs, AWS/GCP playbooks, Google SRE Handbook, DevOps SOPs/runbooks, and example diagnostic cases.
- **Workflow**: Chunking, cleaning, embedding, indexing, and Top-k retrieval with context injection into the LLM.

### 2. PEFT Fine-Tuning (LoRA/QLoRA)

Uses Supervised Fine-Tuning (SFT) on structured incident reasoning and troubleshooting data.

- **Objective**: Improve reasoning structure, reduce hallucinations, and align outputs with DevOps best practices.
- **Implementation**: Open-source LLM (e.g., LLaMA/Mistral) + LoRA/QLoRA adapters, trained via TRL SFTTrainer + PEFT.

### 3. Structured Chain-of-Thought (CoT) Reasoning

The model is fine-tuned to output clean, interpretable, step-by-step reasoning using a defined schema:

1. Symptom Identification
2. Hypothesis Generation
3. Checks / Verifications
4. Root Cause Conclusion
5. Recommended Action Steps

### 4. Incident Evaluation & Ablation Harness

A full evaluation pipeline to rigorously test system performance.

- **Comparisons**: Baseline vs PEFT, RAG vs No-RAG, Zero-shot CoT vs Fine-tuned CoT.
- **Ablations**: Testing chunk size, top-k retriever variations, latency, and performance.
- **Metrics**: LLM-as-a-Judge scoring for quality, summarized in tables, plots, and a report.md.

### 5. Guardrails & Safety Layer

Ensures the agent is safe, reliable, and operates within defined boundaries.

- Rejects unsupported domains and filters junk/off-topic inputs.
- Restricts high-risk operational advice.
- Generates "safe mitigations" aligned with industry SRE practices and requires multi-step verification.

### 6. Model Serving Optimization

Engineered for high-throughput, low-latency inference.

- **Servers**: vLLM and TGI (Text Generation Inference) for robust, containerized deployment.
- **Optimization**: 4-bit quantization (QLoRA, GPTQ, AWQ) for reduced memory footprint.
- **API**: Async FastAPI inference API. Everything runs in Docker.

### 7. Multi-Turn Conversational Memory

Maintains short contextual memory to support multi-step incident resolution.

- **Mechanism**: Rolling window prompts, incident-session tracking, and token budgeting via context trimming.

---

## ⚙️ Technical Architecture

Mímir is modularized for clarity, maintenance, and scalability.

### Core Modules

| Module | Description | Key Components |
|--------|-------------|----------------|
| `/rag_pipeline/` | Ingestion, chunking, embedding, and vector DB management. | `ingest.py`, `retriever.py` |
| `/training/` | Supervised Fine-Tuning (SFT), LoRA adapter generation, and checkpoints. | `sft_train.py`, `lora_config.yaml` |
| `/serving/` | Model deployment using vLLM/TGI and handling quantized models. | `vllm_server.py`, `tgi_server.py` |
| `/api/` | FastAPI backend, structured response schemas, and middleware. | `main.py`, `schemas.py` |
| `/eval/` | Ablations, quality metrics, and LLM-as-a-Judge scoring. | `baseline_eval.py`, `llm_judge.py` |
| `/guardrails/` | Safety classifiers, input filters, and operational safety rules. | `classifier.py`, `safety_rules.yaml` |
| `/memory/` | Multi-turn session management and conversation summarization. | `conversation_manager.py` |
| `/data/` | Storage for raw documents, processed chunks, and training/evaluation datasets. | `raw/`, `train/`, `eval/` |

### System Flow

```
User query → Retriever → Context selection → PEFT-fine-tuned LLM → CoT reasoning → Final actionable output
```

---

## 📁 Folder Structure

```
mimir/
│
├── rag_pipeline/           # RAG components
├── training/               # Fine-tuning scripts & checkpoints
├── serving/                # Model inference servers (vLLM, TGI)
├── api/                    # FastAPI backend
├── eval/                   # Evaluation harness and metrics
├── guardrails/             # Safety & filtering logic
├── memory/                 # Conversational context & session manager
├── data/                   # All raw and processed data
└── README.md
```

---

## 🚀 Planned Milestones

- [ ] RAG MVP working end-to-end
- [ ] CoT dataset creation and preparation
- [ ] PEFT SFT fine-tuning with LoRA completion
- [ ] vLLM inference server deployment
- [ ] Incident evaluation harness operational
- [ ] Ablation experiments and reporting
- [ ] Guardrails & safety layer integration
- [ ] Multi-turn memory implementation
- [ ] Final documentation and demo

---

## 🔥 Vision Statement

**Mímir** is a serious, engineering-grade cognitive assistant for real DevOps/SRE workflows. It demonstrates the high-value skillset of modern applied AI engineering: LLM reasoning, retrieval grounding, safe automation, and production-level system design.

This project is positioned to be both research-credible and industry-admired.

---

## 📝 Notes

For more details on specific components like the PEFT Fine-Tuning process or the Structured Chain-of-Thought schema, please refer to the respective module documentation or open an issue for discussion.