# Concepts of Data Science

A learning repository documenting my journey through AI and Machine Learning concepts during my internship at A79.AI.

## 🎯 Project Overview

This repository serves as a comprehensive collection of AI/ML concepts, with each topic including both theoretical explanations and practical implementations. The focus is on modern AI techniques that bridge natural language processing and structured outputs.

## 📚 Core Concepts

### 1. K-shot Learning
K-shot learning enables models to learn from limited examples, making them more efficient and practical. The implementation includes:

- **Types Covered:**
  - Zero-shot learning (k=0): Learning without explicit examples
  - One-shot learning (k=1): Learning from a single example
  - Few-shot learning (k=2-5): Learning from limited examples

- **Implementation Features:**
  - TF-IDF vectorization with improved settings
  - Cosine similarity for matching
  - Support for multiple categories
  - Configurable number of examples (k)

### 2. Natural Language to DSL Translation
Two main approaches are implemented for converting natural language to Domain Specific Language:

#### A. Structured Decoding
- Direct generation of well-formed DSL outputs
- Built-in validation patterns for operations
- Supports mathematical operations (add, subtract, multiply, divide)
- Ensures syntactic correctness

#### B. Chat Prompting with Parsing
- Natural language interaction
- DSL extraction from conversational responses
- Pattern-based validation
- Flexible response handling

### 3. Performance Benchmarking & Visualization
Comprehensive benchmarking system to compare both approaches:

#### Metrics Tracked
- Accuracy across different k-shot scenarios
- Response time performance
- Confidence scores
- Error rates and types

#### Visualization Suite
- **Accuracy Comparison:** Line plots showing accuracy vs k-shot learning
- **Response Time Analysis:** Box plots of processing speed distribution
- **Confidence Metrics:** Bar charts comparing confidence scores
- **Performance Trade-offs:** Scatter plots of accuracy vs response time

## 📅 Implementation Timeline

| Date       | Concept             | Implementation Details |
|------------|--------------------|-----------------------|
| 24-02-2025 | K-shot Learning    | Agent-based implementation with TF-IDF |
| 24-02-2025 | Structured Decoding| DSL conversion with validation |

## 🛠️ Technical Stack

- **Language:** Python
- **Key Libraries:**
  - scikit-learn (TF-IDF, cosine similarity)
  - NumPy (numerical operations)
  - re (pattern matching)

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/Vishesh711/Concepts-of-Data-Science.git
```

2. Install dependencies:
```bash
pip install scikit-learn numpy
```

3. Explore concept directories:
   - `K_shot_Learning/`: K-shot learning implementations
   - `NL to DSL/`: Natural Language to DSL conversion

## 📖 Directory Structure

```
.
├── K_shot_Learning/
│   ├── simple_K_shot_Learning_in_a_agent.py
│   └── concept_idea.txt
├── NL to DSL/
│   ├── Agents/
│   │   ├── Structured_Decoding_Agent.py
│   │   └── Chat_Prompting_Agent.py
│   ├── Benchmarking/
│   │   ├── Benchmarking_script.py
│   ├── Documentation/
│   │   ├── agent_comparison.txt
│   │   └── implementation_details.txt
└── README.md
```

## ✨ Features

- Detailed concept explanations
- Practical implementations
- Well-documented code
- Real-world applications
- Performance benchmarking

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for questions or bugs
- Submit pull requests for improvements
- Share ideas for new concepts to cover

## 📝 License & Author

<div align="left" style="display: flex; align-items: center; gap: 20px;">
  <a href="https://github.com/Vishesh711">
    <img src="https://github.com/Vishesh711.png" width="100px" alt="Vishesh711" style="border-radius:50%"/>
  </a>
  <div>
    <h3><a href="https://github.com/Vishesh711">@Vishesh711</a></h3>
    <p>Copyright © 2025. All rights reserved.</p>
  </div>
</div>

---
*This is a learning repository - concepts and implementations are continuously being updated and improved.*