# AI Travel Planner Project

## Project Overview

This project is part of Lab 2 for the ID2223 course at KTH. The goal is to fine-tune a large language model (LLM) by using Parameter Efficient Fine Tuning (PEFT) with Low-Rank Adaptation (LoRA) and demonstrate its capabilities through an innovative application. The application, *AI Travel Planner*, is a Gradio-based interface that generates personalized travel itineraries based on user input. The fine-tuned LLM enables detailed and relevant recommendations, showcasing its utility in the travel domain.

---

## Features
1. **Fine-Tuned LLM**:
   - The model, fine-tuned using PEFT with LoRA, enables efficient memory usage while maintaining performance.

2. **Interactive UI**:
   - A Gradio interface enables users to input their desired city and travel preferences to generate tailored travel plans.

3. **Efficiency Optimizations**:
   - The model leverages 4-bit quantization for reduced computational overhead.
   - Checkpointing and memory-efficient techniques were implemented to enable training on a single GPU.

4. **Deployment**:
   - The fine-tuned large language model has been saved to HuggingFace and is publicly accessible there. The Travel Planner UI has been developed using Gradio and deployed through Google Colab to generate a temporary link.

---

## Task 2: Improve Pipeline Scalability and Model Performance

### 1. Performance Improvement Strategies

**Model-Centric Approach**:
- **Hyperparameter Tuning**: Experimented with `learning_rate`, `warmup_steps`, `max_steps`, and `num_train_epochs` to optimize performance.
- **Model Selection**: Evaluated multiple open-source LLMs such as `unsloth/Llama-3.2-3B-Instruct` to identify the best-performing model.
- **Optimization Techniques**:
  - Used `adamw_8bit` optimizer for memory efficiency.
  - Applied gradient accumulation with `gradient_accumulation_steps` to improve training throughput.
  - Employed learning rate schedulers to dynamically adjust learning rates for stable training.

**Data-Centric Approach**:
- **Data Sources**: Augmented the FineTome dataset with travel-related datasets to improve domain-specific performance.
- **Data Preprocessing**: Applied techniques like text normalization and deduplication to enhance data quality.

### 2. Fine-Tuning Multiple LLMs
- Tried fine-tuning smaller models like `Llama-3.1-1B` for faster inference on CPUs.
- Experimented with HuggingFace FineTuning frameworks to explore alternative approaches.

---

## Quantization and LoRA Details

### Quantization
- Applied 4-bit quantization to reduce memory usage without significant loss in accuracy.

### LoRA Parameters
- **lora_alpha**: Adjusted to control the influence of LoRA updates on the model.
- **lora_dropout**: Used to prevent overfitting during fine-tuning.
- **rank stabilized LoRA (RS-LoRA)**: Ensured robust low-rank adaptation across layers.

---

## Training Hyperparameters
- **gradient_accumulation_steps**: Improved memory utilization by accumulating gradients over multiple steps.
- **warmup_steps**: Gradual learning rate increase for stable optimization.
- **max_steps** and **num_train_epochs**: Set based on dataset size and training time constraints.
- **learning_rate**: Tuned for balanced convergence.
- **fp16** and **bf16**: Mixed-precision training for efficiency.
- **output_dir**: Saved checkpoints locally and synced with Google Drive.

---

## Deployment and Inference
- **Colab**: Used Google Colab with T4 GPU for fine-tuning, saving checkpoints to Google Drive.
- **HuggingFace Spaces**: Deployed a Gradio-based UI for inference, enabling interaction with the fine-tuned LLM.
- **Gradio UI**: Offers an intuitive interface for querying the model and generating travel plans.

---

## Project Justification

### Why Use This Model?
- Chose `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` due to its balance of performance and computational efficiency.

### Inference in Colab
- Due to CPU-only limitations on HuggingFace Spaces, inference and demonstrations are done in Colab using a GPU.

---

## Future Work
- Integrate new datasets to further refine model outputs.
- Longer training to increase performance of LLM.
- Optimize the UI.
- Explore additional domains beyond travel planning.

---

## Links
- **Discussion on Task Challenges**: [Canvas Discussion](https://canvas.kth.se/courses/50172/discussion_topics/432284)
- **Gradio App (temporary link)**: [AI Travel Planner](https://huggingface.co/spaces/Eugenius0/ai-travel-planner)
- **Google Colab Inference Notebook**: [Colab Notebook](link)
