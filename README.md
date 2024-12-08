# AI Travel Planner Project - Parameter Efficient Fine-Tunning (PEFT) with Low-Rank Adaptation (LoRA)

## Project Overview

This project is part of Lab 2 for the ID2223 course at KTH. The goal is to fine-tune a large language model (LLM) by using Parameter Efficient Fine Tuning (PEFT) with Low-Rank Adaptation (LoRA) and demonstrate its capabilities through an innovative application. The application, *AI Travel Planner*, is a Gradio-based interface that generates personalized travel recommendations based on user input.

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

#### **(a) Model-Centric Approach**

The model-centric improvements focused on optimizing the fine-tuning process, enhancing memory efficiency, and maximizing the performance of the fine-tuned model. Below are the strategies and configurations implemented:

1. **Hyperparameter Tuning**:
   - **Batch Size and Gradient Accumulation**:
     - Set `per_device_train_batch_size` to 2 to manage memory constraints of the NVIDIA T4 GPU while effectively simulating a batch size of 8 using `gradient_accumulation_steps` set to 4. This balances memory utilization with training stability.
   - **Learning Rate**:
     - Chose a learning rate of `2e-4`, identified through experimentation to ensure optimal convergence without overfitting.
   - **Warmup Steps**:
     - Used `warmup_steps` (5) to gradually ramp up the learning rate during the initial training phase, mitigating potential instability from large gradients.
   - **Epochs and Steps**:
     - Trained for `1` epoch and capped the total number of steps at `60` to fit within the constraints of our hardware while achieving meaningful performance.

2. **Optimization Techniques**:
   - **Memory-Efficient Optimizer**:
     - Utilized the `adamw_8bit` optimizer, which uses 8-bit precision for memory-intensive operations, enabling fine-tuning of large models on GPUs with limited memory.
   - **Precision Settings**:
     - Leveraged mixed precision (`fp16` or `bf16`) to reduce memory usage and accelerate computations without compromising model accuracy.
   - **Regularization**:
     - Applied `weight_decay` (0.01) to reduce overfitting by penalizing large weights, encouraging generalization.

3. **Fine-Tuning Framework**:
   - Used the **Unsloth framework** combined with HuggingFace’s `SFTTrainer` to streamline supervised fine-tuning. This framework is optimized for efficiency and flexibility.

4. **Efficient Checkpointing**:
   - Saved progress periodically using `save_steps` (15) and limited stored checkpoints to `1` (`save_total_limit`). This ensured continuity during interruptions without excessive disk usage.

5. **Quantization**:
   - Used 4-bit quantization during inference, significantly reducing memory and computational requirements while retaining model performance.

**Data-Centric Approach**:
- **Data Sources**: Augmented the FineTome dataset with travel-related datasets to improve domain-specific performance.
- **Data Preprocessing**: Applied techniques like text normalization and deduplication to enhance data quality.

### 2. Fine-Tuning Multiple LLMs
- Tried fine-tuning smaller models like `Llama-3.1-1B` for faster inference on CPUs.
- Experimented with HuggingFace FineTuning frameworks to explore alternative approaches.

---

## Deployment and Inference
- **Colab**: Used Google Colab with T4 GPU for fine-tuning, saving checkpoints to Google Drive.
- **HuggingFace Spaces**: Deployed a Gradio-based UI for inference, enabling interaction with the fine-tuned LLM.
- **Gradio UI**: Offers an intuitive interface for querying the model and generating travel plans.

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
