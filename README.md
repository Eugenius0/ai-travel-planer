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
   - 4-bit quantization is a technique used to reduce the precision of the weights and activations of a neural network from the standard 32-bit floating point to just 4 bits. This drastically reduces the memory footprint and computational requirements during inference which enables the deployment of large language models on resource-constrained hardware like CPUs. Despite the reduction in precision, advanced quantization techniques retain most of the model's performance, ensuring high-quality outputs with significantly improved efficiency.

**Data-Centric Approach**:
- **Data Sources**: Augmented the FineTome dataset with travel-related datasets to improve domain-specific performance.
- **Data Preprocessing**: Applied techniques like text normalization and deduplication to enhance data quality.

### 2. Evaluating Multiple LLMs

As part of the model-centric approach, we evaluated various open-source foundation LLMs to determine the best-performing model for our application while taking our computational and memory limits into consideration and respecting the given time constraints. The goal was to identify a model that balances computational efficiency and quality of generated responses, especially for inference on CPUs.

#### Models Evaluated:
- **unsloth/Meta-Llama-3.1-8B-bnb-4bit**: 
  - Pros: Larger model with strong general capabilities.
  - Cons: Computationally intensive and slow for inference on CPUs.

- **unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit**:
  - Pros: Smaller and faster, with good instruction-following capabilities.
  - Cons: Lacked depth and coherence in generating detailed travel plans.

- **unsloth/Phi-3.5-mini-instruct**:
  - Pros: Very lightweight, suitable for rapid inference on CPUs.
  - Cons: Struggled to maintain fluency and relevance for travel-related queries.

- **unsloth/Llama-3.2-1B-Instruct-bnb-4bit**:
  - Pros: Lightweight and efficient for CPU inference, capable of generating relevant responses.
  - Cons: Limited expressiveness compared to larger models.

- **unsloth/Llama-3.2-3B-Instruct-bnb-4bit** (Selected Model):
  - Pros:
    - Balance between computational efficiency and output quality.
    - Leveraged instruction-tuning for better alignment with user queries.
    - Offered detailed and coherent responses for travel planning tasks.
    - Compatible with 4-bit quantization, significantly reducing memory and computation requirements.
  - Cons:
    - Slightly slower than smaller models but acceptable for our use case.

#### Why Llama-3.2-3B-Instruct-bnb-4bit?
1. **Balance of Size and Performance**:
   - The 3B parameter size provided a sweet spot between the expressiveness of large models and the efficiency of smaller ones.
   - Generated highly coherent and contextually relevant travel plans, acceptable for our use case.

2. **4-Bit Quantization**:
   - Reduced memory usage and allowed for efficient inference without compromising performance.

3. **Instruction-Tuned Capabilities**:
   - Fine-tuned for instruction-following tasks, making it particularly suitable for generating structured travel itineraries.

4. **Adaptability**:
   - The model showed strong performance across a wide range of travel preferences, including historical landmarks, nightlife, and culinary experiences.

By carefully testing these models, we ensured that the chosen LLM aligns with our application’s goals and provides an optimal user experience in the AI Travel Planner.

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
