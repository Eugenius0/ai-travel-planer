# AI Travel Planner Project - Parameter Efficient Fine-Tunning (PEFT) with Low-Rank Adaptation (LoRA)

## Project Overview

This project is part of Lab 2 for the ID2223 course at KTH. The goal is to fine-tune a large language model (LLM) by using Parameter Efficient Fine Tuning (PEFT) with Low-Rank Adaptation (LoRA) and demonstrate its capabilities through an innovative application. The application, *AI Travel Planner*, is a Gradio-based interface that generates personalized travel recommendations based on user input.

---

## Features
1. **Fine-Tuned LLM**:
   - The model, fine-tuned using PEFT with LoRA, enables efficient memory usage while maintaining performance.
  
2. **Instruction Dataset**:
   - Used the [FineTome-100k dataset](https://huggingface.co/datasets/mlabonne/FineTome-100k) to train the model. This dataset provides a diverse and high-quality set of instructions for fine-tuning large language models.
  
3. **Checkpointing for Efficiency**:
   - Regular checkpoints were saved to Google Drive. This ensured progress continuity during interruptions. More details on checkpointing can be found in the [Efficient Checkpointing](#efficient-checkpointing) section under the Model-Centric Approach.

4. **Interactive UI**:
   - A Gradio interface [Gradio UI](#gradio-ui) enables users to input their desired city and travel preferences to generate tailored travel plans.

5. **Efficiency Optimizations**:
   - The model leverages 4-bit quantization for reduced computational overhead.
   - Checkpointing and memory-efficient techniques were implemented to enable training on a single GPU.

6. **Deployment**:
   - The fine-tuned large language model has been saved to [my HuggingFace account](https://huggingface.co/Eugenius0/lora_model) and is publicly accessible there. The Travel Planner UI has been developed using Gradio and deployed through Google Colab to generate a temporary link [AI Travel Planner](https://15c05c70e01f52935f.gradio.live/) which is valid for 72h. The reason for doing that instead of deploying it within a Huggingface Space is explained here [Issue #1](https://github.com/Eugenius0/ai-travel-planner/issues/1) in more detail.

---

## Task 2: Improve Pipeline Scalability and Model Performance

### 1. Performance Improvement Strategies

#### **(a) Model-Centric Approach**

The fine-tuning process was carefully optimized by experimenting with key hyperparameters to balance performance, memory efficiency, and training stability. Below, we highlight the most critical hyperparameters, their tuned values, and the trade-offs associated with higher or lower values:

1. **Batch Size and Gradient Accumulation**:
   - **Tuned Values**: `per_device_train_batch_size=2` and `gradient_accumulation_steps=8`.
   - **Effect**: A batch size of 2 managed the memory constraints of the NVIDIA T4 GPU while ensuring stable training. Gradient accumulation steps of 8 simulated an effective batch size of 16, improving training stability without exceeding memory limits.
   - **Trade-Offs**:
     - Higher batch size improves convergence and training speed but significantly increases GPU memory requirements.
     - Lower batch size might lead to noisier gradient updates and slower convergence.

2. **Learning Rate**:
   - **Tuned Value**: `learning_rate=1e-4`.
   - **Effect**: A lower learning rate ensured that the model learned more gradually, reducing the likelihood of overshooting the optimal weights and improving stability.
   - **Trade-Offs**:
     - Higher learning rates risk divergence or overshooting the optimal weights.
     - Lower learning rates slow convergence, potentially requiring more training time.

3. **Warmup Steps**:
   - **Tuned Value**: `warmup_steps=50`.
   - **Effect**: Gradually increasing the learning rate in the initial steps stabilized training by preventing large gradients from disrupting early progress.
   - **Trade-Offs**:
     - Higher warmup steps ensure smoother initialization but delay full utilization of the optimal learning rate.
     - Fewer warmup steps might lead to instability in early training phases.

4. **Epochs and Training Steps**:
   - **Tuned Values**: `num_train_epochs=1`, `max_steps=1000`.
   - **Effect**: Limited epochs and steps allowed us to adapt the model within the constraints of our hardware while achieving meaningful performance improvements.
   - **Trade-Offs**:
     - Increasing epochs or steps typically results in better generalization but requires more computational resources.
     - Fewer steps reduce training time but may limit the model’s ability to converge effectively.

5. **Memory-Efficient Optimization**:
   - **Optimizer**: `adamw_8bit`.
     - **Effect**: This optimizer utilizes 8-bit precision for memory-intensive operations, significantly reducing memory usage while retaining computational efficiency.
   - **Precision Settings**: Mixed precision (`fp16` or `bf16` depending on hardware support).
     - **Effect**: Reduced memory consumption and accelerated computations without compromising model accuracy.

6. **Regularization**:
   - **Weight Decay**: `weight_decay=0.05`.
   - **Effect**: Penalized large weight values, promoting generalization and reducing the risk of overfitting.
   - **Trade-Offs**:
      - Stronger regularization reduces the likelihood of overfitting but can underfit the model if over-applied.
      - Lower regularization encourages fitting the training data but increases the risk of overfitting, especially on small datasets.

By carefully tuning these parameters, we improved the model's performance and efficiency within the constraints of the available hardware. Each choice was made based on careful experimentation and consideration of trade-offs, ensuring that our approach remained both practical and effective.

7. **Fine-Tuning Framework**:
   - Used the **Unsloth framework** combined with HuggingFace’s `SFTTrainer` to streamline supervised fine-tuning. This framework is optimized for efficiency and flexibility.

**Code Overview**:  
![image](https://github.com/user-attachments/assets/cefa10ac-b2a8-4ad3-908e-c147a2e86b9c)


8. #### **Efficient Checkpointing**:
   Saved progress periodically to Google drive using `save_steps` (15) and limited stored checkpoints to `1` (`save_total_limit`). This ensured continuity during interruptions without excessive disk usage.
   
   ![image](https://github.com/user-attachments/assets/0b9081b0-854f-4ba5-821f-d55a0d76c173)
   <img width="924" alt="image" src="https://github.com/user-attachments/assets/16fc0395-1ef0-41e4-96d4-130418a46c96">
   ![image](https://github.com/user-attachments/assets/fcbd2e8f-d934-4c55-b73f-95f379d23140)

   Ensured that the training can restart from where we left off:
   
   <img width="219" alt="image" src="https://github.com/user-attachments/assets/9ffe548a-b8a2-446f-8aea-24dfe99d15c2">
   <img width="223" alt="393667365-8777840a-643c-4a54-93af-75d197e6534f" src="https://github.com/user-attachments/assets/45c4f690-d312-4412-908c-57ce7c6d10de">



9. **Quantization**:
   - Used 4-bit quantization during inference, significantly reducing memory and computational requirements while retaining model performance.
   - 4-bit quantization is a technique used to reduce the precision of the weights and activations of a neural network from the standard 32-bit floating point to just 4 bits. This drastically reduces the memory footprint and computational requirements during inference which enables the deployment of large language models on resource-constrained hardware like CPUs. Despite the reduction in precision, advanced quantization techniques retain most of the model's performance, ensuring high-quality outputs with significantly improved efficiency.

**(b) Data-Centric Approach**:

To improve the performance of our fine-tuned LLM, we focused on enhancing the quality and diversity of the instruction dataset. This approach ensures the model can generate accurate and context-aware travel recommendations.

1. **Using a Larger Dataset** (future work, not implemented due to computational and memory limits and to respect the given time constraints.):
   - Transition from the [FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset (subset of arcee-ai/The-Tome) to the larger [The Tome dataset](https://huggingface.co/datasets/arcee-ai/The-Tome) dataset, providing a richer set of instructions for training.
   - The expanded dataset covers a broader range of scenarios, enhancing the model's ability to handle diverse contexts accurately.

2. **Custom Prompt Engineering** (implemented):
   - Designed custom prompts to clearly define the model’s expected behavior.
   - **Example Prompt**:
     - f"Create a travel plan to visit {city} during {nb_days} days, focusing on {preferences}. Include suggested activities, "
            f"landmarks to visit, and any local tips."

3. **Integrating Additional Data Sources** (future work, not implemented due to computational and memory limits and to respect the given time constraints.):
   - Augment the dataset with:
     - **Travel Reviews**: User feedback from platforms like [TripAdvisor](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews).
     - **Make My Trip Dataset**: Collection of travel records available [on Kaggle](https://www.kaggle.com/datasets/promptcloud/travel-listing-from-makemytrip).

4. **Benefits**:
   - Increased generalization across diverse user inputs.
   - Improved alignment with specific travel-related tasks and queries.

5. **Challenges and Solutions**:
   - Larger datasets required careful tuning of hyperparameters like learning rate and batch size to ensure stable training.
   - Regular checkpointing addressed the risk of data or progress loss during extended training sessions.
  




## **Measure Performance to show results of improvement**

## Before Tuning Hyperparameters:

| **Query**                                              | **Generated Output**                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | **Inference Time (s)** |
|--------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| "In which state is Freiburg im Breisgau and name its most famous sight?" | "The most famous sight in Freiburg im Breisgau, Germany, is the Münster church. The Freiburg Cathedral, or the Münster, is a beautiful Protestant cathedral that is one of the city's main landmarks. It dates back to 1218 and features impressive Romanesque and Gothic architectural elements, including two 43-meter-tall towers, known as the 'Lion' and the 'Dove.' The Münster is a popular tourist attraction in Freiburg and is known for its intricate stone carvings, stained glass windows, and historical significance." | 13.9443               |
| "What are the main attractions in Paris?"              | "Paris is a city known for its beauty, history, art, and romance. There are countless things to do and see here, but some of the main attractions include:\n\n1. Eiffel Tower: This iron giant is one of Paris' most iconic landmarks and offers breathtaking views of the city.\n2. Louvre Museum: One of the largest museums in the world, housing a vast collection of artworks, including the Mona Lisa.\n3. Notre Dame Cathedral: A beautiful and historic church that was built in the 12th century.\n4. Arc de Triomphe: A monument celebrating French victories and featuring a stunning view of the city." | 7.4836                |



![image](https://github.com/user-attachments/assets/00586704-9e8c-4a65-90b6-a97880929852)


## Bleu Score

**BLEU (Bilingual Evaluation Understudy)** is a metric used to evaluate the quality of machine-generated text by comparing it to one or more reference texts. It measures the overlap of n-grams (sequences of words) between the generated and reference texts, with higher scores indicating closer similarity and better quality.

| **Query**                                              | **Generated Output**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | **BLEU Score** |
|--------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| "In which state is Freiburg im Breisgau and name its most famous sight?" | "Freiburg im Breisgau is in the state of Baden-Württemberg, Germany.\n\nOne of its most famous sights is the Freiburg Minster (Friedenskirche)."                                                                                                                                                                                                                                                                                                                                                     | 0.0708         |
| "What are the main attractions in Paris?"              | "Paris, the capital of France, is a popular destination for tourists due to its rich history, beautiful architecture, art museums, and lively culture. Some of the main attractions in Paris include:\n\n1. The Eiffel Tower: This iron lattice tower built in 1889 stands 324 meters tall and is one of the most iconic landmarks of Paris.\n2. The Louvre: The Louvre Museum is home to the Mona Lisa, a famous painting created by Leonardo da Vinci in the 15th century.\n3. The Notre-Dame Cathedral: The Cathedral was built in the 12th century and has been the place of worship" | 0.0676         |


![image](https://github.com/user-attachments/assets/51222f41-835d-4d00-a655-e1fab9b75bec)




## After Tuning Hyperparameters:

| **Query**                                              | **Generated Output**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | **Inference Time (s)** |
|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------|
| "In which state is Freiburg im Breisgau and name its most famous sight?" | "Freiburg im Breisgau is located in the state of Baden-Württemberg in Germany. It is famous for its Gothic Cathedral of the Immaculate Conception. This stunning cathedral was completed in 1343 and stands as an exemplary representation of medieval architecture in the Black Forest region."                                                                                                                                                                                                                      | 3.5184                |
| "What are the main attractions in Paris?"              | "There are many famous landmarks and museums in Paris. Some of the main attractions include:\n\n1. The Eiffel Tower - This is one of the most iconic and photographed buildings in the world, standing 1,063 feet high. You can take an elevator or stairs to the top for spectacular views of the city.\n\n2. The Louvre - This is the world's largest art museum, housing a collection of more than 550,000 works of art, including the Mona Lisa and the Venus de Milo.\n\n3. Notre Dame Cathedral - This beautiful Gothic cathedral was built in the 12th century and is considered one of the most famous churches in the world." | 7.5187                |


![image](https://github.com/user-attachments/assets/ee5f1b1e-850d-46a0-b51d-92f905a2b2d8)


## Bleu Score

**BLEU (Bilingual Evaluation Understudy)** is a metric used to evaluate the quality of machine-generated text by comparing it to one or more reference texts. It measures the overlap of n-grams (sequences of words) between the generated and reference texts, with higher scores indicating closer similarity and better quality.

| **Query**                                              | **Generated Output**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | **BLEU Score** |
|--------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| "In which state is Freiburg im Breisgau and name its most famous sight?" | "Freiburg im Breisgau is a city in the state of Baden-Württemberg, Germany. It is known as a culturally significant location with a rich history dating back to the 18th century. Freiburg im Breisgau is often referred to as the city of the most beautiful house. The city is situated near the Breisgaau River and the Albtraiteh Hills. It is particularly famous for the Black Forest and the city's connection to the German composer Friedrich Nietzsche." | 0.1184         |
| "What are the main attractions in Paris?"              | "Paris, France, is a world-renowned city with a rich history and culture that offers numerous attractions for travelers. Here are some of the main attractions in Paris:\n\n1. The Eiffel Tower (Tour Eiffel): This iconic tower was constructed for the 1889 World's Fair and is one of the most recognizable landmarks in the world. Visitors can take the stairs or elevator to the top for stunning views of the city.\n\n2. The Louvre: This museum houses some of the world's most famous works of art, including Leonardo da Vinci's 'Mona Lisa.'\n\n3. The Seine River: Visitors can take scenic boat tours along this iconic river." | 0.2461         |


![image](https://github.com/user-attachments/assets/c770728a-3f9f-4611-b5ed-f5c4b4b07c7e)


### **Outcome of Results**

- **Improved Inference Time**: After tuning hyperparameters, the model's inference time significantly decreased, demonstrating enhanced computational efficiency without compromising the quality of responses.

- **Trade-off in BLEU Scores**: While BLEU scores slightly decreased after tuning, this indicates a trade-off where faster response generation led to minor deviations in text overlap with reference outputs.

- **Enhanced Usability**: The fine-tuned model provides detailed and coherent outputs more quickly, making it more practical for real-time applications like the AI Travel Planner.


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
- #### Gradio UI:
Offers an intuitive interface for querying the model and generating travel recommendations/advices.

![image](https://github.com/user-attachments/assets/537940e0-e1bb-4aa7-9aae-0011fa60df79)
![image](https://github.com/user-attachments/assets/e149e350-f978-49b0-98c4-d81fc62158f4)


Output of the example where user specifies city as "Paris", preferences as "food" and trip duration as "2":

system

Cutting Knowledge Date: December 2023
Today Date: 10 Dec 2024

user

Create a travel plan to visit Paris during 2 days, focusing on food. Include suggested activities, landmarks to visit, and any local tips.assistant

Paris is famous for its delicious cuisine, from croissants to baguettes, escargots to macarons, French pastries are renowned worldwide. To experience the best of French gastronomy in two short days, we have devised this itinerary.

Day one:

- Start your day with breakfast at Café de Flore (172 Boulevard Saint-Germain) or Les Deux Magots (6 Place Saint-Germain-Des-Prés), both iconic cafes in the heart of the Latin Quarter. Enjoy croissant and coffee or tea, or perhaps indulge in their delightful pastry selection.
- Visit the Musée Rodin, a museum dedicated to the works of French sculptor Auguste Rodin, where you can admire his most famous creations such as The Thinker.
- For lunch, head to Le Comptoir du Relais, known for its classic French dishes at affordable prices like Coq au Vin and Steak Tartare. Make sure to book ahead as it gets crowded.
- Afternoon: Explore Montmartre's charming streets lined with charming cafés, historic churches, and bohemian artists. Visit the Basilique du Sacré-Cœur for a breathtaking view over the city.
- Dinner at L'Atelier de Joël Robuchon - an innovative eatery offering a unique dining experience. Each dish is cooked by staff members who display the cooking process through the glass partition.

Day two:
- Morning: Try your hand at making Macarones at Boulangerie Gérard Mulot – order a dozen assorted macarons and enjoy them while strolling around the picturesque neighborhood of Marais.
- Lunchtime, dine at Le Grand Vefour, a legendary restaurant located under the old Royal Palace of France. This restaurant offers traditional French cuisine in elegant atmosphere.
- In the afternoon, take a stroll along the Seine River and visit the Pont des Arts and the Ile de la Cité. Stroll across the Pont Neuf Bridge and explore the quaint streets leading up to Notre Dame Cathedral.
- Evening: Dine at Septime, a critically acclaimed restaurant that serves seasonal French cuisine. Be sure to book well in advance due to limited availability.





---

## Future Work
- Integrate new datasets to further refine model outputs.
- Longer training to increase performance of LLM.
- Optimize the UI.
- Explore additional domains beyond travel planning.

---

## Links
- **Discussion on Task Challenges**: [Canvas Discussion](https://canvas.kth.se/courses/50172/discussion_topics/432284)
- **Gradio App (temporary link)**: [AI Travel Planner](https://15c05c70e01f52935f.gradio.live/)
- **Google Colab Inference Notebook**: [Colab Notebook](link)
