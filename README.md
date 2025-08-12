## 1. Introduction 
In the era of large language models, model merging has emerged as a promising technique for integrating multiple task-specific models into a unified multitask model without additional training. However, as the number of tasks grows, conventional model merging approaches suffer from increasingly severe parameter conflicts and task interference. Worse still, the coarse-grained strategies adopted in traditional methods lead to a mismatch between internal communication mechanisms and the merged model structure, and also lack the flexibility needed to manage diverse tasks—especially as the number of models increases.
In this paper, we demonstrate that investigating the relationship between neurons and knowledge is essential for addressing parameter contradictions and task conflicts. We further show that updating the internal communication mechanisms significantly improves the utilization of merged knowledge. 
First, inspired by mutual information theory, we examine the correlation between internal neurons and task-specific signals to identify and prune low-relevance neurons. This selective pruning process retains only highly informative neurons, significantly reducing parameter contradictions.
Second, we introduce a dynamic hierarchical merging strategy guided by neuron-task correlations, which preserves core domain-specific knowledge while effectively integrating cross-domain information. Third, we enhance the routing mechanism by dynamically activating the most relevant expert modules based on the input data, thereby maximizing expert utilization and improving task adaptability. To our knowledge, TE-MERGING is the first method that enables merged models to outperform fine-tuned models in single-task performance, effectively mitigating parameter conflicts. Moreover, our approach shows strong potential to scale to the integration of thousands of models.

## 2. Code Architecture
 ├────── README.md              # Introduction and instructions for running the code
 ├────── ora_vit.ipynb          # Fine-tuning task-specific models using LoRA
 ├────── train_router.ipynb     # Training the router to select relevant model components
 ├────── merge_test.ipynb       # Merging task-specific models and evaluating performance
 ├────── util.py                # Utility functions for data processing and preprocessing

## 3. How to Run the Code

To reproduce our results or use the provided implementation, please follow the steps below:

### Step 1: Install Dependencies

First, ensure you have Python installed (we recommend Python 3.8+). Then, install all required packages by running:

<pre><code>```pip install -r requirements.txt ``` </code></pre>

This will install all dependencies listed in the requirements.txt file, including PyTorch, Transformers, and other supporting libraries.

### Step 2: Prepare and Fine-Tune the Model
Next, open and run the following notebook to apply LoRA-based fine-tuning to a ViT model:

<pre><code>```jupyter notebook lora_vit.ipynb ``` </code></pre>

This notebook performs low-rank adaptation of a Vision Transformer using task-specific datasets.

### Step 3: Train the Router

After fine-tuning, train the router module that dynamically selects expert layers based on input relevance:

<pre><code>```jupyter notebook train_router.ipynb ``` </code></pre>

This step involves training a lightweight routing network that guides expert selection during inference.

### Step 4: Merge and Evaluate

Finally, merge the fine-tuned models and evaluate performance using the following notebook:

<pre><code>```jupyter notebook merge_test.ipynb ``` </code></pre>

This step demonstrates how the merged model performs across multiple tasks and showcases the benefits of the proposed model merging strategy.

Feel free to open an issue if you encounter any problems during setup or training. We welcome contributions and feedback!