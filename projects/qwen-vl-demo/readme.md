Qwen-VL Multimodal Model Deployment

An interactive local deployment of the Qwen2.5-VL-3B-Instruct model using Hugging Face Transformers and Gradio. This project enables users to interact with the model via text or image inputs and switch between different system roles.

Purpose
Demonstrate the integration of large vision-language models into user-friendly interfaces.
Explore role-based prompting for behavior control in AI systems.
Practice model loading, tokenization, and generation pipelines.

Features
Local model loading with `device_map="auto"` and `float16` precision for GPU efficiency.
Dynamic system prompt switching via radio buttons for different AI behaviors.
Support for both text-only and image+text inputs.
Leverages AI-assisted development (e.g., ChatGPT, Qwen) for efficient code generation and debugging.

Requirements
To run this project, you need Python 3.8 or higher. Install the required packages using pip:

How to Run:
Clone this repository or navigate to the project directory.
Ensure all requirements from requirements.txt are installed.
Run the script:
python qwen_demo.py

Open your browser and go to http://localhost:7860 to access the interface.


Model: Qwen/Qwen2.5-VL-3B-Instruct by Alibaba Cloud, https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct

Libraries: Hugging Face Transformers , Gradio


