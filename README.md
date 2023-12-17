# Speech-to-Image Generation with Integrated Prompt Engineering

This repository houses a sophisticated AI system for image generation leveraging the power of the SDXL Turbo model and Hugging Face Chatbot. It allows users to generate images based on prompts engineered from their voice inputs.

### Requirements
- Python >= 3.6
- Libraries: `diffusers`, `hugchat`, `transformers`, `accelerate`

### Installation
```bash
!pip install diffusers hugchat transformers accelerate --upgrade
```

### Overview
1. **Audio to Text Conversion:**
   - Utilize Google Colab's audio recording capabilities combined with OpenAI's Whisper model for speech-to-text conversion.

2. **Chatbot for Prompt Engineering:**
   - Hugging Face's chatbot functionality crafts AI-engineered prompts based on the user's voice input.

3. **Image Generation with SDXL Turbo:**
   - Use the SDXL Turbo model from Diffusers to translate the AI-engineered prompt into a detailed image.

### Usage Instructions
1. **Authenticate Services:** Set up necessary credentials for Hugging Face and initialize required environment settings.
2. **Record Voice Prompt:** Utilize the provided JavaScript code to record audio and convert it to text for prompt generation.
3. **Prompt Engineering:** Engage with the Hugging Face Chatbot to refine and engineer the prompt for image generation.
4. **Image Creation:** Run the provided code to generate images based on the AI-engineered prompt using SDXL Turbo.

### Additional Details
- Ensure the generated prompt is concise (less than 65 tokens) and comprehensively conveys the user's requirements for accurate image generation.
- Refer to the detailed documentation for advanced prompts and guidance on improving image generation quality.

### Contributors
- [Pragadeesh KMS]


### Disclaimer
This project employs advanced AI models, which might necessitate specific resources or permissions for optimal functionality.
