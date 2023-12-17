from transformers import pipeline
from diffusers import DiffusionPipeline, AutoPipelineForText2Image
import torch
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
import base64
from hugchat import hugchat
from hugchat.login import Login

whisper = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2", chunk_length_s=30, device="cuda:0")
pipeline = DiffusionPipeline.from_pretrained("stabilityai/sdxl-turbo")
pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

email = "Hugging face mail id"
passwd = "Password"

sign = Login(email, passwd)
cookies = sign.login()
cookie_path_dir = "./cookies_snapshot"
sign.saveCookiesToDir(cookie_path_dir)

js = Javascript(
    """
    async function recordAudio() {
      const div = document.createElement('div');
      const audio = document.createElement('audio');
      const strtButton = document.createElement('button');
      const stopButton = document.createElement('button');

      strtButton.textContent = 'Start Recording';
      stopButton.textContent = 'Stop Recording';

      document.body.appendChild(div);
      div.appendChild(strtButton);
      div.appendChild(audio);

      const stream = await navigator.mediaDevices.getUserMedia({audio:true});
      let recorder = new MediaRecorder(stream);

      audio.style.display = 'block';
      audio.srcObject = stream;
      audio.controls = true;
      audio.muted = true;

      await new Promise((resolve) => strtButton.onclick = resolve);
        strtButton.replaceWith(stopButton);
        recorder.start();

      await new Promise((resolve) => stopButton.onclick = resolve);
        recorder.stop();
        let recData = await new Promise((resolve) => recorder.ondataavailable = resolve);
        let arrBuff = await recData.data.arrayBuffer();
        stream.getAudioTracks()[0].stop();
        div.remove()

        let binaryString = '';
        let bytes = new Uint8Array(arrBuff);
        bytes.forEach((byte) => { binaryString += String.fromCharCode(byte)});

      const url = URL.createObjectURL(recData.data);
      const player = document.createElement('audio');
      player.controls = true;
      player.src = url;
      document.body.appendChild(player);

    return btoa(binaryString)

          };
          """
)
display(js)

output = eval_js('recordAudio({})')
with open('audio.wav','wb') as file:
    binary = base64.b64decode(output)
    file.write(binary)
print('Recording save to:', file.name)

speech_to_text = whisper("audio.wav")
image_prompt = speech_to_text['text']

System_job = '''
[Your Expertise]: You are an expert in creating prompt using normal plain text sentences. If a user gives you certain sentences, you should engineer the sentence into a image creating perfect prompt.

[User's job]: Users will give you a sentence for which they want the image

[Your job]: You should take the input, which is the user's plain text for creating a image and should perform prompt engineering to make that plain text to a engineer prompt which an AI ca understand to create a image.
          Once after they give you their sentence {image_prompt} to develop a engineered prompt. You should ask the user the following 4 questions IN A SINGLE ATTEMPT with 6-8 options each question for the user to choose. You should give these options based on the user's sentence.
          1. What is your preferred style of image?: It can be photorealistic, low poly, cinematic, cartoon, abstract, surreal, pixel art, graffiti, sketching, minimal, pop art and so on.
          2. What is your preferred quality of image?: It can be High resolution, 2K, 4K, 8K, clear, good lighting, detailed, extremely detailed, sharp focus, intricate, beautiful, realistic+++, complementary colors, high quality, hyper detailed, masterpiece, best quality, artstation, stunning.
          3. What is your preferred shot type?: It can be Wide Shot/Establishing Shot, Long Shot, Full Shot, Medium Shot, Cowboy Shot, Medium Close-Up, Close-Up, Extreme Close-Up, Two-Shot, Over-the-Shoulder Shot, Point-of-View Shot (POV) and so on.
          4. What is your perferred background?: Good lighting, natural, realistic+++ and so on. For this you can give options by recognising the user's prompt and what background will suit for that.
 You should ask only 4 questions to the user which are given above and give options to them also. Once after you got all the queries cleared, then you can proceed to create the perfect prompt to generate an image and give it to them.

[Additional Information] - There are certain things in image prompting which you should ask the user, but you can conclude the based on the user's sentence and the options they choose for your four questions, they are:
1. The emphasis on the word "very" seems to improve generation quality! Repetition can also be used to emphasize subject terms. For example, if you want to generate an image of a planet with aliens, using the prompt A planet with aliens aliens aliens aliens aliens aliens aliens aliens aliens aliens aliens aliens will make it more likely that aliens are in the resultant image.
2. If we want mountains without trees, we can use the prompt mountain | tree:-10. Since we weighted tree very negatively, they do not appear in the generated image. Weighted terms can be combined into more complicated prompts, like A planet in space:10 | bursting with color red, blue, and purple:4 | aliens:-10 | 4K, high quality
3. Using a robust negative prompt, we can generate much more convincing iamges without deformed images. For example: studio medium portrait of Brad Pitt waving his hands, detailed, film, studio lighting, 90mm lens, by Martin Schoeller:6 | disfigured, deformed hands, blurry, grainy, broken, cross-eyed, undead, photoshopped, overexposed, underexposed, lowres, bad anatomy, bad hands, extra digits, fewer digits, bad digit, bad ears, bad eyes, bad face, cropped: -5
   Using a similar negative prompt can help with other body parts as well. Unfortunately, this technique is not consistent, so you may need to attempt multiple generations before getting a good result. In the future, this type of prompting should be unnecessary since models will improve. However, currently it is a very useful technique.
4. Clarity of Description: Ensure your prompt is clear and detailed enough to convey the scene or concept you want to depict. Specificity helps the artist understand your vision.
5. Emotion or Mood: Determine the emotional tone or mood you want the image to convey. This could be happiness, sadness, mystery, etc. Communicate this in your prompt.


Atlast, keep in mind that the final prompt given by you should be short and sweet, less than 65 tokens, but still should convey all the necessary informations to generate the perfect image.
'''
print(chatbot.chat(System_job))
print(chatbot.chat(image_prompt))
print(chatbot.chat(input()))

Engineered_prompt = chatbot.chat("Can you give me your engineered prompt alone inside quotations with all my selected options included")
print("Engineered Prompt:")
print(Engineered_prompt)

prompt = Engineered_prompt
image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
image.save("image.png")
final_image_path = "image.png"
display(Image(filename=final_image_path))
