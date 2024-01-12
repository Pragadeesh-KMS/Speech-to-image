!pip install diffusers hugchat transformers accelerate safetensors --upgrade -q
!pip install git+https://github.com/huggingface/diffusers -q
!pip install ipywidgets -q
!pip install invisible_watermark -q

from transformers import pipeline
from ipywidgets import interactive, widgets
from IPython.display import HTML, Javascript, Image, display
from google.colab.output import eval_js
import base64
from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained("segmind/SSD-1B", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")
whisper = pipeline("automatic-speech-recognition", model="openai/whisper-large-v2", chunk_length_s=30, device="cuda:0")
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

# Display the initial JavaScript code for audio recording
display(js)

# Execute JavaScript to record audio
output = eval_js('recordAudio({})')
with open('audio.wav', 'wb') as file:
    binary = base64.b64decode(output)
    file.write(binary)
print('Recording saved to:', file.name)
speech_to_text = whisper("audio.wav")

# Extract text from speech-to-text result
img = speech_to_text['text']
# Initialize variables for image and display
generated_image = None
image_display = display("", display_id=True)

def generate_image(button):
    global generated_image

    # Retrieve values from sliders
    image_style = style_slider.value
    image_quality = quality_slider.value
    render = render_slider.value
    angle = angle_slider.value
    lighting = lighting_slider.value
    background = background_slider.value
    device = device_slider.value
    emotion = emotion_slider.value

    # Construct the prompt based on user selections
    prompt = f"A stunning {image_style}, {image_quality} shot of {img} captured in {device} using {angle} and rendered by {render}, illuminated by {lighting} light, with {emotion} emotions in a {background} background setting."

    # Use the DiffusionPipeline to generate an image based on the prompt
    neg_prompt = "ugly, blurry, poor quality, deformed structure, very bad lighting, bad colouring, noise"  # Negative prompt here
    generated_image = pipe(prompt=prompt, negative_prompt=neg_prompt).images[0]

    # Save and display the generated image
    generated_image.save("generated_image.png")
    final_image_path = "generated_image.png"
    image_display.update(Image(filename=final_image_path))

# slider options
image_style_options = ["photorealistic", "low poly", "cinematic", "cartoon", "pixel art", "graffiti", "sketching"]
image_quality_options = ["High resolution", "8K", "clear","pixelated NFT", "heavy detailed", "beautiful", "realistic+++", "hyper detailed", "masterpiece"]
render_options = ["Pixar","Octane", "real-time ray tracing", "Christopher Nolan", "James Cameron", "unreal engine", "unity" ]
angle_options = ["Wide-angle lens", "full shot", "Top angle" "Telephoto lens", "Prime lens", "Zoom lens", "Macro lens", "Fisheye lens", "Tilt-shift lens", "Portrait lens", "Anamorphic lens", "Cinematic lens", "Fixed focal length lens", "Variable focal length lens"]
lighting_options = ["Soft", "ambient", "ring" "light", "neon", "Natural", "Soft", "Harsh", "Dramatic", "Backlit", "Studio"]
background_options = ["outdoor", "indoor", "space", "nature", "sci-fi", "neon", "abstract"]
device_options = ["Go Pro", "Iphone 15", "Canon EOS R5","Nikon Z7", "Sony F950", "Drone", "CCTV",]
emotion_options = ["Happy", "Sad", "Mysterious", "Surprised", "Annoyed", "Neutral", "dreamy", "nostalgic"]

# sliders
style_slider = widgets.SelectionSlider(options=image_style_options, description="Style:")
quality_slider = widgets.SelectionSlider(options=image_quality_options, description="Quality:")
render_slider = widgets.SelectionSlider(options=image_quality_options, description="Render by:")
angle_slider = widgets.SelectionSlider(options=angle_options, description="Angle:")
lighting_slider = widgets.SelectionSlider(options=lighting_options, description="Lighting:")
background_slider = widgets.SelectionSlider(options=background_options, description="Background:")
device_slider = widgets.SelectionSlider(options=background_options, description="Device:")
emotion_slider = widgets.SelectionSlider(options=background_options, description="Emotion:")

generate_button = widgets.Button(description="Generate Image")

generate_button.on_click(generate_image)

# interactive widget
interactive_widget = widgets.VBox([
    style_slider, quality_slider, angle_slider, render_slider,
    lighting_slider, background_slider, device_slider, emotion_slider,
    generate_button
])

# Display the  widget
display(interactive_widget)
