import os
from flask import Flask, request, render_template
import whisper
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Ensure ffmpeg is accessible
os.environ["PATH"] += os.pathsep + "C:\\ffmpeg\\bin"
os.environ["PATH"] += os.pathsep + "C:\\ffmpeg\\ffmpeg-7.1.1-essentials_build\\bin"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
whisper_model = whisper.load_model("base")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-small")

def summarize_text(text):
    input_text = "summarize: " + text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    output_ids = t5_model.generate(input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        audio = request.files["audio"]
        if audio:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], audio.filename)
            audio.save(filepath)
            transcript = whisper_model.transcribe(filepath)["text"]
            summary = summarize_text(transcript)
    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
