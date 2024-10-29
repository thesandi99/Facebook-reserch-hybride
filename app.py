import os
import argparse
import time
from pathlib import Path
from flask import Flask, request, render_template, redirect, url_for, flash, send_file
import io
import torch as th
import torchaudio
from werkzeug.utils import secure_filename
from demucs.apply import apply_model
from demucs.audio import save_audio
from demucs.pretrained import get_model_from_args, ModelLoadingError

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your_secret_key_here")

# Check for allowed file types
ALLOWED_EXTENSIONS = {"mp3", "wav"}

# Configure device and model
device = "cuda" if th.cuda.is_available() else "cpu"
args = argparse.Namespace(
    model="htdemucs",
    device=device,
    shifts=1,
    overlap=0.25,
    stem=None,
    int24=False,
    float32=False,
    clip_mode="rescale",
    mp3=True,  # Set to true for mp3 output
    mp3_bitrate=320,
    filename="{track}/{stem}.{ext}",
    split=True,
    segment=None,
    name="htdemucs",
    repo=None
)

try:
    model = get_model_from_args(args)
    model.eval().to(device)
except ModelLoadingError as error:
    print(f"Model loading error: {error}")

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            # Run the separator directly on the uploaded file without saving
            try:
                output_files = separator(file)
                # Render a template with download links for each output file
                return render_template("download.html", output_files=output_files)
            except Exception as e:
                flash(f"Error processing file: {str(e)}")
            return redirect(url_for("upload_file"))

    return render_template("upload.html")

def load_audio(file):
    try:
        waveform, sample_rate = torchaudio.load(file)
        return waveform.to(device), sample_rate
    except Exception as e:
        flash(f"Error loading audio file: {str(e)}")
        return None, None

def separator(file, max_duration=170):
    output_files = []
    wav, sample_rate = load_audio(file)
    if wav is None:
        return output_files

    max_samples = int(sample_rate * max_duration)
    wav = wav[:, :max_samples]

    # Process with the model
    sources = apply_model(model, wav[None], device=device, shifts=1, overlap=0.25)[0]

    for source, name in zip(sources, model.sources):
        # Save each source to an in-memory file
        output_io = io.BytesIO()
        save_audio(source[:, :max_samples], output_io, samplerate=sample_rate, bitrate=args.mp3_bitrate)
        output_io.seek(0)
        
        # Append to the output files list as a tuple (filename, file)
        output_files.append((f"{name}.mp3", output_io))

    return output_files

@app.route("/download/<filename>")
def download_file(filename):
    # Stream the requested file
    file_obj = next((file for name, file in output_files if name == filename), None)
    if file_obj:
        return send_file(file_obj, as_attachment=True, download_name=filename, mimetype="audio/mpeg")
    else:
        flash("File not found.")
        return redirect(url_for("upload_file"))

if __name__ == "__main__":
    app.run()
