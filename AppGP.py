import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, request, jsonify, render_template_string
import time
import numpy as np
import torch
import json
import librosa
from transformers import AutoTokenizer, AutoModel, WhisperProcessor, WhisperForConditionalGeneration
from pinecone import Pinecone, ServerlessSpec
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return obj.__dict__
        except AttributeError:
            return str(obj)

class PineCone_Vdb:
    def __init__(self, api_key, index_name="verse-index", dimensions=768, cloud="aws", region="us-west-2"):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimensions = dimensions
        existing_indexes = self.pc.list_indexes().names()
        if index_name not in existing_indexes:
            spec = ServerlessSpec(cloud=cloud, region=region)
            self.pc.create_index(name=index_name, dimension=dimensions, metric="cosine", spec=spec)
            while True:
                description = self.pc.describe_index(index_name)
                if description["status"]["ready"]:
                    break
                time.sleep(1)
        else:
            print(f"Index '{index_name}' already exists. Connecting to it.")
        self.index = self.pc.Index(index_name)

    def describe_index_stats(self):
        return self.index.describe_index_stats()

    def get_knn(self, k, vector, namespace="ns1"):
        query = self.index.query(
            vector=vector,
            top_k=k,
            include_values=False,
            include_metadata=True,
            namespace=namespace
        )
        return query

tokenizer = AutoTokenizer.from_pretrained("pourmand1376/arabic-quran-nahj-sahife")
model = AutoModel.from_pretrained("pourmand1376/arabic-quran-nahj-sahife", output_hidden_states=True)

def get_sentence_embeddingnew2(sentence):
    if not sentence or not isinstance(sentence, str):
        return np.zeros(model.config.hidden_size)
    inputs = tokenizer(
        sentence,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_state = outputs.last_hidden_state  # shape: (batch_size, seq_length, hidden_size)
    embeddings = torch.mean(last_hidden_state, dim=1)  # shape: (batch_size, hidden_size)
    return embeddings.numpy().flatten()

whisper_processor = WhisperProcessor.from_pretrained("tarteel-ai/whisper-base-ar-quran")
whisper_model = WhisperForConditionalGeneration.from_pretrained("tarteel-ai/whisper-base-ar-quran")

def transcribe_audio(file_path):
    """
    Loads a WAV audio file, transcribes it using Whisper, and returns the transcription text.
    """
    # Load audio at 16kHz sampling rate using librosa
    audio_waveform, sr = librosa.load(file_path, sr=16000)
    input_features = whisper_processor(audio_waveform, sampling_rate=sr, return_tensors="pt").input_features
    with torch.no_grad():
        predicted_ids = whisper_model.generate(input_features)
    transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

app = Flask(__name__)
app.json_encoder = CustomJSONEncoder

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Query Interface</title>
    <script>
        async function sendTextQuery(event) {
            event.preventDefault();
            const text = document.getElementById("inputText").value;
            const response = await fetch("/query", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ text: text })
            });
            const result = await response.json();
            document.getElementById("textResult").innerText = JSON.stringify(result, null, 2);
        }
    </script>
</head>
<body>
    <h1>Text Query</h1>
    <form onsubmit="sendTextQuery(event)">
        <textarea id="inputText" rows="4" cols="50" placeholder="Enter your text here..."></textarea>
        <br>
        <button type="submit">Submit Text</button>
    </form>
    <h2>Text Query Result:</h2>
    <pre id="textResult"></pre>
    <hr>
    <h1>Audio Query</h1>
    <form id="audioForm" enctype="multipart/form-data">
        <input type="file" id="audioFile" name="audio_file" accept=".wav">
        <br><br>
        <button type="button" onclick="sendAudioQuery()">Submit Audio</button>
    </form>
    <h2>Audio Query Result:</h2>
    <pre id="audioResult"></pre>
    <script>
        async function sendAudioQuery() {
            const form = document.getElementById("audioForm");
            const formData = new FormData(form);
            const response = await fetch("/query_audio", {
                method: "POST",
                body: formData
            });
            const result = await response.json();
            document.getElementById("audioResult").innerText = JSON.stringify(result, null, 2);
        }
    </script>
</body>
</html>
"""
API_KEY = "pcsk_3SdXoo_6a9eQ7UjWR7cezBPT7k7jko77EA2oRr3wYuZNPXFUTnUiHH9x3kL9oMp2bBT49n"  
VECTOR_DIMENSION = 768
vdb = PineCone_Vdb(api_key=API_KEY, index_name="verse-index", dimensions=VECTOR_DIMENSION)
def filter_result(result):
    filtered = {}
    if "matches" in result:
        filtered["matches"] = []
        for match in result["matches"]:
            filtered_match = {
                "id": match.get("id"),
                "metadata": match.get("metadata"),
            }
            filtered["matches"].append(filtered_match)
    return filtered
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)
@app.route('/stats', methods=['GET'])
def stats():
    stats_obj = vdb.describe_index_stats()
    return jsonify(stats_obj), 200
@app.route('/knn', methods=['POST'])
def knn():
    data = request.get_json()
    if not data or 'vector' not in data or 'k' not in data:
        return jsonify({"error": "Please provide a 'vector' and 'k' in the payload."}), 400
    query_vector = data['vector']
    k = data['k']
    result = vdb.get_knn(k, query_vector, namespace="ns1")
    filtered = filter_result(result)
    return jsonify(filtered), 200
@app.route('/query', methods=['POST'])
def query_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Please provide 'text' in the payload."}), 400
    text = data['text']
    k = 1
    vector = get_sentence_embeddingnew2(text)
    result = vdb.get_knn(k, vector.tolist(), namespace="ns1")
    if result.get("matches") and result["matches"]:
        top_score = result["matches"][0].get("score", 0)
        if top_score < 0.8:
            return jsonify({"message": "Not found"}), 200
    filtered = filter_result(result)
    return app.response_class(
        response=json.dumps(filtered, default=str),
        mimetype='application/json'
    ), 200
@app.route('/query_audio', methods=['POST'])
def query_audio():
    if "audio_file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    file = request.files["audio_file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    temp_filepath = "temp_audio.wav"
    file.save(temp_filepath)
    try:
        transcription = transcribe_audio(temp_filepath)
        embedding_vector = get_sentence_embeddingnew2(transcription)
        knn_result = vdb.get_knn(k=1, vector=embedding_vector.tolist(), namespace="ns1")
        if knn_result.get("matches") and knn_result["matches"]:
            top_score = knn_result["matches"][0].get("score", 0)
            if top_score < 0.8:
                return jsonify({"message": "Not found"}), 200

        filtered = filter_result(knn_result)
        response_data = {
            "transcription": transcription,
            "knn_result": filtered
        }
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
    
    return app.response_class(
        response=json.dumps(response_data, default=str),
        mimetype='application/json'
    ), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
