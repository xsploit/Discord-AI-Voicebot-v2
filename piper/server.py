from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/tts_to_audio/', methods=['POST'])
def generate_tts():
    data = request.get_json()
    text = data['text']
    speaker_wav = data['speaker_wav']
    language = data['language']

    output_file = 'output.wav'

    try:
        subprocess.run(['piper.exe', '-m', 'en_US-ljspeech-high.onnx', '-c', 'en_en_US_ljspeech_high_en_US-ljspeech-high.onnx.json', '--output_dir', '.', '--json_input'],
                      input=bytes(json.dumps({'text': text, 'speaker_wav': speaker_wav, 'output_file': output_file}), 'utf-8'),
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        with open(output_file, 'rb') as f:
            audio_data = f.read()
        os.remove(output_file)

        return jsonify({'audio_data': audio_data})
    except subprocess.CalledProcessError as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=8000)