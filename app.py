# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import azure.cognitiveservices.speech as speechsdk

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'fake-news-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl', 'rb'))
app = Flask(__name__)



def text_to_speech():
    speech_config = speechsdk.SpeechConfig(subscription="d997a6cd637745089586b8f7b9e02f6f", region="westeurope")
    speech_config.speech_recognition_language = "en-US"
    #speech_config.speech_synthesis_voice_name = "<your-wanted-voice>"

    #audio_config = speechsdk.audio.AudioOutputConfig(filename="path/to/write/file.wav")
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    print("Speech to Text output is")
    #synthesizer.speak_text_async("FAKE NEWS")

    #Synthesize to speaker output
    #audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    return synthesizer



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vector = cv.transform(data).toarray()
        my_prediction = classifier.predict(vector)
        return render_template('result.html', prediction=my_prediction, synthesizer = synthesizer)



if __name__ == '__main__':
    synthesizer = text_to_speech()
    app.run(debug=True)
