import requests
import time
import sounddevice as sd
import io
from scipy.io.wavfile import read


HEADERS = {'Authorization': f'Bearer {API_TOKEN}'}

# get my actor
r = requests.get('https://typecast.ai/api/actor', headers=HEADERS)
my_actors = r.json()['result']
my_first_actor = my_actors[0]
# my_first_actor_id = my_first_actor['actor_id']

# request speech synthesis
r = requests.post('https://typecast.ai/api/speak', headers=HEADERS, json={
    'text': '안녕하세요 누리카세입니다. 무엇을 도와드릴까요?',
    'lang': 'auto',
    'actor_id': '6010088f885570093ad24d53',
    'xapi_hd': True,
    'model_version': 'latest'
})
speak_url = r.json()['result']['speak_v2_url']

# polling the speech synthesis result
for _ in range(120):
    r = requests.get(speak_url, headers=HEADERS)
    ret = r.json()['result']
    # audio is ready
    if ret['status'] == 'done':
        # download audio file
        audio_data = requests.get(ret['audio_download_url']).content
        audio_stream = io.BytesIO(audio_data)  # convert to stream
        sample_rate, audio = read(audio_stream)  # get sample rate and audio data
        sd.play(audio, samplerate=sample_rate)  # play audio
        sd.wait()  # wait until playback is finished
        break