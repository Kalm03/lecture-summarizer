from transformers import BartTokenizer, BartForConditionalGeneration
import torch

from fastai2 import *
import logging
import sys

import os
import tempfile
from threading import *

import ffmpeg

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'august-strata-313904-6e727bacc17c.json'

from google.cloud import storage
from google.cloud import speech

sys.path.append('..')
logging.getLogger().setLevel(100)
from transformers import BartTokenizer, BartForConditionalGeneration

from tkinter import * 
from tkinter import filedialog
from tkinter.filedialog import asksaveasfile

import random

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

def getTranscript(gcs_uri):
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(uri=gcs_uri)
    config = speech.RecognitionConfig(
        language_code='en-US',
        audio_channel_count=2,
    )

    operation = client.long_running_recognize(config=config, audio=audio)
    resultString = ''
    s.set('Waiting for transcription to complete...')
    response = operation.result(timeout=9000)

    for result in response.results:
        resultString += result.alternatives[0].transcript

    return resultString
class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

args = Namespace(
    batch_size=4,
    max_seq_len=512,
    data_path='../data/private_dataset.file',
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    stories_folder='../data/my_own_stories',
    subset=None,
    test_pct=0.1
)

tokenizer = BartTokenizer.from_pretrained(
    'facebook/bart-large-cnn', add_prefix_space=True)

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]

def getSummary(transcript):
    s.set('Waiting for summarization to complete...')

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    inputs = tokenizer([transcript], max_length=1024, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, min_length=150, max_length=500, early_stopping=True)

    return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]

def upload():
    #Ask to upload an mp4 file
    video = filedialog.askopenfilename(initialdir = '/', title = 'Select a File', filetypes = (('MP4 files', '*.mp4*'), ("all files", "*.*")))

    #Get a temporary path to save intermediary audio file and convert mp4 to wav audio file
    tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(video) + str(random.randint(0, 10000)) + '.wav')
    s.set('Getting audio...')
    input = ffmpeg.input(os.path.realpath(video))
    audio = input.audio
    out = ffmpeg.output(audio, tmp_path)
    ffmpeg.run(out)

    #Generate the transcript   
    audioFileName = os.path.basename(tmp_path)
    upload_blob('cpen291__videos', tmp_path, audioFileName)
    transcript = getTranscript('gs://cpen291__videos/' + audioFileName)

    #Using the trained model, we summarize transcript
    summary = getSummary(transcript)

    #Ask where to save summary as text file
    f = asksaveasfile(initialfile = 'Summary.txt', defaultextension = '.txt', filetypes = [('Text Documents','*.txt'), ("All Files","*.*")])
    for line in summary:
        f.write(line)
        f.write('\n')

    s.set('Done! Upload again?')


def threading():
    # Call upload function
    t1 = Thread(target=upload)
    t1.start()

#Main application
root = Tk()
root.title('Summarizer')
root.geometry('500x150') 

s = StringVar()

s.set('Click the button to upload a lecture')

label_upload = Label(root, text='Upload your lecture to summarize').pack(pady=6)
button_explore = Button(root, text='Upload and Summarize', command=threading).pack(pady=6)
lab = Label(root, textvariable=s).pack(side=BOTTOM, pady=6)

root.mainloop()
