# class with ASR and text-2-speech capabilities
# as automatic speech recognition model OpenAI's whisper model is used
# as text-2-speech model ElevenLabs is used
# Should be final, except of
# Documentation and type definitions are NOT YET final (let chatgpt do it).

from robot_environment.common import log_start_end_cls

import numpy as np

import torch

from transformers import pipeline

import os
import time

import sounddevice as sd
from scipy.io.wavfile import write
import tempfile

from whisper_mic import WhisperMic


class Speech2Text:
    """
    class with ASR and text-2-speech capabilities

    """

    # *** CONSTRUCTORS ***
    def __init__(self, el_api_key: str, device: str, torch_dtype: type, use_whisper_mic: bool = True,
                 verbose: bool = False):
        """
        Loads the ASR model and creates an ElevenLabs client object

        Args:
            el_api_key: API Key of ElevenLabs
            device: "cpu" or "cuda"
            torch_dtype: torch.float16 or torch.float32
            use_whisper_mic: if True, then use the package whisper_mic to do ASR, else use the own implementation of
              ASR with whisper model from HuggingFace
            verbose:

        Returns:
            object:
        """
        self._verbose = verbose

        if use_whisper_mic:
            try:
                self._whisper_mic = WhisperMic(model="medium", english=False, pause=1, device=device)
            except (AttributeError, AssertionError, torch.OutOfMemoryError) as e:
                print(e)
                self._whisper_mic = None
            self._asr_model = None
        else:
            # TODO: whisper-large-v3 is better but if gpu is too small...
            # Load the ASR model
            self._asr_model = pipeline("automatic-speech-recognition",
                                       # model="openai/whisper-large-v3",
                                       model="openai/whisper-medium",
                                       torch_dtype=torch_dtype, device=device)
            self._whisper_mic = None

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    # *** PUBLIC methods ***

    def record_and_transcribe(self) -> str:
        """
        Record from microphone and transcribe using Whisper ASR model until silence is detected.

        Returns:
            transcribed text
        """
        if self._whisper_mic is None:
            return self._record_and_transcribe()
        else:
            return self._record_and_transcribe_whisper_mic()

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    def _record_and_transcribe(self) -> str:
        """
        Records speech until silence, transcribes the speech and returns transcribed message. If the speech is
        another language than English, then the speech is translated to English.

        Returns:
            str : transcribed text
        """
        audio_data, sample_rate = self._record_audio_until_silence()

        # Temporäre Datei im temporären Ordner erstellen
        temp_dir = tempfile.gettempdir()  # Verzeichnis für temporäre Dateien
        temp_path = os.path.join(temp_dir, "temp_audio.wav")  # Dateipfad für temporäre Datei

        # Audio-Daten in Datei speichern und sicherstellen, dass die Datei geschlossen wird
        write(temp_path, sample_rate, (audio_data * 32767).astype(np.int16))

        # ASR-Modell anwenden: Transkription mit Whisper durchführen
        result = self._asr_model(temp_path, generate_kwargs={"task": "translate"})

        # Temporäre Datei löschen, nachdem sie genutzt wurde
        os.remove(temp_path)

        # Return transcription text
        return result["text"]

    def _record_and_transcribe_whisper_mic(self) -> str:
        """
        Records speech until silence, transcribes the speech and returns transcribed message. Uses whisper_mic package.
        If the speech is another language than English, then the speech is translated to English.

        Returns:
            str : transcribed text
        """
        result = self._whisper_mic.listen()

        if self.verbose():
            print(result)

        return result

    @classmethod
    def _record_audio_until_silence(cls, silence_threshold: float = 0.0005, silence_duration: float = 3.0):
        """
        Record audio until 3 second of silence is detected

        Args:
            silence_threshold:
            silence_duration:

        Returns:
            object:
        """
        sample_rate = 16000  # Whisper expects 16kHz audio
        audio_data = []
        silence_start = None

        print("Recording... Speak now.")

        # Start recording in chunks to check for silence
        with sd.InputStream(samplerate=sample_rate, channels=1) as stream:
            while True:
                # Read audio in short chunks
                chunk, overflowed = stream.read(int(sample_rate * 0.1))  # 100ms chunks
                audio_data.append(chunk)

                # Compute the volume level of the chunk
                volume = np.linalg.norm(chunk) / len(chunk)

                print(volume)

                # Check for silence (volume below threshold)
                if volume < silence_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= silence_duration:
                        print("Silence detected. Stopping recording.")
                        break  # Stop recording after silence_duration of silence
                else:
                    silence_start = None  # Reset silence timer if speaking resumes

        # Concatenate chunks and return
        audio_data = np.concatenate(audio_data, axis=0)
        return audio_data, sample_rate

    # *** PUBLIC properties ***

    def verbose(self) -> bool:
        """

        Returns: True, if verbose is on, else False

        """
        return self._verbose

    # *** PRIVATE variables ***

    _verbose = False
