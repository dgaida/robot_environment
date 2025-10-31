# class with ASR capabilities
# as automatic speech recognition model OpenAI's whisper model is used

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
    """Speech-to-text class with automatic speech recognition (ASR) capabilities.

    This class supports two ASR modes:
    1. Using the `whisper_mic` package for live transcription via microphone.
    2. Using the Hugging Face Whisper model for local recording and transcription.

    Attributes:
        _verbose (bool): Whether verbose logging is enabled.
        _asr_model: The Whisper ASR model loaded via Hugging Face (optional).
        _whisper_mic: The WhisperMic instance used for microphone-based ASR (optional).
    """

    # *** CONSTRUCTORS ***
    def __init__(
        self,
        device: str,
        torch_dtype: type,
        use_whisper_mic: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initializes the Speech2Text class and loads the ASR model.

        Args:
            device (str): The device to run the ASR model on ("cpu" or "cuda").
            torch_dtype (type): The torch data type (e.g., torch.float16 or torch.float32).
            use_whisper_mic (bool, optional): Whether to use the `whisper_mic` package. Defaults to True.
            verbose (bool, optional): Whether to enable verbose output. Defaults to False.
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
            self._asr_model = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-medium",
                torch_dtype=torch_dtype,
                device=device,
            )
            self._whisper_mic = None

    # *** PUBLIC methods ***

    def record_and_transcribe(self) -> str:
        """Records speech from the microphone and transcribes it using the selected ASR backend.

        Returns:
            str: The transcribed text.
        """
        if self._whisper_mic is None:
            return self._record_and_transcribe()
        else:
            return self._record_and_transcribe_whisper_mic()

    # *** PRIVATE methods ***

    def _record_and_transcribe(self) -> str:
        """Records speech until silence is detected, then performs transcription.

        If the speech is in a non-English language, Whisper automatically translates it to English.

        Returns:
            str: The transcribed and (if needed) translated text.
        """
        audio_data, sample_rate = self._record_audio_until_silence()

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "temp_audio.wav")

        write(temp_path, sample_rate, (audio_data * 32767).astype(np.int16))

        result = self._asr_model(temp_path, generate_kwargs={"task": "translate"})

        os.remove(temp_path)

        return result["text"]

    def _record_and_transcribe_whisper_mic(self) -> str:
        """Uses `whisper_mic` to record and transcribe speech directly from the microphone.

        Returns:
            str: The transcribed text.
        """
        result = self._whisper_mic.listen()

        if self.verbose():
            print(result)

        return result

    @classmethod
    def _record_audio_until_silence(
        cls, silence_threshold: float = 0.0005, silence_duration: float = 3.0
    ) -> tuple[np.ndarray, int]:
        """Records audio until a specified duration of silence is detected.

        Args:
            silence_threshold (float, optional): Amplitude threshold to detect silence. Defaults to 0.0005.
            silence_duration (float, optional): Duration of continuous silence (in seconds) to stop recording. Defaults to 3.0.

        Returns:
            tuple[np.ndarray, int]: A tuple containing the recorded audio data and the sample rate.
        """
        sample_rate = 16000
        audio_data = []
        silence_start = None

        print("Recording... Speak now.")

        with sd.InputStream(samplerate=sample_rate, channels=1) as stream:
            while True:
                chunk, _ = stream.read(int(sample_rate * 0.1))
                audio_data.append(chunk)

                volume = np.linalg.norm(chunk) / len(chunk)
                print(volume)

                if volume < silence_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= silence_duration:
                        print("Silence detected. Stopping recording.")
                        break
                else:
                    silence_start = None

        audio_data = np.concatenate(audio_data, axis=0)
        return audio_data, sample_rate

    # *** PUBLIC properties ***

    def verbose(self) -> bool:
        """Returns whether verbose mode is active.

        Returns:
            bool: True if verbose output is enabled, False otherwise.
        """
        return self._verbose

    # *** PRIVATE variables ***

    _verbose: bool = False
