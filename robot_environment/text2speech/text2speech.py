# class with ASR and text-2-speech capabilities
# as automatic speech recognition model OpenAI's whisper model is used
# as text-2-speech model ElevenLabs is used
# Should be final, except of TODO with TTS alternative
# Documentation and type definitions are NOT YET final (let chatgpt do it).

from ..common.logger import log_start_end_cls

try:
    from elevenlabs import play
    from elevenlabs.client import ElevenLabs
except ImportError as e:
    print("error importing elevenlabs:", e)

from kokoro import KPipeline
import torchaudio
import torch

import sounddevice as sd

import threading


class Text2Speech:
    """
    class with ASR and text-2-speech capabilities

    """

    # *** CONSTRUCTORS ***
    def __init__(self, el_api_key: str, verbose: bool = False):
        """
        Loads the ASR model and creates an ElevenLabs client object

        Args:
            el_api_key: API Key of ElevenLabs
            verbose:

        Returns:
            object:
        """
        self._verbose = verbose

        try:
            self._client = ElevenLabs(
                api_key=el_api_key,
            )
            raise Exception("Sorry, we do not use ElevenLabs anymore.")
        except (NameError, Exception) as e:
            if verbose:
                print(e)
            # 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
            self._client = KPipeline(lang_code='a')  # <= make sure lang_code matches voice
            if verbose:
                print('using Kokoro instead!')
            # use another t2s model as an alternative, such as TTS, see other TODO
            # self._client = None
            # TODO: has to be done in ubuntu only so that meeting owl is used for output
            # sd.default.device[1] = 4  # Output only


    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    # *** PUBLIC methods ***

    @log_start_end_cls()
    def call_text2speech_async(self, text: str) -> threading.Thread:
        """
        Asynchronously calls the text2speech ElevenLabs API with the given text

        Args:
            text: a message that should be passed to text2speech API of ElevenLabs

        Returns:
            the thread object is returned. Once the text is spoken, the thread is being closed.
        """
        thread = threading.Thread(target=self._text2speech_kokoro, args=(text,))
        thread.start()
        return thread

    # *** PUBLIC STATIC/CLASS GET methods ***

    # *** PRIVATE methods ***

    @log_start_end_cls()
    # https://github.com/elevenlabs/elevenlabs-python
    def _text2speech(self, mytext: str) -> None:
        """
        Calls the text2speech ElevenLabs API with the given mytext

        Args:
            mytext: a message that is passed to text2speech API of ElevenLabs
        """
        if self._client is not None:
            try:
                audio = self._client.generate(
                    text=mytext,
                    voice="Brian",
                    model="eleven_multilingual_v2"
                )
                play(audio)
            except Exception as e:
                print(f"Error with ElevenLabs: {e, e.with_traceback(None)}")

    @log_start_end_cls()
    def _text2speech_kokoro(self, mytext: str) -> None:
        """
        Calls the text2speech kokoro model with the given mytext

        Args:
            mytext: a message that is passed to text2speech API of kokoro model
        """
        if self._client is not None:
            try:
                generator = self._client(
                    mytext, voice='af_heart',  # 'af_nicole', #
                    speed=1, split_pattern=r'\n+'
                )
                for gs, ps, audio in generator:
                    Text2Speech._play_audio_safely(audio, original_sample_rate=24000)
                    sd.wait()
            except Exception as e:
                print(f"Error with Kokoro: {e, e.with_traceback(None)}")

    @staticmethod
    def _play_audio_safely(audio_tensor: torch.Tensor, original_sample_rate: int = 24000,
                           device: int = None, volume: float = 0.8):
        """
        Safely plays audio by checking the current output device's supported sample rate
        and resampling the audio if needed.

        Args:
            audio_tensor: A 1D torch tensor containing the audio waveform
            original_sample_rate: The sample rate of the input audio
            device: Optional. Index of the sounddevice output device to use
        """
        try:
            # Use the selected device or default output device
            if device is None:
                device = sd.default.device[1]  # Get default output device

            # print(device)

            device_info = sd.query_devices(device, 'output')
            supported_rate = int(device_info['default_samplerate'])

            # print(supported_rate)

            # Resample if needed
            if original_sample_rate != supported_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate,
                                                           new_freq=supported_rate)
                audio_tensor = resampler(audio_tensor)

            # Normalize and scale volume
            peak = torch.abs(audio_tensor).max()
            if peak > 0:
                audio_tensor = audio_tensor / peak
            audio_tensor = torch.clamp(audio_tensor * volume, -0.95, 0.95)

            # Convert to numpy and play
            audio_np = audio_tensor.cpu().numpy()
            sd.play(audio_np, samplerate=supported_rate, device=device)
            sd.wait()

        except Exception as e:
            print(f"❌ Error during safe audio playback: {e}")

    # *** PUBLIC properties ***

    def verbose(self) -> bool:
        """

        Returns: True, if verbose is on, else False

        """
        return self._verbose

    # *** PRIVATE variables ***

    _verbose = False
