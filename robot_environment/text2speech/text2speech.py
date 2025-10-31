# class with text-2-speech capabilities

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
    """Text-to-Speech (TTS) and Automatic Speech Recognition (ASR) class.

    This class provides text-to-speech functionality using the Kokoro model (as a replacement
    for the deprecated ElevenLabs integration). It supports asynchronous speech synthesis
    and safe audio playback with dynamic resampling.
    """

    def __init__(self, el_api_key: str, verbose: bool = False) -> None:
        """Initialize the Text2Speech instance.

        Attempts to initialize the ElevenLabs client, but defaults to using the Kokoro
        model for TTS instead. This ensures the system works even if ElevenLabs is
        unavailable or deprecated.

        Args:
            el_api_key (str): API key for ElevenLabs (no longer used).
            verbose (bool, optional): If True, prints detailed debug information. Defaults to False.
        """
        self._verbose = verbose

        try:
            self._client = ElevenLabs(api_key=el_api_key)
            raise Exception("Sorry, we do not use ElevenLabs anymore.")
        except (NameError, Exception) as e:
            if verbose:
                print(e)
            # 🇺🇸 'a' => American English, 🇬🇧 'b' => British English
            self._client = KPipeline(lang_code='a')  # Make sure lang_code matches desired voice
            if verbose:
                print('Using Kokoro instead!')
        # TODO: has to be done in ubuntu only so that meeting owl is used for output
        # sd.default.device[1] = 4  # Output only

    @log_start_end_cls()
    def call_text2speech_async(self, text: str) -> threading.Thread:
        """Call text-to-speech asynchronously using the Kokoro model.

        Args:
            text (str): The text to be spoken.

        Returns:
            threading.Thread: The thread handling the asynchronous TTS operation.
        """
        thread = threading.Thread(target=self._text2speech_kokoro, args=(text,))
        thread.start()
        return thread

    @log_start_end_cls()
    def _text2speech(self, mytext: str) -> None:
        """Legacy ElevenLabs TTS method (no longer used).

        Args:
            mytext (str): The text to be synthesized and played.
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
        """Perform TTS using the Kokoro model.

        Args:
            mytext (str): Text to convert to speech.
        """
        if self._client is not None:
            try:
                generator = self._client(
                    mytext,
                    voice='af_heart',
                    speed=1,
                    split_pattern=r'\n+'
                )
                for _, _, audio in generator:
                    Text2Speech._play_audio_safely(audio, original_sample_rate=24000)
                    sd.wait()
            except Exception as e:
                print(f"Error with Kokoro: {e, e.with_traceback(None)}")

    @staticmethod
    def _play_audio_safely(
        audio_tensor: torch.Tensor,
        original_sample_rate: int = 24000,
        device: int | None = None,
        volume: float = 0.8
    ) -> None:
        """Play audio safely by checking supported sample rate and adjusting volume.

        Args:
            audio_tensor (torch.Tensor): The 1D audio waveform tensor.
            original_sample_rate (int, optional): Original sample rate of the audio. Defaults to 24000.
            device (int | None, optional): Output device index. Defaults to system default.
            volume (float, optional): Playback volume multiplier (0.0–1.0). Defaults to 0.8.
        """
        try:
            if device is None:
                device = sd.default.device[1]  # Default output device

            device_info = sd.query_devices(device, 'output')
            supported_rate = int(device_info['default_samplerate'])

            if original_sample_rate != supported_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=original_sample_rate,
                    new_freq=supported_rate
                )
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

    def verbose(self) -> bool:
        """Check whether verbose mode is enabled.

        Returns:
            bool: True if verbose mode is active, otherwise False.
        """
        return self._verbose
