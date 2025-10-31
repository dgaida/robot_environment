# Speech2Text Module

This package provides speech-to-text (ASR) capabilities using OpenAI's Whisper model. It is designed as a submodule of the larger `robot_environment` package.

## Overview

The `speech2text` module provides a Python interface to record audio from a microphone and convert it into text using automatic speech recognition. It supports two operating modes:

1. **WhisperMic Mode** – Uses the `whisper_mic` package for real-time recording and transcription.
2. **Local Whisper Mode** – Uses Hugging Face’s Whisper model pipeline for offline transcription.

Both methods detect silence automatically and stop recording when no speech is detected.

## File Structure

```
robot_environment/
└── speech2text/
    ├── __init__.py
    └── speech2text.py
```

## Installation

Install dependencies:

```bash
pip install torch transformers sounddevice scipy whisper_mic
```

> Note: GPU acceleration is supported but optional.

## Usage Example

```python
from robot_environment.speech2text.speech2text import Speech2Text
import torch

# Initialize the speech-to-text model
stt = Speech2Text(
    device="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.float16,
    use_whisper_mic=True,
    verbose=True,
)

# Record and transcribe audio
text = stt.record_and_transcribe()
print("Transcribed text:", text)
```

## Class: `Speech2Text`

### Methods

#### `__init__(device, torch_dtype, use_whisper_mic=True, verbose=False)`
Initializes the class, loads the ASR model, and sets the configuration.

#### `record_and_transcribe()`
Records and transcribes speech using either WhisperMic or a Hugging Face Whisper model.

#### `verbose()`
Returns the current verbosity setting.

### Private Methods

- `_record_and_transcribe()`: Handles offline recording and transcription.
- `_record_and_transcribe_whisper_mic()`: Handles real-time WhisperMic transcription.
- `_record_audio_until_silence()`: Records audio until silence is detected.

## License

This module is distributed under the MIT License. See the LICENSE file in the root of the `robot_environment` repository for more information.

## Author

Developed as part of the `robot_environment` framework for multimodal robotic interaction.

