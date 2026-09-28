# Speech-to-Text and Text-to-Speech with Whisper and TTS

A practical exploration of speech-to-text and text-to-speech technologies using Whisper and TTS approaches.

This project demonstrates how audio can be converted into text using OpenAI Whisper and how text can be converted back into speech. It also analyzes the complete text-to-audio-to-text pipeline and discusses factors such as audio quality, pronunciation, multilingual support, latency, computational requirements, and accessibility.

## Overview

Modern speech systems enable applications to understand spoken language and generate natural-sounding audio.

This project explores two complementary tasks:

1. Speech-to-Text using Whisper
2. Text-to-Speech using TTS

The notebook also examines how these technologies can be combined into an end-to-end audio pipeline.

The overall workflow is:

```text
Audio Input
     |
     v
Audio Preprocessing
     |
     v
Whisper Speech-to-Text
     |
     v
Text Processing
     |
     v
Text-to-Speech
     |
     v
Audio Output
```

## Objectives

The main objectives of this project are to:

* Transcribe audio using Whisper
* Understand audio preprocessing requirements
* Generate speech from text
* Analyze speech quality and characteristics
* Compare speech-to-text and text-to-speech workflows
* Understand the effect of audio quality on transcription
* Explore multilingual and accent-related challenges
* Analyze the complete text and audio pipeline
* Identify practical applications of speech technologies

## Technologies Used

| Technology                | Purpose                              |
| ------------------------- | ------------------------------------ |
| Python                    | Core programming language            |
| PyTorch                   | Model inference and audio processing |
| Hugging Face Transformers | Whisper model implementation         |
| Whisper                   | Speech-to-text transcription         |
| Torchaudio                | Audio loading and preprocessing      |
| Librosa                   | Audio processing                     |
| SoundFile                 | Audio file handling                  |
| NumPy                     | Numerical operations                 |
| Matplotlib                | Visualization                        |
| gTTS                      | Text-to-speech generation            |
| Jupyter Notebook          | Development and experimentation      |

## Task 1: Speech-to-Text with Whisper

The first task uses Whisper to convert spoken audio into text.

The notebook loads the Whisper processor and model:

```python
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small"
)

model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-small"
)
```

The audio file used in the notebook is:

```text
audio_sample.wav
```

## Audio Preprocessing

Before sending the audio to Whisper, several preprocessing steps are performed.

### Load Audio

The audio is loaded using Torchaudio:

```python
waveform, sample_rate = torchaudio.load(audio_path)
```

### Convert Stereo to Mono

If the audio contains multiple channels, the channels are averaged:

```python
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)
```

This converts multi-channel audio into a single-channel waveform.

### Resample Audio

Whisper expects audio sampled at 16 kHz.

If the input audio uses another sampling rate, it is resampled:

```python
if sample_rate != 16000:
    resampler = torchaudio.transforms.Resample(
        sample_rate,
        16000
    )
    waveform = resampler(waveform)
```

### Process Audio

The processed waveform is passed through the Whisper processor:

```python
inputs = processor(
    waveform.numpy(),
    sampling_rate=sample_rate,
    return_tensors="pt"
)
```

## Transcription

Whisper generates token IDs from the processed audio:

```python
with torch.no_grad():
    predicted_ids = model.generate(
        inputs.input_features
    )
```

The generated tokens are then converted into text:

```python
transcription = processor.batch_decode(
    predicted_ids,
    skip_special_tokens=True
)[0]
```

The final transcription is displayed as the model output.

## Whisper Analysis

The project considers several factors that can affect transcription quality:

* Audio quality
* Background noise
* Accents
* Languages and dialects
* Speech clarity
* Sampling rate
* Domain-specific terminology

The notebook also discusses Whisper's ability to work across multiple languages without task-specific fine-tuning.

## Whisper Model Analysis

The exemplar section uses:

```text
openai/whisper-base
```

for additional experimentation.

It creates synthetic audio signals and evaluates the transcription process.

The analysis includes:

* Expected text
* Transcribed text
* Audio duration
* Word overlap accuracy
* Waveform visualization
* Frequency spectrum
* Mel spectrogram visualization
* Model parameter count

## Word Overlap Accuracy

A basic word-level comparison is performed between the expected and transcribed text.

The notebook calculates common words between the two texts:

```python
common_words = set(expected_words) & set(transcribed_words)
```

and uses the overlap to estimate a basic accuracy measure.

This provides a simple demonstration of transcription evaluation rather than a comprehensive speech recognition benchmark.

## Audio Visualization

The notebook visualizes different characteristics of the audio signal.

### Waveform

The waveform represents amplitude changes over time.

### Frequency Spectrum

The frequency spectrum shows the frequency components present in the audio.

### Mel Spectrogram

The Mel spectrogram represents the audio in a form suitable for Whisper's input processing.

These visualizations help understand how raw audio is transformed before being processed by a speech recognition model.

# Task 2: Text-to-Speech

The second task explores converting text into speech.

The notebook discusses TTS models and provides an implementation using Google Text-to-Speech as an alternative approach.

The implementation attempts to use ESPnet2 first and falls back to gTTS when ESPnet2 is unavailable.

## gTTS

The notebook uses:

```python
from gtts import gTTS
```

The helper function generates speech from text:

```python
def generate_speech_gtts(
    text,
    lang="en",
    slow=False
):
```

The generated speech is saved as an MP3 file.

## Synthetic Speech Generation

The notebook also includes a simple synthetic speech generator for demonstration purposes.

The function:

```python
generate_synthetic_speech()
```

creates a speech-like waveform using multiple frequency components and an amplitude envelope.

The generated signal is then normalized and analyzed.

This is a simplified signal-generation technique for demonstrating audio processing and should not be considered a realistic neural TTS system.

## Test Texts

The notebook evaluates multiple text inputs, including:

```text
Welcome to the generative AI world, where possibilities are endless!

This is a test of text-to-speech synthesis.

The quick brown fox jumps over the lazy dog.

Machine learning is transforming the way we interact with technology.
```

For each input, the notebook analyzes:

* Generated audio duration
* RMS energy
* Sample rate
* Generated waveform

## TTS Analysis

The project considers several aspects of generated speech:

* Naturalness
* Clarity
* Pronunciation
* Speech duration
* Text formatting
* Punctuation
* Sampling rate
* Model-specific parameters

The notebook emphasizes that text formatting and punctuation can influence the flow and quality of generated speech.

# Task 3: Audio Pipeline Analysis

The final task analyzes how speech-to-text and text-to-speech systems can be combined.

The conceptual pipeline is:

```text
Text
  |
  v
Text-to-Speech
  |
  v
Audio
  |
  v
Whisper
  |
  v
Transcribed Text
```

This creates a round-trip text and audio workflow.

## Round-Trip Analysis

The notebook includes a function for analyzing text preservation:

```python
analyze_round_trip(
    original_text,
    tts_audio,
    whisper_transcription
)
```

It compares:

* Original text
* Transcribed text
* Preserved words
* Lost words
* Added words

A word preservation rate is calculated to understand how much of the original text remains after the audio-to-text conversion.

## Pipeline Considerations

The notebook identifies several important considerations when combining speech technologies:

* Quality preservation
* Latency
* Error accumulation
* Computational cost
* User experience
* Audio quality
* Model selection

Errors introduced during one stage can affect later stages of the pipeline.

# Whisper Strengths

The project identifies several strengths of Whisper:

* Multilingual support
* Robustness to background noise
* No task-specific fine-tuning required
* Good performance across diverse accents
* Broad speech recognition capabilities

## Whisper Limitations

Potential limitations include:

* Computational requirements
* Reduced accuracy with very low-quality audio
* Processing time for real-time applications
* Variation in accuracy for domain-specific terminology

# TTS Strengths

The project identifies several general strengths of TTS systems:

* Natural-sounding speech generation
* Multiple voice options
* Support for different languages
* Customizable speech parameters

## TTS Limitations

Potential limitations include:

* Limited emotional expression depending on the model
* Pronunciation difficulties with rare words
* Quality differences between TTS engines
* Computational requirements for high-quality synthesis

# Applications

Speech-to-text and text-to-speech technologies can be used in many applications.

## Meeting Transcription

Whisper can be used to convert spoken meetings into searchable text.

## Podcast and Video Subtitles

Speech recognition can generate subtitles and transcripts for audio and video content.

## Voice Assistants

Speech recognition and TTS can be combined to create conversational voice interfaces.

## Accessibility

Speech technologies can improve accessibility by:

* Converting speech into text
* Reading text aloud
* Supporting users with visual or hearing limitations

## Language Learning

TTS can provide spoken examples while speech recognition can evaluate spoken input.

## Audiobook Generation

TTS systems can convert written content into spoken audio.

# End-to-End Architecture

A complete voice assistant could use the following architecture:

```text
User Speech
    |
    v
Audio Input
    |
    v
Whisper
    |
    v
Text
    |
    v
Language Model / Application Logic
    |
    v
Generated Text
    |
    v
TTS
    |
    v
Generated Audio
    |
    v
User
```

This architecture forms the foundation of many modern voice-based AI applications.

# Installation

Install the dependencies used in the notebook:

```bash
pip install numpy==1.26.4 transformers==4.44.2 accelerate==0.34.2 torchaudio librosa soundfile
```

For the gTTS implementation:

```bash
pip install gtts pygame
```

PyTorch is also required for model inference and audio processing.

## How to Run

### 1. Clone the Repository

```bash
git clone <your-repository-link>
```

### 2. Navigate to the Project

```bash
cd <repository-name>
```

### 3. Install Dependencies

```bash
pip install numpy==1.26.4 transformers==4.44.2 accelerate==0.34.2 torchaudio librosa soundfile gtts pygame
```

### 4. Add an Audio File

The main Whisper example expects:

```text
audio_sample.wav
```

Place the audio file in the same directory as the notebook or update the `audio_path` variable.

### 5. Open the Notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
generate-audio.ipynb
```

### 6. Run the Notebook

Run the cells sequentially.

The notebook will:

1. Install dependencies
2. Load Whisper
3. Load and preprocess audio
4. Resample audio to 16 kHz when required
5. Generate a transcription
6. Explore TTS generation
7. Analyze generated audio
8. Visualize audio characteristics
9. Analyze the complete speech pipeline

# Project Structure

```text
Speech-Audio-Pipeline/
|
├── generate-audio.ipynb
├── audio_sample.wav
└── README.md
```

The audio file is required for reproducing the main Whisper transcription example.

# Learning Outcomes

This project provides practical experience with:

* Speech-to-text
* Text-to-speech
* Whisper
* Audio preprocessing
* Audio resampling
* Mono and stereo audio
* Waveform analysis
* Frequency analysis
* Mel spectrograms
* Speech transcription
* Synthetic audio generation
* TTS processing
* Round-trip audio pipelines
* Multilingual speech processing
* Voice-based AI applications

# Limitations

This notebook is primarily an educational exploration.

The TTS portion does not implement a single dedicated neural TTS model throughout the notebook. It uses gTTS as an alternative and also demonstrates simplified synthetic speech generation.

The round-trip analysis contains simulated transcription for demonstration in the exemplar solution rather than measuring a complete real-world TTS-to-Whisper pipeline.

The basic word-overlap metric is also a simplified evaluation method and should not be treated as a standard speech recognition metric.

# Future Improvements

The project can be extended by:

* Using a dedicated neural TTS model
* Implementing real-time speech recognition
* Adding streaming TTS
* Supporting voice activity detection
* Adding automatic language detection
* Evaluating transcription using Word Error Rate
* Evaluating TTS using human or automated quality metrics
* Adding speaker identification
* Supporting voice cloning with appropriate consent
* Building a real-time voice assistant
* Implementing speech translation
* Deploying the complete pipeline as a web application
* Optimizing models for mobile and edge devices
* Exploring model quantization and distillation

# Ethical and Practical Considerations

Speech applications require careful consideration of:

* Voice cloning and consent
* Privacy of recorded speech
* Sensitive audio data
* Bias in speech recognition
* Language and accent coverage
* Secure handling of user recordings

These considerations become especially important when deploying speech systems in real-world applications.

# Conclusion

This project provides a practical introduction to the complete speech-to-text and text-to-speech workflow.

Whisper is used to transform audio into text, while TTS approaches demonstrate how text can be converted back into audio. The project also explores audio preprocessing, visualization, transcription evaluation, speech characteristics, and round-trip text and audio processing.

Together, these concepts provide a foundation for building applications such as voice assistants, transcription systems, accessibility tools, audiobook generators, language-learning platforms, and multimodal AI applications.

## Author

Aayush Kumar
