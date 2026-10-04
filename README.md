# Deep-Learning-for-Music-Genre-Classification

A PyTorch classifier that listens to music by *looking* at it: raw audio is converted into log-scaled Mel-spectrograms and treated as an image recognition problem, then classified into 10 genres with a fine-tuned ResNet-18.

**Validation accuracy: 94.06% across 10 genres**

## Highlights
- **Audio as vision:** log-scaled Mel-spectrograms turn sound into images, so proven computer vision models can be applied directly.
- **Tenfold data augmentation:** the GTZAN dataset was expanded using Librosa temporal slicing and per-band normalisation.
- **Transfer learning:** a pre-trained ResNet-18 fine-tuned for genre recognition.
- **Stable real-world predictions:** a 5-slice averaging algorithm smooths results across segments of a track, so one odd moment doesn't flip the genre.
- **Live demo:** deployed as a Gradio web app where you can upload audio and get a prediction.

## Tech Stack
PyTorch · Librosa · ResNet-18 · Gradio · GTZAN dataset

## Pipeline
Raw audio → temporal slices → log-scaled Mel-spectrograms (per-band normalised) → ResNet-18 → averaged prediction over 5 slices → genre
