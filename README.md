# Simple Java N-Gram and Neural Language Models

This project demonstrates basic language models in Java, including unigram, bigram, trigram, custom n-gram models, and neural network-based models using DL4J.

## How it works

- **UnigramModel**: Predicts the next word based only on word frequency.
- **BigramModel**: Predicts the next word based on the previous word.
- **NGramModel**: Predicts the next word based on the previous (n-1) words. You can specify `n` for custom models.
- **FeedForwardNNModel**: Simple feedforward neural network language model using DL4J.
- **RNNModel**: Simple LSTM (RNN) language model using DL4J.
- **TransformerModel**: Minimal Transformer-based language model using DL4J's SelfAttentionLayer.

All models implement the `LanguageModel` interface, so you can easily add your own.

## Usage

1. Add DL4J dependencies to your `pom.xml` (see [DL4J documentation](https://deeplearning4j.konduit.ai/)).
2. Compile the Java files.
3. Run `Main.java` to see generated text from each model.

## Custom Models

To create your own model, implement the `LanguageModel` interface or instantiate `NGramModel` with your desired `n`.

## Code Documentation

All classes and methods are documented in the code for clarity.

## Requirements

- Java 8+
- [Deeplearning4j (DL4J)](https://deeplearning4j.konduit.ai/) for neural models

