package com.dimitrijeski.alex_llm.model;

import com.dimitrijeski.alex_llm.network.CustomMultiLayerNetwork;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * AlexLanguageModel is a simple neural network-based language model.
 * It predicts the next word in a sequence given a window of previous words.
 * <p/>
 * This class demonstrates:
 * - Vocabulary encoding (word-to-index and index-to-word)
 * - Neural network construction using DL4J
 * - Training on text data using a sliding window
 * - Generating text from a seed sequence
 * <p/>
 * <b>About Temperature Sampling:</b>
 * Temperature sampling is a technique used during text generation to control the randomness of predictions.
 * The logits (raw scores) from the model are divided by a "temperature" value before applying softmax.
 *   - If temperature = 1.0, the distribution is unchanged.
 *   - If temperature < 1.0, the distribution becomes sharper (more confident, more repetitive).
 *   - If temperature > 1.0, the distribution becomes flatter (more random, more diverse).
 * This allows you to tune the trade-off between repetition and creativity in generated text.
 */
@Slf4j
public class AlexLanguageModel implements LanguageModel {
    private static final double TEMPERATURE = 0.9; // Default temperature for sampling

    // Size of the context window (number of previous words used for prediction)
    private final int windowSize;
    // Number of unique words in the vocabulary
    private final int vocabSize;
    // Maps each word to a unique integer index
    private final ConcurrentHashMap<String, Integer> wordToIdx = new ConcurrentHashMap<>();
    // Maps each index back to its corresponding word
    private final List<String> idxToWord = new ArrayList<>();
    // The neural network model
    private CustomMultiLayerNetwork model;

    // New: Store the input size for the model (windowSize * vocabSize)
    private final int inputSize;

    /**
     * Constructs the language model with a given window size and vocabulary.
     * Initializes the word-index mappings and builds the neural network.
     * 
     * @param windowSize Number of previous words to use as input
     * @param vocab Set of unique words in the corpus
     */
    public AlexLanguageModel(int windowSize, Set<String> vocab) {
        this.windowSize = windowSize;
        this.vocabSize = vocab.size();
        this.inputSize = windowSize * vocabSize;
        int idx = 0;
        for (String word : vocab) {
            wordToIdx.put(word, idx++);
            idxToWord.add(word);
        }
        buildModel();
    }

    /**
     * Builds a simple feedforward neural network with two hidden layers.
     * - Input: One-hot encoded vectors for each word in the window
     * - Hidden layers: Dense layers with ReLU activation
     * - Output: Softmax over vocabulary (predicts next word)
     * <p/>
     * Why add multiple layers?
     *   - Multiple (hidden) layers allow the network to learn more complex patterns and representations.
     *   - Each layer can extract higher-level features from the previous layer's output.
     *   - With only one layer, the model can only learn very simple relationships.
     * <p/>
     * Do we need an output layer?
     *   - Yes, the output layer transforms the final hidden representation into the desired output format.
     *   - For language modeling, the output layer produces a probability distribution over all possible next words.
     *   - The softmax activation ensures the outputs are valid probabilities.
     * <p/>
     * What if you added 20 layers?
     *   - Adding many layers (like 20) increases the model's capacity to learn complex patterns,
     *     but also makes training much harder.
     *   - Deep networks can suffer from vanishing/exploding gradients, making them hard to train without special techniques.
     *   - More layers mean more parameters, which increases memory usage and risk of overfitting if you don't have enough data.
     *   - In practice, very deep networks require careful initialization, normalization (like batch normalization), and lots of data.
     */
    private void buildModel() {
        // Create the configuration for the neural network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123) // Set random seed for reproducibility
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // Use SGD optimizer (updates weights using small random batches, helps escape local minima)
                .updater(new Adam(0.2)) // Use Adam optimizer (adaptive moment estimation: combines momentum and adaptive learning rates for faster, more stable training) with learning rate 0.1
                .list()
                // Adjust input size to windowSize * vocabSize
                // First hidden layer: Dense (fully connected), input size = vocabSize, output size = 128
                .layer(new DenseLayer.Builder()
                        .nIn(inputSize) // Number of input neurons (size of vocabulary)
                        .nOut(128) // Number of output neurons (hidden units: neurons not directly exposed as output, used for learning internal representations)
                        .activation(Activation.RELU) // Activation function: ReLU (Rectified Linear Unit, outputs max(0, x), helps with non-linearity and avoids vanishing gradients)
                        .build())
                // Second hidden layer: Dense, input size = 128, output size = 64
                .layer(new DenseLayer.Builder()
                        .nIn(128) // Number of input neurons (from previous layer)
                        .nOut(64) // Number of output neurons (hidden units)
                        .activation(Activation.RELU) // Activation function: ReLU
                        .build())
                // Third hidden layer: Dense, input size = 64, output size = 32
                .layer(new DenseLayer.Builder()
                        .nIn(64) // Number of input neurons (size of vocabulary)
                        .nOut(32) // Number of output neurons (hidden units: neurons not directly exposed as output, used for learning internal representations)
                        .activation(Activation.RELU) // Activation function: ReLU (Rectified Linear Unit, outputs max(0, x), helps with non-linearity and avoids vanishing gradients)
                        .build())
                // Output layer: predicts the next word as a probability distribution over the vocabulary
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT) // Multi-class cross-entropy loss (measures how well the predicted probability distribution matches the true distribution for classification tasks)
                        .nIn(32) // Number of input neurons (from previous layer)
                        .nOut(vocabSize) // Number of output neurons (one per word in vocabulary; output neurons produce the final predictions)
                        .activation(Activation.SOFTMAX) // Softmax activation (converts raw scores to probabilities that sum to 1, used for multi-class classification)
                        .build())
                .build();

        // Initialize the custom neural network with the configuration
        model = new CustomMultiLayerNetwork(conf);
        model.init(); // Actually allocate and initialize the network parameters
    }

    /**
     * Trains the model on the provided text.
     * - Splits text into tokens (words)
     * - For each window, creates a one-hot input for the context and a one-hot label for the next word
     * - Stacks all examples into a DataSet and fits the model
     * <p/>
     * Labels:
     *   For each input window (context), the label is a one-hot encoded vector representing
     *   the next word in the sequence. For example, if the vocabulary is ["I", "like", "cats"]
     *   and the next word is "cats", the label would be [0, 0, 1].
     *   During training, the model learns to predict this next word given the context.
     *
     * @param text Training text
     * @return The trained LanguageModel (this)
     */
    @Override
    public LanguageModel train(String text) {
        String[] tokens = text.split("\\s+");
        List<INDArray> inputs = new ArrayList<>();
        List<INDArray> labels = new ArrayList<>();

        // Use concatenated one-hot encoding for each word in the window
        for (int i = 0; i <= tokens.length - windowSize - 1; i++) {
            INDArray input = Nd4j.zeros(1, inputSize);
            for (int j = 0; j < windowSize; j++) {
                Integer idx = wordToIdx.get(tokens[i + j]);
                if (idx != null) {
                    input.putScalar((long) j * vocabSize + idx, 1.0);
                }
            }
            inputs.add(input);

            INDArray label = Nd4j.zeros(1, vocabSize);
            Integer labelIdx = wordToIdx.get(tokens[i + windowSize]);
            if (labelIdx != null) {
                label.putScalar(labelIdx, 1.0);
            }
            labels.add(label);
        }

        DataSet ds = new DataSet(Nd4j.vstack(inputs), Nd4j.vstack(labels));

        // Logging: show training info
        log.info("Training examples: {}", inputs.size());
        log.info("Input shape: {}", ds.getFeatures().shapeInfoToString());
        log.info("Label shape: {}", ds.getLabels().shapeInfoToString());
        if (!inputs.isEmpty()) {
            log.info("Sample input: {}", inputs.get(0));
            log.info("Sample label: {}", labels.get(0));
        }

        model.fit(ds);
        return this;
    }

    /**
     * Generates text by repeatedly predicting the next word.
     * Uses temperature sampling for more diverse outputs.
     * <p/>
     * <b>Temperature Sampling:</b>
     * The model's output logits are divided by a temperature value before applying softmax.
     * Lower temperature (<1) makes the model more confident and deterministic.
     * Higher temperature (>1) increases randomness and diversity in the generated text.
     *
     * @param seed   The initial sequence of words to start generation
     * @param length The number of words to generate
     * @return The generated text as a string
     */
    @Override
    public String generateText(List<String> seed, int length) {
        // Start with the seed words
        StringBuilder result = new StringBuilder(String.join(" ", seed));
        // Use a linked list to maintain the current context window
        LinkedList<String> context = new LinkedList<>(seed);

        for (int i = 0; i < length; i++) {
            // Prepare the input: concatenated one-hot vectors for the current context window
            INDArray input = Nd4j.zeros(1, inputSize);
            for (int j = 0; j < windowSize; j++) {
                if (j < context.size()) {
                    Integer idx = wordToIdx.get(context.get(j));
                    if (idx != null) {
                        input.putScalar((long) j * vocabSize + idx, 1.0);
                    }
                }
            }

            // Log the current context window
            log.info("Step {} context: {}", i + 1, context);

            // Get the model's output logits for the next word prediction
            INDArray output = model.output(input);
            INDArray logits = output.dup();

            // Log the raw logits
            log.info("Logits: {}", Arrays.toString(logits.toDoubleVector()));

            // Apply temperature scaling to logits (controls randomness/diversity)
            if (TEMPERATURE != 1.0) {
                logits = logits.div(TEMPERATURE);
            }

            // Convert logits to probabilities using softmax
            INDArray probsArr = Nd4j.nn.softmax(logits, 1);
            double[] probs = probsArr.toDoubleVector();

            // Log the probabilities
            log.info("Probabilities: {}", Arrays.toString(probs));

            // Sample the next word index from the probability distribution
            int nextIdx = sampleFromDistribution(probs);
            String nextWord = idxToWord.get(nextIdx);

            // Log the chosen word
            log.info("Chosen word: {}", nextWord);

            // Append the predicted word to the result and update the context window
            result.append(" ").append(nextWord);
            context.add(nextWord);
            if (context.size() > windowSize) {
                context.removeFirst();
            }
        }

        return result.toString();
    }

    // Helper: Sample index from probability distribution
    private int sampleFromDistribution(double[] probs) {
        double p = Math.random();
        double cumulative = 0.0;
        for (int i = 0; i < probs.length; i++) {
            cumulative += probs[i];
            if (p < cumulative) return i;
        }
        return probs.length - 1; // fallback
    }
}