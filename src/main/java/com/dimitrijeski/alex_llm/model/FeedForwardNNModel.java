package com.dimitrijeski.alex_llm.model;


import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;


import java.util.*;

/**
 * FeedForwardNNModel demonstrates a simple feedforward neural network language model using DL4J.
 * This model predicts the next word given a fixed window of previous words, using one-hot encoding for input.
 * <p/>
 * The architecture is:
 * Input (concatenated one-hot vectors for windowSize words) ->
 * DenseLayer (hidden layer) ->
 * OutputLayer (softmax over vocabulary)
 * <p/>
 * This model is a basic example of a neural language model and is useful for understanding how neural networks
 * can be used for next-word prediction with fixed context size.
 *
 * <h2>Layer Types and Activations</h2>
 * <ul>
 *   <li><b>DenseLayer</b>: A fully connected layer. Every input neuron is connected to every output neuron.
 *       <ul>
 *         <li>Activation: <b>RELU</b> (Rectified Linear Unit). Outputs max(0, x). Common for hidden layers as it helps with gradient flow and non-linearity.</li>
 *       </ul>
 *   </li>
 *   <li><b>OutputLayer</b>: A fully connected layer for producing the final output.
 *       <ul>
 *         <li>Activation: <b>SOFTMAX</b>. Converts the output into a probability distribution over the vocabulary, so the model can "choose" the most likely next word.</li>
 *         <li>Loss: <b>MCXENT</b> (Multi-Class Cross Entropy). Standard for classification tasks with more than two classes (here, the vocabulary size).</li>
 *       </ul>
 *   </li>
 * </ul>
 */
public class FeedForwardNNModel implements LanguageModel {
    private final int windowSize;
    private final int vocabSize;
    private final Map<String, Integer> wordToIdx = new HashMap<>();
    private final List<String> idxToWord = new ArrayList<>();
    private MultiLayerNetwork model;

    /**
     * Constructs a FeedForwardNNModel with the specified window size and vocabulary.
     * @param windowSize number of previous words to use as input
     * @param vocab corpus vocabulary (unique words)
     */
    public FeedForwardNNModel(int windowSize, Set<String> vocab) {
        // Build word-index mappings for one-hot encoding and decoding
        this.windowSize = windowSize;
        this.vocabSize = vocab.size();
        int idx = 0;
        for (String word : vocab) {
            wordToIdx.put(word, idx++);
            idxToWord.add(word);
        }
    }

    /**
     * Trains the feedforward neural network model on the given text.
     * <p/>
     * The training data is constructed as follows:
     * - For each window of 'windowSize' words in the corpus, the input is a concatenated one-hot vector of those words.
     * - The label is the next word after the window, one-hot encoded.
     * <p/>
     * The model learns to predict the next word based on the fixed-size context window.
     *
     * @param text the training corpus
     */
    @Override
    public FeedForwardNNModel train(String text) {
        // Split the input text into tokens (words)
        String[] tokens = text.split("\\s+");
        List<INDArray> inputs = new ArrayList<>();
        List<INDArray> labels = new ArrayList<>();

        // For each window in the corpus, create an input vector and a label vector
        for (int i = 0; i <= tokens.length - windowSize - 1; i++) {
            // Create a zero vector for the input of size windowSize * vocabSize
            INDArray input = Nd4j.zeros(windowSize * vocabSize);
            // For each word in the window, set the corresponding position in the input vector to 1 (one-hot encoding)
            for (int j = 0; j < windowSize; j++) {
                int idx = wordToIdx.get(tokens[i + j]);
                input.putScalar((long) j * vocabSize + idx, 1.0);
            }
            inputs.add(input);

            // Create a zero vector for the label (next word after the window)
            INDArray label = Nd4j.zeros(vocabSize);
            int labelIdx = wordToIdx.get(tokens[i + windowSize]);
            // Set the position for the next word to 1 (one-hot encoding)
            label.putScalar(labelIdx, 1.0);
            labels.add(label);
        }

        // Stack all input and label vectors into matrices for training
        DataSet ds = new DataSet(Nd4j.vstack(inputs), Nd4j.vstack(labels));
        // Use a DataSetIterator for batching during training
        ListDataSetIterator<DataSet> iter = new ListDataSetIterator<>(ds.asList(), 16);

        // Build the neural network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.01))
                .list()
                // Hidden dense layer to learn patterns in the input
                .layer(new DenseLayer.Builder()
                        .nIn(windowSize * vocabSize)
                        .nOut(64)
                        .activation(Activation.RELU)
                        .build())
                // Output layer with softmax to predict the next word
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(64)
                        .nOut(vocabSize)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        // Initialize and train the model
        model = new MultiLayerNetwork(conf);
        model.init();
        // Fit the model for a small number of epochs (for demonstration/learning)
        model.fit(iter, 10);
        return this;
    }

    /**
     * Generates text from a seed using the trained model.
     * At each step, the model predicts the next word based on the previous windowSize words.
     *
     * @param seed initial words
     * @param length number of words to generate
     * @return generated text
     */
    @Override
    public String generateText(List<String> seed, int length) {
        List<String> result = new ArrayList<>(seed);
        for (int i = 0; i < length; i++) {
            // Prepare the input vector for the current context (last windowSize words)
            INDArray input = Nd4j.zeros(1, windowSize * vocabSize);
            for (int j = 0; j < windowSize; j++) {
                String word = result.get(result.size() - windowSize + j);
                Integer idx = wordToIdx.get(word);
                if (idx != null) {
                    input.putScalar(0, (long) j * vocabSize + idx, 1.0);
                }
            }
            // Use the model to predict the next word (output is a probability distribution over the vocabulary)
            INDArray output = model.output(input);
            // Pick the word with the highest probability
            int nextIdx = Nd4j.argMax(output, 1).getInt(0);
            result.add(idxToWord.get(nextIdx));
        }
        // Return the generated sequence as a string
        return String.join(" ", result);
    }
}