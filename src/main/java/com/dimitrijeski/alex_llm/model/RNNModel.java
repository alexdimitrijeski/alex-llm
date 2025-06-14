package com.dimitrijeski.alex_llm.model;

// This class implements an LSTM-based RNN language model.
// LSTM (Long Short-Term Memory) is a special type of RNN designed to handle long-term dependencies.
// In DL4J and most libraries, using an LSTM layer means your RNN is an LSTM model.
// If you want a "vanilla" (simple) RNN, you would use a SimpleRnn layer instead.
// For most NLP tasks, LSTM is the standard RNN variant.

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.Serial;
import java.util.*;

/**
 * RNNModel demonstrates a simple LSTM-based recurrent neural network language model using DL4J.
 * This model predicts the next word given a sequence of previous words (windowSize), using one-hot encoding for input.
 * <p/>
 * The architecture is:
 * Input (sequence of one-hot vectors, shape [minibatch, vocabSize, windowSize]) ->
 * LSTM layer (captures sequential dependencies) ->
 * RnnOutputLayer (softmax over vocabulary at each time step)
 *
 * <h2>Layer Types and Activations</h2>
 * <ul>
 *   <li><b>LSTM</b>: Long Short-Term Memory layer. A type of recurrent neural network (RNN) layer that can remember information for long periods and is resistant to vanishing gradients.
 *       <ul>
 *         <li>Activation: <b>TANH</b>. Outputs values between -1 and 1. Used in LSTM cells for state updates and gating mechanisms.</li>
 *       </ul>
 *   </li>
 *   <li><b>RnnOutputLayer</b>: Output layer for RNNs. Produces a prediction at each time step in the sequence.
 *       <ul>
 *         <li>Activation: <b>SOFTMAX</b>. Converts the output at each time step into a probability distribution over the vocabulary.</li>
 *         <li>Loss: <b>MCXENT</b> (Multi-Class Cross Entropy). Used for multi-class classification at each time step.</li>
 *       </ul>
 *   </li>
 * </ul>
 */
public class RNNModel implements LanguageModel {
    @Serial
    private static final long serialVersionUID = 1L;
    // Number of previous words to use as input (window size)
    private final int windowSize;
    // Size of the vocabulary (number of unique words in the corpus)
    private final int vocabSize;
    // Mapping from word to index for one-hot encoding
    private final Map<String, Integer> wordToIdx = new HashMap<>();
    // Mapping from index to word for decoding predictions
    private final List<String> idxToWord = new ArrayList<>();
    // The neural network model
    private MultiLayerNetwork model;

    /**
     * Constructs an RNNModel with the specified window size and vocabulary.
     * @param windowSize number of previous words to use as input
     * @param vocab corpus vocabulary (unique words)
     */
    public RNNModel(int windowSize, Set<String> vocab) {
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
     * Trains the LSTM-based RNN model on the given text.
     * <p/>
     * The training data is constructed as follows:
     * - For each window of 'windowSize' words in the corpus, the input is a sequence of one-hot vectors for those words.
     * - The label is a sequence of one-hot vectors for the next words at each time step (shifted by one).
     * <p/>
     * The model learns to predict the next word at each time step in the sequence.
     *
     * @param text the training corpus
     */
    @Override
    public RNNModel train(String text) {
        // Split the input text into tokens (words)
        String[] tokens = text.split("\\s+");
        List<INDArray> inputs = new ArrayList<>();
        List<INDArray> labels = new ArrayList<>();

        // For each training example, create a sequence of length windowSize
        // Input: [1, vocabSize, windowSize]
        // Label: [1, vocabSize, windowSize]
        for (int i = 0; i <= tokens.length - windowSize - 1; i++) {
            // Create a zero tensor for the input sequence
            INDArray input = Nd4j.zeros(1, vocabSize, windowSize);
            // Create a zero tensor for the label sequence
            INDArray label = Nd4j.zeros(1, vocabSize, windowSize);

            // For each position in the window, set the one-hot encoding for the input and the label
            for (int j = 0; j < windowSize; j++) {
                int idx = wordToIdx.get(tokens[i + j]);
                input.putScalar(0, idx, j, 1.0);

                // The label for time step j is the next word after tokens[i + j]
                int labelIdx = wordToIdx.get(tokens[i + j + 1]);
                label.putScalar(0, labelIdx, j, 1.0);
            }
            inputs.add(input);
            labels.add(label);
        }

        // Stack all input and label tensors into lists for batching
        List<DataSet> dataSets = new ArrayList<>();
        for (int i = 0; i < inputs.size(); i++) {
            dataSets.add(new DataSet(inputs.get(i), labels.get(i)));
        }
        // Use a DataSetIterator for batching during training
        ListDataSetIterator<DataSet> iter = new ListDataSetIterator<>(dataSets, 16);

        // Build the neural network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.01))
                .list()
                // LSTM layer to capture sequential dependencies
                .layer(new LSTM.Builder()
                        .nIn(vocabSize)
                        .nOut(64)
                        .activation(Activation.TANH)
                        .build())
                // RnnOutputLayer with softmax activation to predict the next word at each time step
                .layer(new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
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
            // Only generate if we have enough context
            if (result.size() < windowSize) break;
            // Prepare the input tensor for the current context (last windowSize words)
            INDArray input = Nd4j.zeros(1, vocabSize, windowSize);
            for (int j = 0; j < windowSize; j++) {
                String word = result.get(result.size() - windowSize + j);
                Integer idx = wordToIdx.get(word);
                if (idx != null) {
                    input.putScalar(0, idx, j, 1.0);
                }
            }
            // Use the model to predict the next word (output is [1, vocabSize, windowSize])
            INDArray output = model.output(input);
            // Take the prediction at the last time step in the window
            INDArray lastStep = output.get(point(0), all(), point(windowSize - 1));
            int nextIdx = Nd4j.argMax(lastStep, 0).getInt(0);
            result.add(idxToWord.get(nextIdx));
        }
        // Return the generated sequence as a string
        return String.join(" ", result);
    }

    // Helper methods for ND4J indexing
    private static org.nd4j.linalg.indexing.INDArrayIndex point(int i) {
        return org.nd4j.linalg.indexing.NDArrayIndex.point(i);
    }
    private static org.nd4j.linalg.indexing.INDArrayIndex all() {
        return org.nd4j.linalg.indexing.NDArrayIndex.all();
    }
}