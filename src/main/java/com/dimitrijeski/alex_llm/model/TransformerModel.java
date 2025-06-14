package com.dimitrijeski.alex_llm.model;

import org.deeplearning4j.datasets.iterator.utilty.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SelfAttentionLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.Serial;
import java.util.*;

/**
 * TransformerModel demonstrates a minimal transformer-like language model using DL4J's SelfAttentionLayer.
 * This class is for educational purposes and is not a full transformer implementation.
 * <p/>
 * The model predicts the next word given a window of previous words, using self-attention to capture relationships
 * between words in the input window. The architecture is:
 * <p/>
 * Input (one-hot encoded sequence) ->
 * SelfAttentionLayer ->
 * GlobalPoolingLayer (reduces sequence dimension) ->
 * DenseLayer ->
 * OutputLayer (softmax over vocabulary)
 *
 * <h2>Layer Types and Activations</h2>
 * <ul>
 *   <li><b>SelfAttentionLayer</b>: Implements self-attention, allowing the model to weigh the importance of each word in the input window relative to the others.
 *       <ul>
 *         <li>No explicit activation function; the attention mechanism itself is the core operation.</li>
 *         <li>nHeads: Number of attention heads (parallel attention mechanisms).</li>
 *       </ul>
 *   </li>
 *   <li><b>GlobalPoolingLayer</b>: Reduces the sequence dimension (windowSize) to a single vector by averaging (PoolingType.AVG) across the sequence.
 *       <ul>
 *         <li>No activation function; this is a pooling operation.</li>
 *       </ul>
 *   </li>
 *   <li><b>DenseLayer</b>: A fully connected layer for further processing after attention and pooling.
 *       <ul>
 *         <li>Activation: <b>RELU</b> (Rectified Linear Unit). Adds non-linearity and helps with gradient flow.</li>
 *       </ul>
 *   </li>
 *   <li><b>OutputLayer</b>: Final layer to produce the next word prediction.
 *       <ul>
 *         <li>Activation: <b>SOFTMAX</b>. Converts the output into a probability distribution over the vocabulary.</li>
 *         <li>Loss: <b>MCXENT</b> (Multi-Class Cross Entropy). Used for multi-class classification.</li>
 *       </ul>
 *   </li>
 * </ul>
 */
public class TransformerModel implements LanguageModel {
    @Serial
    private static final long serialVersionUID = 1L;
    // Size of the input window (number of previous words to consider)
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
     * Constructs a TransformerModel with the specified window size and vocabulary.
     * @param windowSize number of previous words to use as input
     * @param vocab corpus vocabulary (unique words)
     */
    public TransformerModel(int windowSize, Set<String> vocab) {
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
     * Trains the transformer-like model on the given text.
     * <p/>
     * The training data is constructed as follows:
     * - For each window of 'windowSize' words in the corpus, the input is a one-hot encoded sequence of those words.
     * - The label is the next word after the window, one-hot encoded.
     * <p/>
     * The model uses a SelfAttentionLayer to learn relationships between words in the window,
     * then pools the sequence dimension, and finally predicts the next word.
     *
     * @param text the training corpus
     */
    @Override
    public TransformerModel train(String text) {
        // Tokenize the input text into words
        String[] tokens = text.split("\\s+");
        List<INDArray> inputs = new ArrayList<>();
        List<INDArray> labels = new ArrayList<>();

        // For each possible window in the corpus, create input and label arrays
        for (int i = 0; i <= tokens.length - windowSize - 1; i++) {
            // Input: shape [1, vocabSize, windowSize] (one-hot encoded sequence)
            INDArray input = Nd4j.zeros(1, vocabSize, windowSize);
            for (int j = 0; j < windowSize; j++) {
                int idx = wordToIdx.get(tokens[i + j]);
                input.putScalar(0, idx, j, 1.0);
            }
            inputs.add(input);

            // Label: shape [1, vocabSize] (one-hot encoded next word)
            INDArray label = Nd4j.zeros(1, vocabSize);
            int labelIdx = wordToIdx.get(tokens[i + windowSize]);
            label.putScalar(0, labelIdx, 1.0);
            labels.add(label);
        }

        // Stack all input and label arrays into a single DataSet
        DataSet ds = new DataSet(Nd4j.concat(0, inputs.toArray(new INDArray[0])), Nd4j.concat(0, labels.toArray(new INDArray[0])));
        // Use a DataSetIterator for batching
        ListDataSetIterator<DataSet> iter = new ListDataSetIterator<>(ds.asList(), 16);

        // Build the model configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(0.01))
                .list()
                // SelfAttentionLayer captures relationships between words in the input window
                .layer(new SelfAttentionLayer.Builder()
                        .nIn(vocabSize)
                        .nOut(64)
                        .nHeads(2)
                        .projectInput(true) // Required when nHeads > 1
                        .build())
                // GlobalPoolingLayer reduces the sequence dimension (windowSize) to a single vector
                .layer(new GlobalPoolingLayer.Builder()
                        .poolingType(PoolingType.AVG)
                        .build())
                // DenseLayer for further processing
                .layer(new DenseLayer.Builder()
                        .nIn(64)
                        .nOut(32)
                        .activation(Activation.RELU)
                        .build())
                // OutputLayer with softmax activation to predict the next word
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nIn(32)
                        .nOut(vocabSize)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        // Initialize and train the model
        model = new MultiLayerNetwork(conf);
        model.init();
        // Fit the model for a small number of epochs (for demonstration)
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
            // Prepare input: shape [1, vocabSize, windowSize]
            INDArray input = Nd4j.zeros(1, vocabSize, windowSize);
            for (int j = 0; j < windowSize; j++) {
                String word = result.get(result.size() - windowSize + j);
                Integer idx = wordToIdx.get(word);
                if (idx != null) {
                    input.putScalar(0, idx, j, 1.0);
                }
            }
            // Model output: [1, vocabSize]
            INDArray output = model.output(input);
            int nextIdx = Nd4j.argMax(output, 1).getInt(0);
            result.add(idxToWord.get(nextIdx));
        }
        return String.join(" ", result);
    }
}