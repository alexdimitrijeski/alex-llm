package com.dimitrijeski.alex_llm.network;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

/**
 * CustomMultiLayerNetwork extends DL4J's MultiLayerNetwork to allow custom behavior.
 * <p/>
 * Example customizations:
 * - Scaling gradients during backpropagation (for experimentation or debugging)
 * - Adding custom logging during training
 * <p/>
 * This class is a starting point for experimenting with neural network internals.
 * <p/>
 * Other useful methods you can override for advanced customization:
 * - output(...) : Customize how predictions are made.
 * - score(...) : Change how the loss is computed.
 * - update(...) : Modify how parameters are updated.
 * - fit(DataSet) or fit(DataSetIterator) : Add hooks or custom logic for training on datasets.
 * - predict(...) : Change prediction logic or add post-processing.
 * - setListeners(...) : Attach custom listeners for monitoring training.
 * - init() : Customize initialization of network parameters.
 */
@Slf4j
public class CustomMultiLayerNetwork extends MultiLayerNetwork {

    /**
     * Constructs the network with the given configuration.
     * @param conf MultiLayerConfiguration describing the network architecture
     */
    public CustomMultiLayerNetwork(MultiLayerConfiguration conf) {
        super(conf);
    }

    /**
     * Customizes backpropagation by scaling all gradients by 0.5.
     * Scaling the gradients means multiplying the computed gradients by a constant factor (here, 0.5)
     * before updating the model's weights. This reduces the size of each weight update, which can
     * slow down learning or help stabilize training.
     * <p/>
     * Example:
     *   Suppose the computed gradient for a weight is 0.8.
     *   With scaling by 0.5, the gradient becomes 0.4 before the weight update.
     *   This means the weight will be updated less aggressively.
     *
     * @param epsilon The error signal to backpropagate
     * @param workspaceMgr Workspace manager for memory
     * @return Pair of (Gradient, INDArray) as in the base implementation
     */
    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        Pair<Gradient, INDArray> originalGradient = super.backpropGradient(epsilon, workspaceMgr);
        Gradient gradient = originalGradient.getFirst();
        // Add small noise to gradients to encourage exploration.
        for (String key : gradient.gradientForVariable().keySet()) {
            INDArray grad = gradient.getGradientFor(key);
            try (INDArray noise = Nd4j.randn(grad.shape())) {
                grad.addi(noise.muli(0.1));
            }
        }
        return originalGradient;
    }

    /**
     * Overrides the fit method to add custom logging.
     * The fit method is responsible for training the neural network on a batch of input data.
     * It performs forward and backward passes, computes gradients, and updates the model's weights
     * to minimize the loss between predictions and labels.
     * <p/>
     * Example:
     *   INDArray input = ...;  // input features
     *   INDArray labels = ...; // expected outputs
     *   model.fit(input, labels); // trains the model on this batch
     *
     * @param input Input features
     * @param labels Target labels (expected outputs for each input)
     */
    @Override
    public void fit(INDArray input, INDArray labels) {
        // Example: Add custom logging during training
        log.info("Custom fit method called");
        log.info("Input shape: {}", Arrays.toString(input.shape()));
        log.info("Labels shape: {}", Arrays.toString(labels.shape()));
        if (input.rows() > 0) {
            log.info("Sample input row: {}", input.getRow(0));
            log.info("Sample label row: {}", labels.getRow(0));
        }
        log.info("Custom fit method called");
        super.fit(input, labels);
    }
}