package com.dimitrijeski.alex_llm.network;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CustomMultiLayerNetworkTest {

    @Test
    void testCustomFitLogging() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Adam(0.01))
                .list()
                .layer(new DenseLayer.Builder().nIn(10).nOut(5).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE).nIn(5).nOut(1).activation(Activation.IDENTITY).build())
                .build();

        CustomMultiLayerNetwork network = new CustomMultiLayerNetwork(conf);
        network.init();

        assertDoesNotThrow(() -> network.fit(Nd4j.rand(10, 10), Nd4j.rand(10, 1)));
    }
}
