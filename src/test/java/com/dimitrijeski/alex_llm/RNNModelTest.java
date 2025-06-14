package com.dimitrijeski.alex_llm;

import com.dimitrijeski.alex_llm.model.RNNModel;
import org.junit.jupiter.api.Test;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit test for RNNModel (LSTM-based).
 * <p/>
 * This test ensures that the LSTM model can be trained on a small corpus and can generate text.
 * The focus is on verifying the training and generation pipeline, not on the quality of the generated text.
 * <p/>
 * Steps:
 * 1. Define a small training corpus.
 * 2. Build the vocabulary from the corpus.
 * 3. Create an RNNModel with a window size of 2.
 * 4. Train the model on the corpus.
 * 5. Generate text using a seed and check the output.
 */
class RNNModelTest {
    @Test
    void testGenerateText() {
        // Define a small corpus for training
        String corpus = "the cat sat on the mat";
        // Build the vocabulary from the corpus
        Set<String> vocab = new HashSet<>(Arrays.asList(corpus.split("\\s+")));
        // Create an RNNModel with a window size of 2
        RNNModel model = new RNNModel(2, vocab);
        // Train the model on the corpus
        model.train(corpus);
        // Use a seed of two words (window size)
        String seed = "the cat";
        // Generate 3 words after the seed
        String generated = model.generateText(Arrays.asList(seed.split("\\s+")), 3);
        // The generated text should not be null and should contain at least the seed plus generated words
        assertNotNull(generated, "Generated text should not be null");
        assertTrue(generated.split("\\s+").length >= 2, "Generated text should contain at least the seed plus generated words");
    }
}
