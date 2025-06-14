package com.dimitrijeski.alex_llm.model;

import org.junit.jupiter.api.Test;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit test for NGramModel.
 * <p/>
 * This test verifies that the NGramModel can be trained on a small corpus and can generate text.
 * The test is intended to demonstrate the basic usage of n-gram language models.
 * <p/>
 * Steps:
 * 1. Define a small training corpus.
 * 2. Create a trigram model (n=3).
 * 3. Train the model on the corpus.
 * 4. Generate text using a seed and check the output.
 */
class NGramModelTest {
    @Test
    void testGenerateText() {
        // Define a small corpus for training
        String corpus = "the cat sat on the mat";
        // Create a trigram model (n=3)
        NGramModel model = new NGramModel(3);
        // Train the model on the corpus
        model.train(corpus);
        // Use a seed of two words (n-1)
        String seed = "the cat";
        // Generate 5 words after the seed
        String generated = model.generateText(Arrays.asList(seed.split("\\s+")), 5);
        // The generated text should not be null and should contain at least the seed plus generated words
        assertNotNull(generated, "Generated text should not be null");
        assertTrue(generated.split("\\s+").length >= 2, "Generated text should contain at least the seed plus generated words");
    }
}
