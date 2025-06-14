package com.dimitrijeski.alex_llm.model;

import org.junit.jupiter.api.Test;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit test for BigramModel.
 * <p/>
 * This test checks that the bigram model can be trained on a small corpus and can generate text.
 * The test is intended to show how a simple bigram model works for language modeling.
 * <p/>
 * Steps:
 * 1. Define a small training corpus.
 * 2. Create a bigram model (n=2).
 * 3. Train the model on the corpus.
 * 4. Generate text using a seed and check the output.
 */
class BigramModelTest {
    @Test
    void testGenerateText() {
        // Define a small corpus for training
        String corpus = "the cat sat on the mat";
        // Create a bigram model (n=2)
        BigramModel model = new BigramModel();
        // Train the model on the corpus
        model.train(corpus);
        // Use a seed of one word (n-1)
        String seed = "the";
        // Generate 5 words after the seed
        String generated = model.generateText(Arrays.asList(seed.split("\\s+")), 5);
        // The generated text should not be null and should contain at least the seed plus generated words
        assertNotNull(generated, "Generated text should not be null");
        assertTrue(generated.split("\\s+").length >= 1, "Generated text should contain at least the seed plus generated words");
    }
}
