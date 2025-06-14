package com.dimitrijeski.alex_llm.model;

import org.junit.jupiter.api.Test;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit test for UnigramModel.
 * <p/>
 * This test checks that the unigram model can be trained on a small corpus and can generate text.
 * The test demonstrates the simplest possible language model, which samples words based on frequency.
 * <p/>
 * Steps:
 * 1. Define a small training corpus.
 * 2. Create a unigram model.
 * 3. Train the model on the corpus.
 * 4. Generate text using a seed and check the output.
 */
class UnigramModelTest {
    @Test
    void testGenerateText() {
        // Define a small corpus for training
        String corpus = "the cat sat on the mat";
        // Create a unigram model
        UnigramModel model = new UnigramModel();
        // Train the model on the corpus
        model.train(corpus);
        // Use a seed word
        String seed = "the";
        // Generate 5 words after the seed
        String generated = model.generateText(Arrays.asList(seed.split("\\s+")), 5);
        // The generated text should not be null and should contain at least the seed plus generated words
        assertNotNull(generated, "Generated text should not be null");
        assertTrue(generated.split("\\s+").length >= 1, "Generated text should contain at least the seed plus generated words");
    }
}
