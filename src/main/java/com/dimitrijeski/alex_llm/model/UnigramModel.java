package com.dimitrijeski.alex_llm.model;

import java.io.Serial;
import java.util.*;

/**
 * UnigramModel is the simplest possible language model.
 * It predicts the next word by randomly sampling from the observed word frequencies in the training corpus.
 * <p/>
 * This model ignores all context and only considers the overall distribution of words.
 * It is useful as a baseline and for understanding the limitations of context-free models.
 */
public class UnigramModel implements LanguageModel {
    @Serial
    private static final long serialVersionUID = 1L;
    // List of all words seen in the training corpus (including duplicates for frequency)
    private final List<String> words = new ArrayList<>();
    private final Random random = new Random();

    /**
     * Trains the model by collecting all words from the corpus.
     * The more frequent a word, the more likely it is to be sampled during generation.
     *
     * @param text the training corpus
     */
    @Override
    public UnigramModel train(String text) {
        // Split the text into tokens (words) and add them to the list
        String[] tokens = text.split("\\s+");
        words.addAll(Arrays.asList(tokens));
        return this;
    }

    /**
     * Generates text by randomly sampling words from the corpus.
     * The seed is ignored for the unigram model, as it does not use context.
     * @param seed ignored for unigram model
     * @param length number of words to generate
     * @return generated text
     */
    @Override
    public String generateText(List<String> seed, int length) {

        List<String> result = new ArrayList<>(seed);
        for (int i = 0; i < length; i++) {
            if (words.isEmpty()) break;
            // Randomly pick a word from the list (frequency-based)
            result.add(words.get(random.nextInt(words.size())));
        }
        return String.join(" ", result);
    }
}