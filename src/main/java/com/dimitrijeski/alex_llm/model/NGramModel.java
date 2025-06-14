package com.dimitrijeski.alex_llm.model;

import java.io.Serial;
import java.util.*;

/**
 * NGramModel is a generalization of the bigram and trigram models.
 * It predicts the next word based on the previous (n-1) words, where n is configurable.
 * <p/>
 * The model learns a mapping from (n-1)-word contexts to possible next words, and samples from the observed continuations.
 * This approach demonstrates how increasing context size can improve language modeling, but also increases data sparsity.
 */
public class NGramModel implements LanguageModel {
    @Serial
    private static final long serialVersionUID = 1L;
    // The size of the n-gram (e.g., 2 for bigram, 3 for trigram)
    private final int n;
    // Mapping from (n-1)-word context to a list of possible next words
    private final Map<List<String>, List<String>> ngramMap = new HashMap<>();

    private final Random random = new Random();

    /**
     * Constructs an NGramModel with the specified n-gram size.
     * @param n the size of the n-gram (e.g., 2 for bigram, 3 for trigram)
     */
    public NGramModel(int n) {
        this.n = n;
    }

    /**
     * Trains the model on the given text corpus.
     * For each sequence of (n-1) words, records the next word that follows in the corpus.
     *
     * @param text the training corpus
     */
    @Override
    public NGramModel train(String text) {
        String[] tokens = text.split("\\s+");
        for (int i = 0; i <= tokens.length - n; i++) {
            // Extract the (n-1)-word context as the key
            List<String> key = Arrays.asList(Arrays.copyOfRange(tokens, i, i + n - 1));
            // The next word after the context
            String nextWord = tokens[i + n - 1];
            // Store the next word as a possible continuation for this context
            ngramMap.computeIfAbsent(key, k -> new ArrayList<>()).add(nextWord);
        }

        return this;
    }

    /**
     * Generates text starting from the given seed.
     * At each step, predicts the next word based on the previous (n-1) words.
     * If the context is unseen, generation stops.
     * @param seed the initial words to start generation
     * @param length the number of words to generate
     * @return the generated text
     */
    @Override
    public String generateText(List<String> seed, int length) {
        List<String> result = new ArrayList<>(seed);
        for (int i = 0; i < length; i++) {
            if (result.size() < n - 1 || ngramMap.getOrDefault(result.subList(result.size() - (n - 1), result.size()), Collections.emptyList()).isEmpty()) {
                break;
            }
            // Get the last (n-1) words as the context
            List<String> key = result.subList(result.size() - (n - 1), result.size());
            List<String> possibleNext = ngramMap.get(key);
            // Randomly pick a possible next word for this context
            String next = possibleNext.get(random.nextInt(possibleNext.size()));
            result.add(next);
        }
        return String.join(" ", result);
    }
}