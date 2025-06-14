package com.dimitrijeski.alex_llm.model;

import org.junit.jupiter.api.Test;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

class AlexLanguageModelTest {

    @Test
    void testTrain() {
        Set<String> vocab = new HashSet<>(Arrays.asList("I", "like", "cats"));
        AlexLanguageModel model = new AlexLanguageModel(2, vocab);
        model.train("I like cats");

        assertNotNull(model);
        assertEquals(3, model.getVocabSize());
    }

    @Test
    void testGenerateText() {
        Set<String> vocab = new HashSet<>(Arrays.asList("I", "like", "cats"));
        AlexLanguageModel model = new AlexLanguageModel(2, vocab);
        model.train("I like cats");

        List<String> seed = Arrays.asList("I", "like");
        String generatedText = model.generateText(seed, 3);

        assertNotNull(generatedText);
        assertTrue(generatedText.contains("cats"));
    }
}
