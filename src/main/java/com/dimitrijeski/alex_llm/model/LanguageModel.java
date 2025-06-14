package com.dimitrijeski.alex_llm.model;

import java.io.Serializable;
import java.util.List;

public interface LanguageModel extends Serializable {
    LanguageModel train(String text);
    String generateText(List<String> seed, int length);
}
