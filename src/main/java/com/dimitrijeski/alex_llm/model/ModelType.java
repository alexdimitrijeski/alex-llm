package com.dimitrijeski.alex_llm.model;

import lombok.Getter;

@Getter
public enum ModelType {
    UNIGRAM("unigram"),
    BIGRAM("bigram"),
    NGRAM("ngram"),
    FFNN("ffnn"),
    RNN("rnn"),
    TRANSFORMER("transformer"),
    ALEX("alex");

    private final String name;

    ModelType(String name) {
        this.name = name;
    }

}