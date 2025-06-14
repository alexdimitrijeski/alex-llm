package com.dimitrijeski.alex_llm.model;

import java.io.Serial;

public class BigramModel extends NGramModel implements java.io.Serializable {
    @Serial
    private static final long serialVersionUID = 1L;

    public BigramModel() {
        super(2);
    }
}
