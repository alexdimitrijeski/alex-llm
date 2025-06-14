package com.dimitrijeski.alex_llm.api;

import com.dimitrijeski.alex_llm.model.*;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;
import org.springframework.jdbc.core.JdbcTemplate;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.io.*;

@Slf4j
@RestController
@RequestMapping("/api/models")
@RequiredArgsConstructor
@SuppressWarnings("unused")
public class ModelPlaygroundController {

    public static final String WORD_TO_IDX = "wordToIdx";
    public static final String WINDOW_SIZE = "windowSize";
    public static final String IDX_TO_WORD = "idxToWord";
    public static final String MODEL = "model";
    public static final String ERROR_READING_FIELDS = "Error reading fields: ";
    private final JdbcTemplate jdbcTemplate;

    final Map<String, LanguageModel> models = new ConcurrentHashMap<>();

    @GetMapping
    public List<String> listModels() {
        return Arrays.asList("unigram", "bigram", "ngram", "ffnn", "rnn", "transformer", "alex");
    }

    @PostMapping("/{model}/train")
    public String train(
            @PathVariable String model,
            @RequestBody TrainRequest req
    ) {
        String corpus = req.getCorpus();
        int windowSize = req.getWindowSize() != null ? req.getWindowSize() : 3;
        int ngramN = req.getNgramN() != null ? req.getNgramN() : 3;

        Set<String> vocab = new HashSet<>(Arrays.asList(corpus.split("\\s+")));

        LanguageModel lm;
        switch (model.toLowerCase()) {
            case "unigram":
                lm = new UnigramModel();
                break;
            case "bigram":
                lm = new BigramModel();
                break;
            case "ngram":
                lm = new NGramModel(ngramN);
                break;
            case "ffnn":
                lm = new FeedForwardNNModel(windowSize, vocab);
                break;
            case "rnn":
                lm = new RNNModel(windowSize, vocab);
                break;
            case "transformer":
                lm = new TransformerModel(windowSize, vocab);
                break;
            case "alex":
                lm = new AlexLanguageModel(windowSize, vocab);
                break;
            default:
                return "Unknown model: " + model;
        }
        lm.train(corpus);
        models.put(model, lm);
        saveModelToDb(model, lm);
        return "Model '" + model + "' trained and saved.";
    }

    @PostMapping("/{model}/generate")
    public String generate(
            @PathVariable String model,
            @RequestBody GenerateRequest req
    ) {
        LanguageModel lm = models.get(model);
        if (lm == null) {
            // Try to load from DB
            lm = loadModelFromDb(model);
            if (lm == null) {
                return "Model not trained yet: " + model;
            }
            models.put(model, lm);
        }
        List<String> seed = Arrays.asList(req.getSeed().split("\\s+"));
        int length = req.getLength() != null ? req.getLength() : 10;
        return lm.generateText(seed, length);
    }

    @GetMapping("/training-data")
    public List<String> listTrainingDataNames() {
        return jdbcTemplate.query("SELECT name FROM training_data", (rs, rowNum) -> rs.getString("name"));
    }

    @GetMapping("/training-data/{name}")
    public String getTrainingData(@PathVariable String name) {
        List<String> result = jdbcTemplate.query("SELECT corpus FROM training_data WHERE name = ?",
                (rs, rowNum) -> rs.getString("corpus"), name);
        return result.isEmpty() ? "" : result.get(0);
    }

    private String getModelDetails(LanguageModel lm) {
        if (lm instanceof FeedForwardNNModel ffnn) {
            return getFeedForwardNNModelDetails(ffnn);
        } else if (lm instanceof RNNModel rnn) {
            return getRNNModelDetails(rnn);
        } else if (lm instanceof TransformerModel transformer) {
            return getTransformerModelDetails(transformer);
        } else if (lm instanceof BigramModel bigram) {
            return getBigramModelDetails(bigram);
        } else if (lm instanceof NGramModel ngram) {
            return getNGramModelDetails(ngram);
        } else if (lm instanceof UnigramModel unigram) {
            return getUnigramModelDetails(unigram);
        } else if (lm instanceof AlexLanguageModel alex) {
            return getAlexLanguageModelDetails(alex);
        } else {
            return getUnknownModelDetails(lm);
        }
    }

    private String getFeedForwardNNModelDetails(FeedForwardNNModel ffnn) {
        StringBuilder sb = new StringBuilder("FeedForwardNNModel details:\n");
        try {
            sb.append("  " + WINDOW_SIZE + ": ").append(getFieldValue(ffnn, WINDOW_SIZE)).append("\n");
            sb.append("  " + WORD_TO_IDX + ": ").append(getFieldValue(ffnn, WORD_TO_IDX)).append("\n");
            sb.append("  " + IDX_TO_WORD + ": ").append(getFieldValue(ffnn, IDX_TO_WORD)).append("\n");
            sb.append("  " + MODEL + ": ").append(getModelDetailsFromDL4J(ffnn)).append("\n");
        } catch (Exception e) {
            sb.append(ERROR_READING_FIELDS).append(e.getMessage());
        }
        return sb.toString();
    }

    private String getRNNModelDetails(RNNModel rnn) {
        StringBuilder sb = new StringBuilder("RNNModel details:\n");
        try {
            sb.append("  " + WINDOW_SIZE + ": ").append(getFieldValue(rnn, WINDOW_SIZE)).append("\n");
            sb.append("  " + WORD_TO_IDX + ": ").append(getFieldValue(rnn, WORD_TO_IDX)).append("\n");
            sb.append("  " + IDX_TO_WORD + ": ").append(getFieldValue(rnn, IDX_TO_WORD)).append("\n");
            sb.append("  " + MODEL + ": ").append(getModelDetailsFromDL4J(rnn)).append("\n");
        } catch (Exception e) {
            sb.append(ERROR_READING_FIELDS).append(e.getMessage());
        }
        return sb.toString();
    }

    private String getTransformerModelDetails(TransformerModel transformer) {
        StringBuilder sb = new StringBuilder("TransformerModel details:\n");
        try {
            sb.append("  " + WINDOW_SIZE + ": ").append(getFieldValue(transformer, WINDOW_SIZE)).append("\n");
            sb.append("  " + WORD_TO_IDX + ": ").append(getFieldValue(transformer, WORD_TO_IDX)).append("\n");
            sb.append("  " + IDX_TO_WORD + ": ").append(getFieldValue(transformer, IDX_TO_WORD)).append("\n");
            sb.append("  " + MODEL + ": ").append(getModelDetailsFromDL4J(transformer)).append("\n");
        } catch (Exception e) {
            sb.append(ERROR_READING_FIELDS).append(e.getMessage());
        }
        return sb.toString();
    }

    private String getNGramModelDetails(NGramModel ngram) {
        StringBuilder sb = new StringBuilder("NGramModel details:\n");
        try {
            @SuppressWarnings("unchecked")
            Map<List<String>, List<String>> ngramMap = (Map<List<String>, List<String>>) getFieldValue(ngram, "ngramMap");
            sb.append("  ngramMap (context -> next words):\n");
            for (Map.Entry<List<String>, List<String>> entry : ngramMap.entrySet()) {
                sb.append("    ").append(entry.getKey()).append(" -> ").append(entry.getValue()).append("\n");
            }
        } catch (NoSuchFieldException e) {
            sb.append(ERROR_READING_FIELDS).append(e.getMessage()).append("\n");
            sb.append("Available fields:\n");
            for (var field : ngram.getClass().getDeclaredFields()) {
                sb.append("  ").append(field.getName()).append("\n");
            }
        } catch (Exception e) {
            sb.append(ERROR_READING_FIELDS).append(e.getMessage());
        }
        return sb.toString();
    }

    private String getBigramModelDetails(BigramModel bigram) {
        return getNGramModelDetails(bigram);
    }

    private String getUnigramModelDetails(UnigramModel unigram) {
        StringBuilder sb = new StringBuilder("UnigramModel details:\n");
        try {
            @SuppressWarnings("unchecked")
            List<String> words = (List<String>) getFieldValue(unigram, "words");
            Map<String, Integer> freq = new HashMap<>();
            for (String w : words) freq.put(w, freq.getOrDefault(w, 0) + 1);
            sb.append("  Word frequencies:\n");
            for (Map.Entry<String, Integer> entry : freq.entrySet()) {
                sb.append("    ").append(entry.getKey()).append(": ").append(entry.getValue()).append("\n");
            }
        } catch (Exception e) {
            sb.append(ERROR_READING_FIELDS).append(e.getMessage());
        }
        return sb.toString();
    }

    private String getAlexLanguageModelDetails(AlexLanguageModel alex) {
        StringBuilder sb = new StringBuilder("AlexLanguageModel details:\n");
        try {
            sb.append("  " + WINDOW_SIZE + ": ").append(getFieldValue(alex, WINDOW_SIZE)).append("\n");
            sb.append("  " + WORD_TO_IDX + ": ").append(getFieldValue(alex, WORD_TO_IDX)).append("\n");
            sb.append("  " + IDX_TO_WORD + ": ").append(getFieldValue(alex, IDX_TO_WORD)).append("\n");
            sb.append("  " + MODEL + ": ").append(getModelDetailsFromDL4J(alex)).append("\n");
        } catch (Exception e) {
            sb.append(ERROR_READING_FIELDS).append(e.getMessage());
        }
        return sb.toString();
    }

    @SuppressWarnings("java:S3011")
    private String getUnknownModelDetails(LanguageModel lm) {
        StringBuilder sb = new StringBuilder("Unknown model type. Fields:\n");
        try {
            for (var field : lm.getClass().getDeclaredFields()) {
                field.setAccessible(true);
                sb.append("  ").append(field.getName()).append(": ").append(field.get(lm)).append("\n");
            }
        } catch (Exception e) {
            sb.append(ERROR_READING_FIELDS).append(e.getMessage());
        }
        return sb.toString();
    }

    @SuppressWarnings("java:S3011")
    private Object getFieldValue(Object obj, String fieldName) throws NoSuchFieldException, IllegalAccessException {
        var field = obj.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        return field.get(obj);
    }

    private String getModelDetailsFromDL4J(Object obj) throws NoSuchFieldException, IllegalAccessException {
        Object value = getFieldValue(obj, ModelPlaygroundController.MODEL);
        if (value instanceof org.deeplearning4j.nn.multilayer.MultiLayerNetwork mln) {
            return "DL4J Model Configuration:\n" + mln.getLayerWiseConfigurations().toJson() +
                    "\nNumber of parameters: " + mln.numParams();
        }
        return value.toString();
    }

    @GetMapping("/{model}/data")
    public String getModelData(@PathVariable String model) {
        LanguageModel lm = loadModelFromDb(model);
        if (lm == null) {
            return "No stored model data for: " + model;
        }
        return getModelDetails(lm);
    }

    // --- Model Serialization Helpers ---

    private void saveModelToDb(String name, LanguageModel model) {
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream();
             ObjectOutputStream out = new ObjectOutputStream(bos)) {
            out.writeObject(model);
            out.flush();
            byte[] data = bos.toByteArray();
            // Upsert logic
            int updated = jdbcTemplate.update(
                    "UPDATE trained_models SET model_data = ? WHERE name = ?",
                    data, name
            );
            if (updated == 0) {
                jdbcTemplate.update(
                        "INSERT INTO trained_models (name, model_data) VALUES (?, ?)",
                        name, data
                );
            }
        } catch (IOException e) {
            log.error(e.getMessage(), e);
        }
    }

    private LanguageModel loadModelFromDb(String name) {
        try {
            List<byte[]> results = jdbcTemplate.query(
                    "SELECT model_data FROM trained_models WHERE name = ?",
                    (rs, rowNum) -> rs.getBytes("model_data"), // Use getBytes to map to byte[]
                    name
            );

            if (results.isEmpty()) return null;
            byte[] data = results.get(0);
            try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(data))) {
                return (LanguageModel) in.readObject();
            }
        } catch (Exception e) {
            log.error(e.getMessage(), e);
            return null;
        }
    }

    // DTOs
    @Setter
    @Getter
    public static class TrainRequest {
        private String corpus;
        private Integer windowSize;
        private Integer ngramN;

    }

    @Setter
    @Getter
    public static class GenerateRequest {
        private String seed;
        private Integer length;

    }
}
