package com.dimitrijeski.alex_llm.config;

import com.dimitrijeski.alex_llm.model.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.CommandLineRunner;
import org.springframework.context.annotation.Configuration;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.jdbc.core.JdbcTemplate;

import java.io.*;
import java.nio.file.Files;
import java.util.*;

@Slf4j
@Configuration
@RequiredArgsConstructor
@SuppressWarnings("unused")
public class DataPreLoader implements CommandLineRunner {

    private final JdbcTemplate jdbcTemplate;

    private final ResourceLoader resourceLoader;

    @Override
    public void run(String... args) throws Exception {
        log.info("Application startup: DataPreLoader is running...");
        log.info("Step 1/5: Creating database tables...");
        createTables();
        log.info("Step 2/5: Reloading training data...");
        String trainingCorpus = loadTrainingCorpus();
        log.info("Step 3/5: Clearing trained models...");
        clearTrainedModels();
        log.info("Step 4/5: Training supported models...");
        try {
            trainSupportedModels(trainingCorpus);
        } catch (UnsatisfiedLinkError e) {
            log.error("Native library loading failed. This may be due to missing CUDA or other native dependencies required by ND4J/DL4J.");
            log.error("Error message: {}", e.getMessage());
            log.error("If you do not have a compatible GPU or CUDA installation, consider using the CPU backend for ND4J.");
            throw e;
        }
        log.info("Step 5/5: Startup complete.");
        log.info("DataPreLoader finished: All preloading and training tasks are complete.");
        log.info("Application is ready for use. Controllers are ready to accept requests.");
    }

    // Loads the entire training_data.txt as a single corpus string
    String loadTrainingCorpus() throws IOException {
        Resource resource = resourceLoader.getResource("classpath:training_data.txt");
        if (resource.exists()) {
            List<String> lines = Files.readAllLines(resource.getFile().toPath());
            String corpus = String.join(" ", lines).trim();
            if (corpus.isEmpty()) {
                log.warn("training_data.txt is empty. No models will be trained.");
                return null;
            }
            return corpus;
        } else {
            log.warn("training_data.txt not found. No models will be trained.");
            return null;
        }
    }

    void createTables() {
        jdbcTemplate.execute("CREATE TABLE IF NOT EXISTS training_data (" +
                "id IDENTITY PRIMARY KEY, " +
                "name VARCHAR(255), " +
                "corpus CLOB" +
                ")");
        jdbcTemplate.execute("CREATE TABLE IF NOT EXISTS trained_models (" +
                "name VARCHAR(255) PRIMARY KEY, " +
                "model_data BLOB" +
                ")");
    }

    private void clearTrainedModels() {
        jdbcTemplate.execute("DELETE FROM trained_models");
    }

    private void trainSupportedModels(String trainingCorpus) {
        List<ModelType> supportedModels = getSupportedModels();

        if (trainingCorpus == null || trainingCorpus.isEmpty()) {
            log.warn("No training corpus available. Skipping model training.");
            return;
        }

        int total = supportedModels.size();
        int count = 0;
        for (ModelType modelName : supportedModels) {
            count++;
            log.info("Training model {}/{}: {} ...", count, total, modelName.getName());
            LanguageModel model = createAndTrainModel(modelName, trainingCorpus);
            if (model != null) {
                saveModelToDb(modelName, model);
                log.info("Model {} trained and saved.", modelName.getName());
            } else {
                log.warn("Model {} could not be trained.", modelName.getName());
            }
        }
        log.info("All {} models trained.", total);
    }

    private List<ModelType> getSupportedModels() {
        return Arrays.asList(ModelType.values());
    }

    private LanguageModel createAndTrainModel(ModelType modelType, String corpus) {
        Set<String> vocab = new HashSet<>(Arrays.asList(corpus.split("\\s+")));
        int windowSize = 3;
        int ngramN = 3;

        return switch (modelType) {
            case UNIGRAM -> new UnigramModel().train(corpus);
            case BIGRAM -> new BigramModel().train(corpus);
            case NGRAM -> new NGramModel(ngramN).train(corpus);
            case FFNN -> new FeedForwardNNModel(windowSize, vocab).train(corpus);
            case RNN -> new RNNModel(windowSize, vocab).train(corpus);
            case TRANSFORMER -> new TransformerModel(windowSize, vocab).train(corpus);
            case ALEX -> new AlexLanguageModel(windowSize, vocab).train(corpus);
        };
    }

    private void saveModelToDb(ModelType name, LanguageModel model) {
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream();
             ObjectOutputStream out = new ObjectOutputStream(bos)) {
            out.writeObject(model);
            out.flush();
            byte[] data = bos.toByteArray();
            // Upsert logic
            int updated = jdbcTemplate.update(
                "UPDATE trained_models SET model_data = ? WHERE name = ?",
                data, name.getName()
            );
            if (updated == 0) {
                jdbcTemplate.update(
                    "INSERT INTO trained_models (name, model_data) VALUES (?, ?)",
                    name.getName(), data
                );
            }
        } catch (IOException e) {
            log.error(e.getMessage(), e);
        }
    }
}