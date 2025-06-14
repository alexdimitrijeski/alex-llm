package com.dimitrijeski.alex_llm.api;

import com.dimitrijeski.alex_llm.model.AlexLanguageModel;
import com.dimitrijeski.alex_llm.model.LanguageModel;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentMatchers;
import org.springframework.jdbc.core.JdbcTemplate;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class ModelPlaygroundControllerTest {

    @Test
    void testTrainEndpoint() {
        JdbcTemplate jdbcTemplate = mock(JdbcTemplate.class);
        ModelPlaygroundController controller = new ModelPlaygroundController(jdbcTemplate);

        ModelPlaygroundController.TrainRequest request = new ModelPlaygroundController.TrainRequest();
        request.setCorpus("I like cats");
        request.setWindowSize(2);

        String response = controller.train("alex", request);
        assertTrue(response.contains("trained and saved"));
    }

    @Test
    void testGenerateEndpoint() {
        JdbcTemplate jdbcTemplate = mock(JdbcTemplate.class);
        ModelPlaygroundController controller = new ModelPlaygroundController(jdbcTemplate);

        LanguageModel model = mock(AlexLanguageModel.class);
        when(model.generateText(anyList(), anyInt())).thenReturn("Generated text");

        controller.models.put("alex", model);

        ModelPlaygroundController.GenerateRequest request = new ModelPlaygroundController.GenerateRequest();
        request.setSeed("I like");
        request.setLength(3);

        String response = controller.generate("alex", request);
        assertEquals("Generated text", response);
    }

    @Test
    void testListModels() {
        JdbcTemplate jdbcTemplate = mock(JdbcTemplate.class);
        ModelPlaygroundController controller = new ModelPlaygroundController(jdbcTemplate);

        List<String> models = controller.listModels();
        assertNotNull(models);
        assertTrue(models.contains("alex"));
        assertTrue(models.contains("unigram"));
    }

    @Test
    void testListTrainingDataNames() {
        JdbcTemplate jdbcTemplate = mock(JdbcTemplate.class);
        // Mock the RowMapper version of query (disambiguate with ArgumentMatchers)
        when(jdbcTemplate.query(eq("SELECT name FROM training_data"), ArgumentMatchers.<org.springframework.jdbc.core.RowMapper<String>>any()))
            .thenReturn(Arrays.asList("data1", "data2"));

        ModelPlaygroundController controller = new ModelPlaygroundController(jdbcTemplate);

        List<String> trainingDataNames = controller.listTrainingDataNames();
        assertNotNull(trainingDataNames);
        assertEquals(2, trainingDataNames.size());
        assertTrue(trainingDataNames.contains("data1"));
        assertTrue(trainingDataNames.contains("data2"));
    }

    @Test
    void testGetTrainingData() {
        JdbcTemplate jdbcTemplate = mock(JdbcTemplate.class);
        // Mock the RowMapper version of query with parameter (disambiguate with ArgumentMatchers)
        when(jdbcTemplate.query(eq("SELECT corpus FROM training_data WHERE name = ?"), ArgumentMatchers.<org.springframework.jdbc.core.RowMapper<String>>any(), eq("data1")))
            .thenReturn(Collections.singletonList("Sample corpus"));

        ModelPlaygroundController controller = new ModelPlaygroundController(jdbcTemplate);

        String trainingData = controller.getTrainingData("data1");
        assertNotNull(trainingData);
        assertEquals("Sample corpus", trainingData);
    }
}