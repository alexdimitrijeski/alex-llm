package com.dimitrijeski.alex_llm.config;

import org.junit.jupiter.api.Test;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.core.io.ResourceLoader;

import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class DataPreLoaderTest {

    @Test
    void testCreateTables() {
        JdbcTemplate jdbcTemplate = mock(JdbcTemplate.class);
        ResourceLoader resourceLoader = mock(ResourceLoader.class);

        DataPreLoader preLoader = new DataPreLoader(jdbcTemplate, resourceLoader);
        preLoader.createTables();

        verify(jdbcTemplate, times(2)).execute(anyString());
    }

    @Test
    void testLoadTrainingCorpus() throws IOException {
        JdbcTemplate jdbcTemplate = mock(JdbcTemplate.class);
        ResourceLoader resourceLoader = mock(ResourceLoader.class);
        org.springframework.core.io.Resource resource = mock(org.springframework.core.io.Resource.class);

        when(resourceLoader.getResource(anyString())).thenReturn(resource);
        when(resource.exists()).thenReturn(false); // Simulate file not existing

        DataPreLoader preLoader = new DataPreLoader(jdbcTemplate, resourceLoader);
        String corpus = preLoader.loadTrainingCorpus();

        assertNull(corpus); // Should be null if file does not exist
    }
}
