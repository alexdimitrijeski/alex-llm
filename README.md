# LLM Playground

This project is an **LLM Playground** for experimenting with Java-based language models, including n-gram and neural models (DL4J). It provides a modular architecture for training, evaluating, and interacting with language models via a REST API. It is designed for anyone who wants to learn about language models and experiment with them in a hands-on way.

## Components

- **Controller**: Exposes a REST API for interacting with the language models. You can generate text, train models, and query model status via HTTP endpoints.
- **Postman Collection**: A ready-to-use Postman collection is provided for testing the API endpoints easily. (See `postman_collection.json` or similar in the repo.)
- **Preloader**: Handles loading and preprocessing of training data before model training.
- **training_data.txt**: A text file containing sample datasets for training and evaluating models.

## Project Structure

- `src/` - Java source code for models, controller, and utilities.
- `training_data.txt` - Example dataset for model training.
- `postman_collection.json` - Postman collection for API testing.
- `README.md` - Project documentation.

## How it works

- **UnigramModel**: Predicts the next word based only on word frequency.
- **BigramModel**: Predicts the next word based on the previous word.
- **NGramModel**: Predicts the next word based on the previous (n-1) words. You can specify `n` for custom models.
- **FeedForwardNNModel**: Simple feedforward neural network language model using DL4J.
- **RNNModel**: Simple LSTM (RNN) language model using DL4J.
- **TransformerModel**: Minimal Transformer-based language model using DL4J's SelfAttentionLayer.
- **AlexLanguageModel**: A custom language model designed specifically for this project, offering enhanced capabilities for text generation and evaluation.

All models implement the `LanguageModel` interface, so you can easily add your own.

## Setup

1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/alex-llm.git
   cd alex-llm
   ```

2. **Add DL4J dependencies**  
   Add the required DL4J dependencies to your `pom.xml` (see [DL4J documentation](https://deeplearning4j.konduit.ai/)).

3. **Compile the Java files**
   ```sh
   mvn clean install
   ```

4. **Prepare training data**  
   Place your datasets in the `training_data.txt` file.

5. **Run the application**
   - To run from the command line:
     ```sh
     java -cp target/alex-llm-1.0-SNAPSHOT.jar Main
     ```
   - Or run `Main.java` from your IDE.

6. **Start the Controller (REST API)**
   - Run the controller class (e.g., `Controller.java`) to start the REST API server.

7. **Test the API**
   - Import the provided Postman collection and use it to interact with the API endpoints.

8. **Configure train-on-startup**  
   To enable automatic training on startup, set the `train-on-startup` property to `true` in the YAML configuration file (`config.yaml`):
   ```yaml
   train-on-startup: true
   ```

## API Endpoints

Typical endpoints exposed by the controller include:
- `POST /train` - Train a model with provided data.
- `POST /generate` - Generate text using a specified model.
- `GET /status` - Get model or server status.

Refer to the Postman collection for detailed request/response examples.

## Custom Models

To create your own model, implement the `LanguageModel` interface or instantiate `NGramModel` with your desired `n`.

## Code Documentation

All classes and methods are documented in the code for clarity.

## Requirements

- Java 8+
- [Deeplearning4j (DL4J)](https://deeplearning4j.konduit.ai/) for neural models
- Maven (for dependency management and building)

## Learning and Experimentation

This project is ideal for anyone who wants to learn about language models and experiment with them. Whether you're exploring n-gram models or diving into neural networks, this playground provides a practical way to understand and interact with LLMs.

## Attribution

This project was developed with significant assistance from **GitHub Copilot**, an AI-powered coding assistant. Copilot was used to generate much of the code and documentation, making the development process faster and more efficient.

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.  
You are free to:
- Share — copy and redistribute the material in any medium or format.
- Adapt — remix, transform, and build upon the material.

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
- **NonCommercial** — You may not use the material for commercial purposes.

For more details, see the [license description](https://creativecommons.org/licenses/by-nc/4.0/).