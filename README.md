# Autoregressive Text Generation Using RNN, BiRNN, LSTM and GRU

**Course:** ICT4403 — Foundations of Generative AI  
**Assignment:** Assignment 2 — Text Generation  
**Author:** Adarsh Agrawal  
**Institution:** Manipal Institute of Technology, MAHE

## 1. Overview

This project implements and compares four recurrent neural network architectures for autoregressive text generation using Shakespeare's plays:

- Simple Recurrent Neural Network (SimpleRNN)
- Bidirectional Recurrent Neural Network (Bidirectional RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

The models are trained to predict the next word from a sequence of preceding words. After training, each model generates text sequentially from the same seed prompts. The experiments compare predictive performance, training time, generated-text characteristics, and the effect of sampling temperature.

## 2. Objectives

1. Preprocess a textual dataset and construct next-word prediction sequences.
2. Train and compare SimpleRNN, Bidirectional RNN, LSTM and GRU under comparable experimental conditions.
3. Evaluate the models using loss, accuracy, perplexity and training time.
4. Examine how sampling temperatures affect generated-text diversity and repetition.
5. Identify the most suitable model based on the observed experimental results.

## 3. Dataset

**Dataset:** Shakespeare Plays  
**Source:** [Shakespeare Plays dataset on Kaggle](https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays)

The dataset contains dialogue from Shakespeare's plays. It provides a literary text corpus with historical English vocabulary, character dialogue, and varied sentence structures, making it suitable for experimenting with next-word prediction and text generation.

### Preprocessing and sequence construction

The notebook performs the following operations:

1. Loads the `Shakespeare_data.csv` file and selects the `PlayerLine` text field.
2. Removes empty text entries and combines the text for tokenization.
3. Converts the text into integer token sequences using a vocabulary capped at 10,000 tokens.
4. Constructs fixed-length contexts of 24 tokens for next-word prediction.
5. Uses a chronological 80% / 10% / 10% train-validation-test split.
6. Limits the generated training examples to 50,000, with 10,000 examples each for validation and testing.

The resulting dataset provides the context-target pairs required for autoregressive next-word prediction.

## 4. Model Architectures

| Model | Description |
|---|---|
| SimpleRNN | Uses recurrent hidden states to process sequential word representations. |
| Bidirectional RNN | Processes the context in forward and backward directions to build a bidirectional representation. |
| LSTM | Uses input, forget and output gates with a cell state to model sequential dependencies. |
| GRU | Uses update and reset gates to control information flow through recurrent states. |

All four models use an embedding layer and a classification output over the vocabulary. The experiment uses an embedding dimension of 64, 96 recurrent hidden units, a dense layer of 128 units, and a vocabulary of up to 10,000 tokens.

## 5. Experimental Setup

The experiments were run in Kaggle Notebooks using TensorFlow/Keras and the available GPU environment.

| Setting | Value |
|---|---|
| Context length | 24 tokens |
| Vocabulary limit | 10,000 |
| Embedding dimension | 64 |
| Recurrent hidden units | 96 |
| Dense units | 128 |
| Batch size | 256 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Maximum epochs | 10 |
| Training controls | Early stopping and learning-rate reduction |
| Prediction task | Next-word prediction |

The notebook contains the implementation, training procedure, evaluation, plots and text-generation experiments.

## 6. Quantitative Results

The following values are taken from the recorded notebook execution.

| Model | Best epoch | Training time (s) | Validation loss | Validation accuracy | Test loss | Test accuracy | Test perplexity |
|---|---:|---:|---:|---:|---:|---:|---:|
| SimpleRNN | 4 | 14.40 | 5.9082 | 12.73% | 5.7287 | 12.28% | 307.58 |
| Bidirectional RNN | 3 | 15.70 | 6.0120 | 12.45% | 5.8659 | 11.56% | 352.81 |
| LSTM | 4 | 20.16 | 5.8960 | 13.04% | 5.7396 | 12.67% | 310.93 |
| GRU | 3 | 14.84 | 5.8744 | 12.88% | 5.7292 | 12.30% | 307.72 |

**Interpretation**

- GRU achieved the lowest validation loss (5.8744), so it was selected as the primary model based on validation loss.
- SimpleRNN achieved marginally lower test loss and test perplexity than GRU.
- LSTM achieved the highest test accuracy (12.67%) but required the longest recorded training time among these runs.
- Bidirectional RNN had the highest test loss and perplexity and the lowest test accuracy in this comparison.
- The differences are relatively small for some metrics; therefore, no single model should be described as best across every metric.

Perplexity is derived from cross-entropy loss and reflects how uncertain the model is about the next token. Lower perplexity is generally preferable when comparing models evaluated on the same tokenization and test set.

## 7. Text Generation and Sampling Temperature

Each of the four models generated text using the same three seed prompts:

- `to be or not to be that is the question`
- `wherefore art thou romeo`
- `all the world's a stage`

Generation was evaluated at temperatures of 0.5, 1.0 and 1.5, producing 36 samples in total.

Temperature changes the sharpness of the next-token probability distribution:

- **Temperature 0.5:** Concentrates probability on more likely tokens. The observed outputs had lower lexical diversity and more repetition.
- **Temperature 1.0:** Provides a more balanced sampling distribution, with greater diversity than at temperature 0.5.
- **Temperature 1.5:** Produces more varied and less repetitive tokens, but greater diversity does not guarantee grammatical or contextual coherence.

### Recorded diversity and repetition

| Temperature | Distinct-1 | Distinct-2 | Repetition ratio |
|---|---:|---:|---:|
| 0.5 | 0.4444 | 0.8192 | 0.5556 |
| 1.0 | 0.7722 | 0.9958 | 0.2278 |
| 1.5 | 0.9306 | 1.0000 | 0.0694 |

Distinct-1 measures the proportion of unique unigrams, while Distinct-2 measures the proportion of unique bigrams among generated text. The repetition ratio measures the repeated-token fraction as implemented in the notebook.

In these experiments, increasing temperature raised lexical diversity and reduced the measured repetition ratio. However, qualitative inspection also showed that many outputs were not consistently grammatical or contextually coherent. Temperature therefore represents a trade-off between predictable, repetitive output and more varied, less predictable output.

## 8. Qualitative Analysis

The generated samples contain Shakespearean vocabulary and references to characters or familiar phrases. However, the models often fail to maintain coherent syntax, consistent meaning, or strong continuation of the original prompt.

The lower-temperature outputs tend to be more repetitive, while higher-temperature outputs vary more widely in wording. This is consistent with the quantitative diversity measurements.

These observations are qualitative descriptions of the generated examples, not results from a formal human-rating study. The generated samples and the qualitative evaluation worksheet are included with the results where available.

## 9. Conclusion

This assignment compared SimpleRNN, Bidirectional RNN, LSTM and GRU for autoregressive next-word prediction on Shakespeare's plays.

GRU was selected as the primary model because it achieved the lowest validation loss. However, SimpleRNN obtained slightly better test loss and perplexity, while LSTM achieved the highest test accuracy. The results demonstrate why model selection should consider multiple evaluation metrics rather than relying on a single score.

The temperature experiments showed that increasing sampling temperature improved lexical diversity and reduced repetition in the generated samples, but did not guarantee improved coherence. The experiment highlights the importance of evaluating both quantitative prediction metrics and qualitative generation behaviour.

## 10. Repository Contents

- `notebooks/` — Jupyter Notebook containing preprocessing, model training, evaluation and text generation.
- `results/metrics/` — Quantitative evaluation results.
- `results/plots/` — Training and comparison plots.
- `results/` — Generated text samples, qualitative worksheet, training histories and metadata.

Actual filenames and subfolders may vary slightly depending on the exported results bundle.

## 11. How to Run

1. Open the notebook in Kaggle or another compatible Jupyter environment.
2. Make the Shakespeare Plays dataset available to the notebook.
3. Install or enable the required TensorFlow/Keras, pandas, NumPy and Matplotlib dependencies if they are not already available.
4. Execute the notebook cells in order to preprocess the dataset, train the four models, evaluate them and generate samples.
5. Inspect the exported metrics, plots and generated-text files in the results directory.

Training time and outputs can vary depending on the hardware, software versions and execution environment.

## 12. Limitations

- The vocabulary is limited to the configured maximum size.
- The models predict one next word at a time and do not explicitly optimize for long-range semantic consistency.
- The generated outputs can be grammatically weak or contextually inconsistent.
- The temperature comparison uses a limited set of prompts and samples.
- The quantitative metrics alone do not fully capture the quality of generated language.

## 13. Acknowledgements

The Shakespeare Plays dataset is sourced from Kaggle. TensorFlow/Keras is used for neural-network implementation and training, and Kaggle Notebooks provides the experimental environment.

---

**Author:** Adarsh Agrawal  
**Course:** ICT4403 — Foundations of Generative AI  
**Assignment:** Assignment 2 — Autoregressive Text Generation
