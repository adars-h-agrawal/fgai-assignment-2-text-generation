# Comparative Autoregressive Text Generation Using RNNs

## 1. Problem Statement and Objective

The objective is to develop an autoregressive next-word text-generation
system and compare Simple RNN, Bidirectional RNN, LSTM and GRU models
under a consistent experimental setup.

## 2. Dataset

- **Name:** Shakespeare Plays
- **Source:** https://www.kaggle.com/datasets/kingburrito666/shakespeare-plays
- **File used:** Shakespeare_data.csv
- **Column used:** PlayerLine
- **Non-empty dialogue lines:** 111,396
- **Token count:** 1,002,945

The text is tokenized at the word level. The vocabulary is built using
training text only. Data is split chronologically into 80% training,
10% validation and 10% test partitions. Sequence windows are constructed
inside each partition.

## 3. Models

- **Simple RNN:** A vanilla recurrent neural network.
- **Bidirectional RNN:** Processes the supplied context in forward and
  backward directions.
- **LSTM:** Uses gated memory to regulate information flow.
- **GRU:** Uses update and reset gates with a simpler gated architecture.

Each model receives a context of 24 tokens and predicts the
next token. Models share the vocabulary, embedding size, hidden-unit
count, output layer, optimiser and training protocol. The Bidirectional
RNN has a different parameter count because it has two recurrent directions.

## 4. Experimental Configuration

- Random seed: 42
- Vocabulary size: 10000
- Context length: 24
- Embedding dimension: 64
- Hidden units: 96
- Batch size: 256
- Maximum epochs: 10
- Optimiser: Adam
- Initial learning rate: 0.001
- Early stopping: validation loss, patience 3
- Temperature values: [0.5, 1.0, 1.5]
- Generated tokens per sample: 60
- Generation prompts: ['to be or not to be that is the question', 'wherefore art thou romeo', "all the world's a stage"]

Early stopping and learning-rate reduction are based on validation loss.
The best validation-loss checkpoint is evaluated on the test partition.
The test partition is not used for model selection.

## 5. Quantitative Results

           Model  Parameters  Best_Epoch  Epochs_Trained  Training_Time_Seconds  Validation_Loss  Validation_Accuracy  Validation_Perplexity  Test_Loss  Test_Accuracy  Test_Perplexity
             GRU     1989072           3               6                  14.84         5.874403               0.1288             355.812177   5.729193         0.1230       307.720756
            LSTM     2004240           4               7                  20.16         5.895957               0.1304             363.564598   5.739556         0.1267       310.926426
       SimpleRNN     1957872           4               7                  14.40         5.908167               0.1273             368.031069   5.728740         0.1228       307.581538
BidirectionalRNN     1985616           3               6                  15.70         6.011996               0.1245             408.297579   5.865927         0.1156       352.809136

Loss is sparse categorical cross-entropy. Perplexity is exp(loss);
lower loss and perplexity generally indicate better next-token
probability estimates. Accuracy is the fraction of examples whose
highest-probability prediction matches the target. Training time is
hardware- and runtime-dependent.

## 6. Model Selection

The selected model is **GRU**, based on the lowest validation
loss (5.8744). Its held-out test loss
is 5.7292, test accuracy is
0.1230, and test perplexity is
307.72.

This selection criterion does not guarantee that the model is best for
every qualitative generation characteristic. The other models should
also be discussed using the quantitative comparison and generated samples.

## 7. Temperature Analysis

All four models generate text from the same three seed prompts at
temperatures [0.5, 1.0, 1.5], producing 36 samples.

See `generated_samples.csv`, `generated_samples.txt`,
`temperature_analysis.csv` and `model_temperature_analysis.csv`.

Lower temperature generally concentrates probability on more likely
tokens. Higher temperature generally increases sampling randomness.
Distinct-1 and Distinct-2 measure unique unigram and bigram proportions.
Repetition ratio measures 1 minus the unique-token fraction.
These metrics describe lexical variation, not semantic coherence.

## 8. Qualitative Evaluation

Inspect generated passages for coherence, contextual consistency,
readability and repetition. Use `qualitative_evaluation_worksheet.csv`
to record human ratings and observations. Qualitative conclusions must
be based on the actual generated passages and completed ratings; they
are not inferred from diversity metrics alone.

## 9. Limitations

This is a word-level model with a finite vocabulary and context window.
The Shakespeare corpus contains historical language and stage directions.
A Bidirectional RNN is included as required for comparison, but it is
not a conventional causal language-model architecture. It only processes
the supplied context and does not receive future target tokens.

The experiment uses a bounded sample of sequence windows and a maximum
epoch count. Results depend on the dataset version, hardware, random
seed and training configuration. Further tuning or multiple runs may
change the model ranking.

## 10. Reproducibility

1. Open the notebook in Kaggle.
2. Attach the Shakespeare Plays dataset.
3. Select a GPU accelerator if available.
4. Run all notebook cells in order.
5. Review the generated samples and qualitative worksheet.

Dependencies: Python, TensorFlow, NumPy, pandas and Matplotlib.

## 11. Files

- `model_comparison.csv`: quantitative model metrics
- `*_training_history.csv`: per-epoch training and validation metrics
- `generated_samples.csv` and `.txt`: generated passages
- `temperature_analysis.csv`: mean generation metrics by temperature
- `model_temperature_analysis.csv`: generation metrics by model and temperature
- `qualitative_evaluation_worksheet.csv`: human evaluation template
- PNG files: comparison and temperature plots
- `experiment_summary.json`: experiment configuration and results

Dataset attribution and licence terms should be checked before redistribution.
