# anymat2vec
Network for generating word2vec embeddings for any material composition, regardless of whether it is in the training corpus. Also, models for predicting materials properties from these generated embeddings. Based on the ideas in https://github.com/materialsintelligence/mat2vec



## todos

#### by submodule
 
- `common_data`: everything needed for downloading/preprocessing commonly used training data
    - raw training data + methods to retrieve/access
    - preprocessed, normalized, input ready training data with clear processing pipeline scripts
- `word2vec`: the word2vec model, preferably in gensim
    - `train`: train a word2vec model
    - `pretrained`: load/access previously trained models
    - `tests`: should test against the full analogy set with decent accuracy
- `hiddenrep`: the hiddenrep model which outputs embeddings
    - `train`: train a new hiddenrep model, given a word2vec model
    - `pretrained`: load/access previously trained hiddenrep models
    - `tests`: ?
- `tasks`: train, validate, and use models for real-world tasks
    - `regression`: 
        - `data`: everything needed for loading/preprocessing/etc regression task data
        - `train`: train a new regression model given a hiddenrep pretrained model
    - `analogy_testing`:
        - `data`: access to full analogy sets; load and create _restricted_ analogy sets for hiddenrep validation
        - `validate`: validate the word2vec+hiddenrep network on analogies


## conceptual problems with old repo

#### analogy testing
- some of the analogies words (e.g., 'Db', various common compounds) are not in the corpus(BPE?)