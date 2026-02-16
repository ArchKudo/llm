## Build a LLM from Scratch, Raschka

## About

### Running the project

- Tools used to bootstrap python: `pixi`

https://pixi.prefix.dev/latest/#installation

- Install uv for dependency management
    - Mostly as `pixi` doesn't allow separation between deps, devDeps

```
pixi add uv
```

- Use precommit for cleaning up file formatting

```
pixi shell
uv add --dev pre-commit
```

- Run a watcher using `watchexec`

```
watchexec -e py pixi run uv run python embeddings.py
```


### Notes

- Parameters
	-  Adjustable weights in the network optimized during training to predict the next sequence.
- Embeddings
	- Mapping from discrete objects words, images, etc. to continuous vector space
	- Can be of words, sentences, paragraphs, or large texts
	- Sentence embedding are useful for Retrieval Augmentented generation to generate texts from relevant information
	- For predicting one word at a time word embeddings are used
	- Gives semantic meaning
	- Opposed to Tokens
		- Used to segment text into linguistic representation
		- Text -> Tokens -> Embeddings
		- From github issues, it is what tokens represent in llm pricing
		- Well-know [Tiktoken](https://github.com/openai/tiktoken) is based on [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding) algorithm
- Deep learning allows self-supervised learning where it creates its own labels, without requiring manual curation
- Process for creating an LLM
	- Pretraining
		- First training stage, often called a base or foundation model.
		- e.g. GPT3 which allows text completion, and few-shot techniques allowing to perform task base on few instructions  instead of large texts
	- Finetuning
		- Further training possible on labelled data

### Updates

#### 1

- Create a repository using pixi, found via: [RNA folding pipeline](https://github.com/JaneliaSciComp/jrc-rna-structure-pipeline?tab=readme-ov-file#prerequisites)

- Download The_Verdict.txt from Wiki: https://ws-export.wmcloud.org/?lang=en&page=The+Verdict&format=txt&fonts=&credits=false&images=false


- Add embeddings.py
    - Open the The_Verdict.txt which has 21K characters and read the file

### 2

- Create a simple token class
    - Initialized from a book
        - Remove all non-alphanumerics and store them as tokens {word: token}
    - Provide encode and decode functions for sampled text

### 3

- Embeddings v2
    - Use tiktoken library
    - Used special context clues <endoftext>
