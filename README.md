## Build a LLM from Scratch, Raschka

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
