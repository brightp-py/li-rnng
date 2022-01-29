## Steps

- [x] Load in a database of sentences.
    - [x] Pull sentences from Penn Treebank.
        - [x] Generate vocabulary.
        - [x] Save sequences of sentence ids.
        - [x] Replace nonterminals with dummy "AP", "BP", etc.
- [ ] Generate "decided" tree to compare others to.
    - [ ] Use beam search to create trees from a batch of sentences.
        - [ ] Create Database for beam search use.
        - [ ] Turn trees into something useful.
    - [ ] Remove one word from the sentence to be re-decided later.
    - [ ] Assign a random other sentence to the decided tree as well.
    - [x] Do one of the following:
        - [ ] Change dynamic Database object with new trees.
        - [x] Create new Database object.
- [ ] Compare the augmented tree to the decided one.
    - [ ] Run training on augmented sentence with decided tree being correct.
- [ ] Compare the random tree to the decided one.
    - [ ] Run training on random tree with NEGATIVE PUNISHMENT against decided.