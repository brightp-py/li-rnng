## Steps

- [ ] Load in a database of sentences.
    - [ ] Pull sentenes from Penn Treebank.
    - [ ] Replace nonterminals with dummy "AP", "BP", etc.
- [ ] Generate "decided" tree to compare others to.
    - [ ] Use beam search to create trees from a batch of sentences.
    - [ ] Remove one word from the sentence to be re-decided later.
    - [ ] Assign a random other sentence to the decided tree as well.
    - [ ] Do one of the following:
        - [ ] Change dynamic Database object with new trees.
        - [ ] Create new Database object.
- [ ] Compare the augmented tree to the decided one.
    - [ ] Run training on augmented sentence with decided tree being correct.
- [ ] Compare the random tree to the decided one.
    - [ ] Run training on random tree with NEGATIVE PUNISHMENT against decided.