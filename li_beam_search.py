import torch

def parse(model, tokens, subword_end_mask, beam_size, word_beam_size,
          shift_size, return_beam_history=False, stack_size_bound=-1):
    return model.word_sync_beam_search(tokens, subword_end_mask, beam_size,
                                       word_beam_size, shift_size,
                                       return_beam_history=return_beam_history,
                                       stack_size_bound=stack_size_bound)

def beam_search(model, device, dataset, beam_size=200, word_beam_size=20,
                shift_size=5, block_size=100, stack_size_bound=-1,
                max_length_diff=20):
    
    # cur_block_size = 0

    with torch.no_grad():

        block_idx = []
        block_parses = []
        block_surprisals = []
        batches = [batch for batch in dataset.test_batches(
                    block_size, max_length_diff)]
        
        for batch in batches:
            tokens, subword_end_mask, batch_idx = batch
            tokens = tokens.to(device)
            subword_end_mask = subword_end_mask.to(device)

            parses, surprisals = parse(model, tokens, subword_end_mask,
                                       beam_size, word_beam_size, shift_size,
                                       stack_size_bound=stack_size_bound)
            if any(len(p) == 0 for p in parses):
                # failed_sents = [(idx, " ".join(dataset.sents[idx].orig_tokens))
                #                 for p, idx in zip(parses, batch_idx)
                #                 if len(p) == 0]
                # failed_sents_str = "\n".join(f"{i}: {s}"
                #                              for i, s in failed_sents)
                if stack_size_bound < tokens.size(1):
                    parses, surprisals = parse(model, tokens, subword_end_mask,
                                               beam_size, word_beam_size,
                                               shift_size, stack_size_bound=-1)
            
            best_actions = [p[0][0] for p in parses] # p[0][1] is likelihood
            subword_end_mask = subword_end_mask.cpu().numpy()
            trees = something
            block_idx.extend(batch_idx)
            block_parses.extend(trees)
            block_surprisals.extend(surprisals)
            # cur_block_size += tokens.size(0)

            # if cur_block_size >= args.block_size:
            #     assert cur_block_size == args.block_size
            #     sort_and_print_trees(block_idxs, block_parses, block_surprisals)
            #     block_idxs = []
            #     block_parses = []
            #     block_surprisals = []
            #     cur_block_size = 0