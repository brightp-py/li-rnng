from data import LI_Dataset
from fixed_stack_in_order_models import FixedStackInOrderRNNG
from in_order_models import InOrderRNNG
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

        # block_idx = []
        sents = []
        # block_surprisals = []
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
            # trees = [action_dict.build_tree_str(best_actions[i],
            #                                     dataset.sents[batch_idx[i]].token_ids,
            #                                     ['X' for _ in dataset.sents[batch_idx[i]].token_ids],
            #                                     subword_end_mask[i])
            #     for i in range(len(batch_idx))]
            # block_idx.extend(batch_idx)
            for i in range(len(batch_idx)):
                sents.append(
                    {'action_ids': best_actions[i],
                     'token_ids': dataset.sents[batch_idx[i]].token_ids
                    })
            # block_surprisals.extend(surprisals)
            # cur_block_size += tokens.size(0)

            # if cur_block_size >= args.block_size:
            #     assert cur_block_size == args.block_size
            #     sort_and_print_trees(block_idxs, block_parses, block_surprisals)
            #     block_idxs = []
            #     block_parses = []
            #     block_surprisals = []
            #     cur_block_size = 0
    
    return sents

def load_model(checkpoint, action_dict, vocab):
  if 'model_state_dict' in checkpoint:
    from train import create_model
    model = create_model(checkpoint['args'], action_dict, vocab)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
  else:
    return checkpoint['model']

if __name__ == "__main__":
    device = 'cuda:0'

    torch.manual_seed(1)
    checkpoint = torch.load("rnng.pt")
    vocab = checkpoint['vocab']
    action_dict = checkpoint['action_dict']
    prepro_args = checkpoint['prepro_args']
    model = load_model(checkpoint, action_dict, vocab).to(device)

    li_dataset = LI_Dataset.from_json("data/ptb-train.json", 16)
    model.eval()

    if isinstance(model, InOrderRNNG) or isinstance(model, FixedStackInOrderRNNG):
        model.max_cons_nts = 3
    
    block_parses = beam_search(model, device, li_dataset.get_beam_dataset())
    print(block_parses)