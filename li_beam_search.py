import torch

def parse(model, tokens, subword_end_mask, beam_size, word_beam_size,
          shift_size, return_beam_history=False, stack_size_bound=-1):
    return model.word_sync_beam_search(tokens, subword_end_mask, beam_size,
                                       word_beam_size, shift_size,
                                       return_beam_history=return_beam_history,
                                       stack_size_bound=stack_size_bound)

def validate_actions(action_list):
  valid = []
  depth = 0
  for a in action_list:
    if a == 2:          # REDUCE
      if depth > 1:
        valid.append(a)
        depth -= 1
    elif a > 2:         # NT(X)
      valid.append(a)
      depth += 1
    else:               # SHIFT
      valid.append(a)
  for _ in range(depth):
    valid.append(2)
  return valid

def beam_search(model, device, dataset, beam_size=200, word_beam_size=20,
                shift_size=5, block_size=100, stack_size_bound=-1,
                max_length_diff=20):

    with torch.no_grad():

        sents = []
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
                if stack_size_bound < tokens.size(1):
                    parses, surprisals = parse(model, tokens, subword_end_mask,
                                               beam_size, word_beam_size,
                                               shift_size, stack_size_bound=-1)
            
            best_actions = [p[0][0] for p in parses] # p[0][1] is likelihood
            subword_end_mask = subword_end_mask.cpu().numpy()
            for i in range(len(batch_idx)):
                sents.append(
                    {'action_ids': validate_actions(best_actions[i]),
                     'tokens': dataset.sents[batch_idx[i]].tokens,
                     'token_ids': dataset.sents[batch_idx[i]].token_ids
                    })
    
            # print(sents)
    return sents

def load_model(checkpoint, action_dict, vocab):
  if 'model_state_dict' in checkpoint:
    from train import create_model
    model = create_model(checkpoint['args'], action_dict, vocab)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
  else:
    return checkpoint['model']
