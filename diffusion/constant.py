generate_kwargs = {
    'beam': 
    {'max_length':64, 'min_length':5, 'do_sample':False, 'num_beams':4, 'no_repeat_ngram_size':3, 'repetition_penalty':1.2},
    'nucleus':
    {'max_length':64, 'min_length':5, 'do_sample':True, 'top_p':.95, 'num_beams':1, 'no_repeat_ngram_size':3, 'repetition_penalty':1.2}}
