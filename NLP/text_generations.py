def generating_stories(markov_model, start = 'the', limit = 150) :
  n = 0 
  current_state = start
  text = " "
  text += current_state + " 
  while n < limit :
    word_list = [word for word in markov_model[current_state].keys()]
    #print(markov_model[current_state].keys())
    value_list = [word for word in  markov_model[current_state].values()]
    future_state = random.choices(word_list, value_list)
   # print(future_state)
    future_state = future_state[0]
    story += future_state + " "
    current_state = future_state
    n += 1
    
  return story
