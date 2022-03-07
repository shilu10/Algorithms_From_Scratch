def markov_chains(a_list1) :
  transition_values = { }
  for i in range(len(a_list1)-1) :
    current_state = a_list1[i]
    future_state = a_list1[i + 1]
    if not current_state in transition_values :
      transition_values[current_state] = { }
      transition_values[current_state][future_state] = 1
    else :
      if not future_state in transition_values[current_state] :
        transition_values[current_state][future_state] = 1
      else :
        transition_values[current_state][future_state] += 1
  for current_state, future_state in transition_values.items() :
    total = sum(future_state.values()) 
    for key, value in future_state.items() :
      value = value/total 
      transition_values[current_state][key] = value       
  return transition_values
