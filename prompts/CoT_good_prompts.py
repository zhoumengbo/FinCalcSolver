CoT_en_prompts = {
    'CoT_1': "Below is a question about financial calculation.\nQuestion: {query}",
    'CoT_3': "Below is a question about financial calculation.\nQuestion: {query}\nLet's solve the question step by "
             "step.",
    'CoT_4': "Below is a question about financial calculation.\nQuestion: {query}\nLet's list all calculation formulas",
    'CoT_5': "Below is a question about financial calculation.\nQuestion: {query}\nPlease list all variables and "
             "required calculation formulas in the question.",
    'CoT_6': "Below is a question about financial calculation.\nQuestion: {query}\nPlease list all the numerical "
             "information and calculation formulas in the question.",
    'CoT_11': "Please first list all the numerical information in the question. Focus exclusively on the numbers "
              "given in the question, assigning each a unique variable. Then list all calculation formulas."
              "\nQuestion: {query}\nAnswer:",
    'CoT_15': open("prompts/template/CoT_shot/CoT_two-shot", 'r').read(),
}
