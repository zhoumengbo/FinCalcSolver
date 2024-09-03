CoT_en_prompts = {
    'CoT_1': "Below is a question about financial calculation.\nQuestion: {query}",
    'CoT_2': "Below is a question about financial calculation.\nQuestion: {query}\nLet's calculate step by step.",
    'CoT_3': "Below is a question about financial calculation.\nQuestion: {query}\nLet's solve the question step by "
             "step.",
    'CoT_4': "Below is a question about financial calculation.\nQuestion: {query}\nLet's list all calculation formulas",
    'CoT_5': "Below is a question about financial calculation.\nQuestion: {query}\nPlease list all variables and "
             "required calculation formulas in the question.",
    'CoT_6': "Below is a question about financial calculation.\nQuestion: {query}\nPlease list all the numerical "
             "information and calculation formulas in the question.",
    'CoT_7': "Below is a question about financial calculation.\nQuestion: {query}\nPlease list all the numerical "
             "information in the question. Focus exclusively on the numbers given in the question, assigning each a "
             "unique variable.",
    'CoT_8': "Below is a question about financial calculation.\nQuestion: {query}\nPlease first list all the numerical"
             "information in the question. Focus exclusively on the numbers given in the question, assigning each a "
             "unique variable. Then list all calculation formulas.",
    'CoT_9': "Below is a question about financial calculation.\nQuestion: {query}\nPlease list the following in the "
             "specified format: variable definition, concept interpretation, calculation formula.\nAnswer:",
    'CoT_10': "Please list all the numerical information and calculation formulas in the question.\nQuestion: {"
              "query}\nAnswer:",
    'CoT_11': "Please first list all the numerical information in the question. Focus exclusively on the numbers "
              "given in the question, assigning each a unique variable. Then list all calculation formulas."
              "\nQuestion: {query}\nAnswer:",
    'CoT_12': "Below is a question about financial calculation. Question: {query}\nSeparate your answer into the "
              "following 3 steps:\n1. Think: what formulas or theorems should be used in order to solve this problem? "
              "Write out the formulas and theorems.\n2. Apply the above written formulas and theorems to the "
              "question: write out your solution symbolically and abstractly. Make sure you don't do any numerical "
              "calculations at this stage.\n3. Then separately, substitute the numerical values from the question "
              "into your solution and work out a number as your final answer.",

    'CoT_13': open("prompts/template/CoT_shot/CoT_one-shot", 'r').read(),
    'CoT_14': open("prompts/template/CoT_shot/CoT_example-shot", 'r').read(),
    'CoT_15': open("prompts/template/CoT_shot/CoT_two-shot", 'r').read(),

    'CoT_16': "Please list all variables and required calculation formulas in the question."
              "\nQuestion: {query}\nAnswer:",
}
