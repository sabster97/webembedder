import csv
def get_sys_prompt():
    sys_prompt_dict = {}
    with open("./sys_prompts.csv", mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            for key in reader.fieldnames:
                sys_prompt_dict[key] = row[key]
    
    return sys_prompt_dict

