from transformers import pipeline
from pprint import pprint
import esupar

# https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne-capitel-pos
model_es = pipeline("token-classification", model="PlanTL-GOB-ES/roberta-large-bne-capitel-pos")
example_es = "Me llamo José"

# https://huggingface.co/KoichiYasuoka/bert-base-russian-upos
model_ru = esupar.load("KoichiYasuoka/bert-base-russian-upos")
example_ru = "Меня зовут Дмитрий"

# https://github.com/KoichiYasuoka/esupar
model_ko = esupar.load("ko")
example_ko = "내 이름은 Kim입니다"

pos_results_es = model_es(example_es)
pos_results_ru = model_ru(example_ru)
pos_results_ko = model_ko(example_ko)

pprint(pos_results_es)
pprint(pos_results_ru)
pprint(pos_results_ko)