from transformers import pipeline
from pprint import pprint
import esupar

# https://huggingface.co/PlanTL-GOB-ES/roberta-large-bne-capitel-pos
model_es = pipeline("token-classification", model="PlanTL-GOB-ES/roberta-large-bne-capitel-pos")
example_es = "Tengo un bonito perro. Mi perro corre rápido."

# https://huggingface.co/KoichiYasuoka/bert-base-russian-upos
model_ru = esupar.load("KoichiYasuoka/bert-base-russian-upos")
example_ru = "У меня есть красивая собака."

# https://github.com/KoichiYasuoka/esupar
model_ko = esupar.load("ko")
example_ko = "예쁜 개가 있어요."

pos_results_es = model_es(example_es)
pos_results_ru = model_ru(example_ru)
pos_results_ko = model_ko(example_ko)

pprint(pos_results_es)
pprint(pos_results_ru)
pprint(pos_results_ko)