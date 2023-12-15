from transformers import pipeline
from pprint import pprint
from stylometric_analyzer import StylometricAnalyzer

test_text_es = """
Era el tono, no las palabras. No se dirigía al doctor, estaba más allá de la risa,
como si aquel a quien se dirigía fuera impermeable a la risa; no fue el doctor quien
se detuvo; el doctor aún seguía trotando sobre sus cortas piernas sedentarias, tras
la luz ajetreada de la linterna, hacia la indecisa luz que lo esperaba; era el
bautista, el provinciano, que parecía detenerse mientras el hombre, no ya el
médico, pensaba sin escándalo, pero en una especie de asombro desesperado:
¿Habré de vivir siempre tras una barricada de perenne inocencia como un pollo en la
cáscara?
Habló cuidadosamente en voz alta; el velo se descorría disolviéndose, estaba a
punto de partirse ahora y él no quería ver lo que había detrás; sabía que para
eterna tranquilidad de su conciencia no se animaba, y sabía que era ya demasiado
tarde y que no podía contenerse; oyó a su voz hacer la pregunta que no quería y
recibir la respuesta que no quería:
—¿Dice usted que está sangrando? ¿Por dónde?
—¿Por dónde sangran las mujeres? —dijo el otro; gritó con una voz irritada y
áspera sin detenerse—. ¡Yo no soy médico! Si lo fuera, ¿cree que iba a gastar cinco
dólares en usted?
"""

test_text_es = "Mi nombre es José"
model_es = pipeline("token-classification", model="PlanTL-GOB-ES/roberta-large-bne-capitel-pos")
test_text_es_pos_tags = {
    'nouns': len([word for word in test_text_es.split() if model_es(word)[0]['entity'] == 'NOUN']),
    'verbs': len([word for word in test_text_es.split() if model_es(word)[0]['entity'] == 'VERB']),
    'adjectives': len([word for word in test_text_es.split() if model_es(word)[0]['entity'] == 'ADJ']),
    'adverbs': len([word for word in test_text_es.split() if model_es(word)[0]['entity'] == 'ADV']),
}

stylometric_analyzer = StylometricAnalyzer(test_text_es, 'es', test_text_es_pos_tags)
pprint(stylometric_analyzer.get_feature_vector())