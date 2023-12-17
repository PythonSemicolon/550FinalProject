from transformers import pipeline
from pprint import pprint
from stylometric_analyzer import StylometricAnalyzer

test_text_es_1 = """
Era el tono, no las palabras.
No se dirigía al doctor, estaba más allá de la risa, como si aquel a quien se dirigía fuera impermeable a la risa; no fue el doctor quien se detuvo; el doctor aún seguía trotando sobre sus cortas piernas sedentarias, tras la luz ajetreada de la linterna, hacia la indecisa luz que lo esperaba; era el bautista, el provinciano, que parecía detenerse mientras el hombre, no ya el médico, pensaba sin escándalo, pero en una especie de asombro desesperado: ¿Habré de vivir siempre tras una barricada de perenne inocencia como un pollo en la cáscara?
Habló cuidadosamente en voz alta; el velo se descorría disolviéndose, estaba a punto de partirse ahora y él no quería ver lo que había detrás; sabía que para eterna tranquilidad de su conciencia no se animaba, y sabía que era ya demasiado tarde y que no podía contenerse; oyó a su voz hacer la pregunta que no quería y recibir la respuesta que no quería: —¿Dice usted que está sangrando?
¿Por dónde?
—¿Por dónde sangran las mujeres?
—dijo el otro; gritó con una voz irritada y áspera sin detenerse—.
¡Yo no soy médico!
Si lo fuera, ¿cree que iba a gastar cinco dólares en usted?
"""

test_text_es_2 = "Mi nombre es José"

# stylometric_analyzer_1 = StylometricAnalyzer(test_text_es_1, 'es', test_text_es_pos_tags_1)
stylometric_analyzer_1 = StylometricAnalyzer(test_text_es_1, 'es')
print(stylometric_analyzer_1.sentences)
# print(stylometric_analyzer_1.words)
stylometric_analyzer_2 = StylometricAnalyzer(test_text_es_2, 'es')
pprint(stylometric_analyzer_1.get_feature_vector())
pprint(stylometric_analyzer_2.get_feature_vector())