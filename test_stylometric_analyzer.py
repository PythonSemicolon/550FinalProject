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

test_text_ru_1 = """
Скажите, мистер Дезерт, а вы-то находите что-нибудь реальное в нынешней политике?

А разве для нас на свете есть что-нибудь реальное, сэр?

Да, подоходный налог.

Майкл засмеялся.

Кроме дворянского звания, нет ничего лучше простодушной веры.

Предположим, твои друзья придут к власти, Майкл; отчасти это неплохо, они бы выросли немного, а? Но что они смогли бы сделать? Могут ли они воспитать вкус народа? Уничтожить кино? Научить англичан хорошо готовить? Предотвратить угрозу войны со стороны других стран? Заставить нас самих растить свой хлеб? Остановить рост городов? Разве они перевешают изобретателей ядовитых газов? Разве они могут запретить самолетам летать во время войны? Разве они могут ослабить собственнические инстинкты где бы то ни было? Разве они вообще могут что-нибудь сделать, кроме как переменить немного распределение собственности? Политика всякой партии — это только глазурь на торте. Нами управляют изобретатели и человеческая природа; и мы сейчас в тупике, мистер Дезерт.
"""

#엽
test_text_ko_1 = """
아이들은 누구나 자라게 마련이다. 비록 예외도 하나 있긴 하지만. 아이들은 머지않아 자기가 자라게 된다는 것을 아는데, 웬디는 다음과 같은 방식으로 이 사실을 알게 되었다. 그녀가 두 살일 때에 하루는 정원에서 놀다가, 꽃을 또 한 송이 꺾어서 들고 어머니에게 달려간 적이 있었다. 아마도 웬디는 십중팔구 기뻐하는 모습이었나 보다. 왜냐하면 달링 부인이 한 손을 자기 가슴에 얹으면서 이렇게 탄식했기 때문이다. “아, 너는 왜 이런 상태로 영원히 남아 있을 수는 없는 거니!” 이 주제에 관해서 두 사람 사이에 오간 말은 이게 전부였지만, 그때 이후로 웬디는 자기가 반드시 자라게 된다는 것을 알았다. 여러분도 나이 두 살이 지난 뒤에는 틀림없이 알고 말 것이다. 두 살이야말로 종말의 시작이다.

물론 그들은 14번지에 살았고, 웬디가 태어나기 전까지는 그녀의 어머니가 이 집에서 제일 중요한 사람이었다. 달링 부인은 사랑스러운 숙녀였으며, 낭만적인 정신을 가졌고, 매우 귀고 조롱하는 듯한 입을 지니고 있었다. 그녀의 낭만적인 정신은 마치 저 수수께끼 같은 동양에서 건너온 작은 상자들처럼, 하나 안에 또 하나가 들어 있는 식이었다. 이럴 경우 여러분이 아무리 많은 상를 발견하더라도, 거기에는 항상 또 하나의 상자가 남아 있게 마련이다. 그녀의 귀엽고 조롱하는 듯한 입에는 키스가 하나 달려 있었는데, 이 키스로 말하자면 웬디조차도 결코 얻을 수 없는 것이었다. 비록 분명히 거기, 오른쪽 입가에 완벽하고 뚜렷하게 나타나 있는데도 말이다.1)

달링 씨가 부인을 얻게 된 경위는 이러했다. 달링 부인이 소녀였을 때에 소년이었던 여러 신사들은 자기가 그녀를 사랑한다는 것을 동시에 깨달았고, 급기야 청혼을 하려고 모두 그녀의 집까지 뛰어갔다. 반면 달링 씨 혼자만큼은 마차를 잡아타서 모두를 간발의 차이로 따돌렸으며, 그리하여 부인을 얻게 된 것이었다. 그는 그녀의 다른 모든 것을 얻었지만 가장 깊은 곳에 있는 상자와 키스만큼은 예외였다. 그는 상자에 관해서는 전혀 알지 못했고, 키스를 얻으려는 노력은 머지않아 포기하고 말았다. 웬디가 생각하기에는 나폴레옹 정도는 되어야 그걸 얻을 수 있을 것 같았다. 하지만 내 머릿속에는 나폴레옹조차도 시도했다가 버럭 화를 내면서 문을 쾅 닫고 나가 버리는 모습이 떠오른다.
"""


stylometric_analyzer_1 = StylometricAnalyzer(test_text_es_1, 'es')
# print(stylometric_analyzer_1.sentences)
pprint(stylometric_analyzer_1.get_feature_vector())

stylometric_analyzer_2 = StylometricAnalyzer(test_text_ru_1, 'ru')
# pprint(stylometric_analyzer_2.lemmas_with_punc)
# pprint(stylometric_analyzer_2.lemmas)
# pprint(stylometric_analyzer_2.words)
# pprint(stylometric_analyzer_2.sentences)
pprint(stylometric_analyzer_2.get_feature_vector())

stylometric_analyzer_3 = StylometricAnalyzer(test_text_ko_1, 'ko')
# pprint(stylometric_analyzer_3.lemmas_with_punc)
# pprint(stylometric_analyzer_3.lemmas)
# pprint(stylometric_analyzer_3.words)
# pprint(stylometric_analyzer_3.sentences)
pprint(stylometric_analyzer_3.get_feature_vector())