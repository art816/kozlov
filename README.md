# kozlov
## Поиск вхождения эталона в иследуемое изображения по методу кодовых расстояний.
Работаем с изображениями в градациях серого. 
Размер эталона исследуемого изображения, крайне желательно меньше 20 точек.
Пока запуск через тесты. test_find_by_4code

###Последовательность вызова функций.
```
#Формируем путь к фаилу в текущей директории
image_path = os.path.join( os.path.dirname( os.path.abspath(__file__)), cfg.image_names[0])
#Получаем массив символов соответствующий изображения, уже произведена бинаризация
pix_array = app_code.open_image(image_path)
#Получаем коды трехточечников(площади) и массив индексов "черных" точек на изображении
code3, index_of1 = app_code.get_code3(pix_array)
#Получаем код четырех точечников
code4 = app_code.get_code4(code3)
#Добавляем указатели на следующий четырехточечник для рассмотрения
app_code.add_key_for_next_small4(code4)
#Получаем массив пар соответствующих точек
pairs_equal_code = app_code.find_by_code4(target_code4, example_code4)
#Нужно получить координаты точек на изображении из pairs_equal_code, смотрим test_find_by_4code
#Строим матрицу преобразования A методом наименьших квадратов и получаем среднеквадратичное отклонение
A, std = app_code.find_transform(target, example)
#Дальше накладываем эталон на цель, смотрим test_find_by_4code
```

###Поиск соотвествия между точками
 | Эталон | Экземпляр 
-- | ------ | ---------
Код | (S_1 S_2 S_3 S_4) | (SS_1 SS_2 SS_3 SS_4)
S_* площадь треугольника (так же мы знаем индексы точек треугольника (i, j, k), об этом ниже)
Для эталона будут использованы одинарные буквы, для экземпляра двойные (S_1, SS_1) 

Cравнение (S_1 S_2 S_3 S_4)/S_1  (SS_1 SS_2 SS_3 SS_4)/SS_1

У меня площади в коде упорядочены
S_1 >= S_2 >= S_3 >= S_4
Очень плохо если S_1 = S_2 = S_3 = S_4, точки сопостовляются рандомно

Если (S_1 S_2 S_3 S_4)/S_1  примерно равно (SS_1 SS_2 SS_3 SS_4)/SS_1
Получаем соответствие S_1==SS_1 S_2==SS_2 S_3==SS_3 S_4==SS_4

Для всех S_i мы знаем индексы точек из которых они составлен (i, j, k, l)
Для каждой точки строим вектор V_* (0, 1, 0, 1), где 1 в n-ой позиции означает,
что * присутствует в S_n
V_i (0, 1, 1, 1) --> точка i присутствует в треугольниках с площадями S_2, S_3, S_4
VV_ii (1, 0, 0, 0) --> точка ii присутствует в треугольниках с площадями SS_1

Далее, по равенству векторов V_* VV_* Устанавливаем связь между точками.
V_i (0, 1, 1, 1) == VV_kk (0, 1, 1, 1) значит точка i соответствует kk