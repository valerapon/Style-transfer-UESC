Руководство по запуску программы.
------------------------------------------------------------------------------------------

Главный файл запуска: test.py
Конструкция запуска:
        python ./test.py --content <путь к контенту> --style <список путей к стилям>
Пример:
        python ./test.py --content ./Images/c1.jpg --style ./Images/s1.jpg ./Images/s2.jpg
Результат работы - набор изображений в папке <<output>>, а именно найденный похожий стиль
и конечная стилизация под него.
Вывод программы содержит вспомогательную информацию о статусе завершенности программы:
        LOAD DATA: OK
        LOAD VGG: OK
        1: ./Images/s1.jpg:
                class: Abstract_Expressionism
                similar image: aaron-siskind_the-tree-35-1973.jpg
                stylization: ./output/0.jpg
                status: OK
        ...

------------------------------------------------------------------------------------------

Создание массива признаков: create_database.py
Конструкция запуска:
        python ./create_database.py
Результат работы - файл <<database_HOG_LBP_VGG_CONV.npy>> в папке <<output>>,
который прдеставляет собой признаковое описание изображений,
находящихся в директории <<train>>.
Вывод программы содержит информацию о процессе обработки изображений:
        Load VGG:OK
        Start:
        0/27 Abstract_Expressionism: 0/100; time: 0
        0/27 Abstract_Expressionism: 20/100; time: 6
        0/27 Abstract_Expressionism: 40/100; time: 11
        0/27 Abstract_Expressionism: 60/100; time: 16
        0/27 Abstract_Expressionism: 80/100; time: 21
        time: 26
        1/27 Action_painting: 0/98; time: 0
        1/27 Action_painting: 20/98; time: 4
        1/27 Action_painting: 40/98; time: 9
        1/27 Action_painting: 60/98; time: 14
        1/27 Action_painting: 80/98; time: 19
        time: 24
        ...

------------------------------------------------------------------------------------------

Обучение нейронной сети: train.py
Конструкция запуска:
        python ./train.py
Результат работы - обученная нейронная сеть <<model_HOG_LBP_VGG.pt>> в папке <<output>>.
Вывод программа содержит информацию о процессе обучения нейронной сети:
        Load data:OK
        Create models:OK
        Start train HOG_LBP_VGG model
        Train epoch: 0, {0}, Loss: 3.3780453205108643
        Train epoch: 1, {0}, Loss: 4.551957130432129
        Train epoch: 2, {0}, Loss: 3.5879030227661133
        Train epoch: 3, {0}, Loss: 2.9638638496398926
        Train epoch: 4, {0}, Loss: 2.86600232124
        ...

------------------------------------------------------------------------------------------