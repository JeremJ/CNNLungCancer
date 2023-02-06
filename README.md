# CNNLungCancer

1. Należy zainstalować wymagane bilbioteki z pliku requirements.txt ('pip install') - wersja pythona 3.9
2. W pliku preprocessing.py - znajdują się algorytmy segmentacji, należy wywołać jedną z metod full_segmentation() lub semi_segmentation() - instrukcje znajdują się również w kodzie.
3. model.py - model oparty o sieć VGG-16 - wymagane jest podanie odpowiednich ścieżek do folderów z danymi, analogicznie do tego co widnieje w kodzie, dane zawarte są w repozytorium w folderze 'dane'.
4. own_cnn_implementation.py - autorski model sieci konwolucyjnej - wymagane jest podanie odpowiednich ścieżek do folderów z danymi, analogicznie do tego co widnieje w kodzie, dane zawarte są w repozytorium w folderze 'dane'.


- w folderze Eksperymenty widoczne są przebiegi uczenia które również zostały zamieszczone w pracy.
- w folderze Filmiki znajdują się:
  - przykładowe uruchomienie procesu uczenia autorskiego modelu sieci, w skróconym wariancie czyli przez 10 epoch.
  - przykładowe uruchomienie segmentacji danych wejściowych. Dane w formacie DICOM znajdują się również w repozytorium - '\Dane\dicom'. katalog"chore" zawiera dane na których są obrazy nowotworów płuc, folder "zdrowe" zawiera obrazy z badań pacjentów u których nowotworu złośliwego płuc nie stwierdzono.