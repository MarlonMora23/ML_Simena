import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
import numpy as np
import category_encoders as ce
import warnings

def procesarDatos(data, encoder=None):
    if encoder is None:
        encoder = ce.CountEncoder()
    
    data['ESTU_TIPODOCUMENTO'] = encoder.fit_transform(data['ESTU_TIPODOCUMENTO'])
    data['ESTU_NACIONALIDAD'] = encoder.fit_transform(data['ESTU_NACIONALIDAD'])
    data['ESTU_GENERO'] = data['ESTU_GENERO'].map({'M': 0, 'F':1})
    data['ESTU_FECHANACIMIENTO'] = encoder.fit_transform(data['ESTU_FECHANACIMIENTO'])
    data['PERIODO'] = encoder.fit_transform(data['PERIODO'])
    data['ESTU_CONSECUTIVO'] = encoder.fit_transform(data['ESTU_CONSECUTIVO'])
    data['ESTU_ESTUDIANTE'] = encoder.fit_transform(data['ESTU_ESTUDIANTE'])
    data['ESTU_TIENEETNIA'] = data['ESTU_TIENEETNIA'].map({'No':0, 'Si':1})
    data['ESTU_PAIS_RESIDE'] = encoder.fit_transform(data['ESTU_PAIS_RESIDE'])
    data['ESTU_ETNIA'] = encoder.fit_transform(data['ESTU_ETNIA'])
    data['ESTU_DEPTO_RESIDE'] = encoder.fit_transform(data['ESTU_DEPTO_RESIDE'])
    data['ESTU_MCPIO_RESIDE'] = encoder.fit_transform(data['ESTU_MCPIO_RESIDE'])
    data['FAMI_ESTRATOVIVIENDA'] = data['FAMI_ESTRATOVIVIENDA'].map({'Estrato 1': 1, 'Estrato 2': 2, 'Estrato 3': 3 , 'Estrato 4': 4 , 'Estrato 5': 5 , 'Estrato 6': 6})
    data['FAMI_PERSONASHOGAR'] = encoder.fit_transform(data['FAMI_PERSONASHOGAR'])
    data['FAMI_CUARTOSHOGAR'] = data['FAMI_CUARTOSHOGAR'].map({'Uno':1, 'Dos':2, 'Tres':3, 'Cuatro':4, 'Cinco':5, 'Seis':6, 'Siete':7})
    data['FAMI_EDUCACIONPADRE'] = encoder.fit_transform(data['FAMI_EDUCACIONPADRE'])
    data['FAMI_EDUCACIONMADRE'] = encoder.fit_transform(data['FAMI_EDUCACIONMADRE'])
    data['FAMI_TRABAJOLABORPADRE'] = encoder.fit_transform(data['FAMI_TRABAJOLABORPADRE'])
    data['FAMI_TRABAJOLABORMADRE'] = encoder.fit_transform(data['FAMI_TRABAJOLABORMADRE'])
    data['FAMI_TIENEINTERNET'] = data['FAMI_TIENEINTERNET'].map({'No': 0, 'Si': 1})
    data['FAMI_TIENESERVICIOTV'] = data['FAMI_TIENESERVICIOTV'].map({'No': 0, 'Si': 1})
    data['FAMI_TIENECOMPUTADOR'] = data['FAMI_TIENECOMPUTADOR'].map({'No': 0, 'Si': 1})
    data['FAMI_TIENELAVADORA'] = data['FAMI_TIENELAVADORA'].map({'No': 0, 'Si': 1})
    data['FAMI_TIENEHORNOMICROOGAS'] = data['FAMI_TIENEHORNOMICROOGAS'].map({'No': 0, 'Si': 1})
    data['FAMI_TIENEAUTOMOVIL'] = data['FAMI_TIENEAUTOMOVIL'].map({'No': 0, 'Si': 1})
    data['FAMI_TIENEMOTOCICLETA'] = data['FAMI_TIENEMOTOCICLETA'].map({'No': 0, 'Si': 1})
    data['FAMI_TIENECONSOLAVIDEOJUEGOS'] = data['FAMI_TIENECONSOLAVIDEOJUEGOS'].map({'No': 0, 'Si': 1})
    data['FAMI_NUMLIBROS'] = encoder.fit_transform(data['FAMI_NUMLIBROS'])
    data['FAMI_COMELECHEDERIVADOS'] = encoder.fit_transform(data['FAMI_COMELECHEDERIVADOS'])
    data['FAMI_COMECARNEPESCADOHUEVO'] = encoder.fit_transform(data['FAMI_COMECARNEPESCADOHUEVO'])
    data['FAMI_COMECEREALFRUTOSLEGUMBRE'] = encoder.fit_transform(data['FAMI_COMECEREALFRUTOSLEGUMBRE'])
    data['FAMI_SITUACIONECONOMICA'] = encoder.fit_transform(data['FAMI_SITUACIONECONOMICA'])
    data['ESTU_DEDICACIONLECTURADIARIA'] = encoder.fit_transform(data['ESTU_DEDICACIONLECTURADIARIA'])
    data['ESTU_DEDICACIONINTERNET'] = encoder.fit_transform(data['ESTU_DEDICACIONINTERNET'])
    data['ESTU_HORASSEMANATRABAJA'] = encoder.fit_transform(data['ESTU_HORASSEMANATRABAJA'])
    data['ESTU_TIPOREMUNERACION'] = encoder.fit_transform(data['ESTU_TIPOREMUNERACION'])
    data['COLE_COD_DANE_ESTABLECIMIENTO'] = encoder.fit_transform(data['COLE_COD_DANE_ESTABLECIMIENTO'])
    data['COLE_NOMBRE_ESTABLECIMIENTO'] = encoder.fit_transform(data['COLE_NOMBRE_ESTABLECIMIENTO'])
    data['COLE_GENERO'] = data['COLE_GENERO'].map({'MASCULINO': 0, 'FEMENINO': 1, 'MIXTO':2})
    data['COLE_NATURALEZA'] = data['COLE_NATURALEZA'].map({'NO OFICIAL':0, 'OFICIAL':1})
    data['COLE_CALENDARIO'] = data['COLE_CALENDARIO'].map({'A':0, 'B':1, 'OTRO':2})
    data['COLE_BILINGUE'] = data['COLE_BILINGUE'].map({'N': 0, 'S': 1})
    data['COLE_CARACTER'] = encoder.fit_transform(data['COLE_CARACTER'])
    data['COLE_COD_DANE_SEDE'] = encoder.fit_transform(data['COLE_COD_DANE_SEDE'])
    data['COLE_NOMBRE_SEDE'] = encoder.fit_transform(data['COLE_NOMBRE_SEDE'])
    data['COLE_SEDE_PRINCIPAL'] = data['COLE_SEDE_PRINCIPAL'].map({'S': 0, 'N': 1})
    data['COLE_AREA_UBICACION'] = data['COLE_AREA_UBICACION'].map({'RURAL': 0, 'URBANO': 1})
    data['COLE_JORNADA'] = encoder.fit_transform(data['COLE_JORNADA'])
    data['COLE_MCPIO_UBICACION'] = encoder.fit_transform(data['COLE_MCPIO_UBICACION'])
    data['COLE_DEPTO_UBICACION'] = encoder.fit_transform(data['COLE_DEPTO_UBICACION'])
    data['ESTU_PRIVADO_LIBERTAD'] = data['ESTU_PRIVADO_LIBERTAD'].map({'N': 0, 'S': 1})
    data['ESTU_MCPIO_PRESENTACION'] = encoder.fit_transform(data['ESTU_MCPIO_PRESENTACION'])
    data['ESTU_DEPTO_PRESENTACION'] = encoder.fit_transform(data['ESTU_DEPTO_PRESENTACION'])
    data['ESTU_INSE_INDIVIDUAL'] = encoder.fit_transform(data['ESTU_INSE_INDIVIDUAL'])
    data['ESTU_ESTADOINVESTIGACION'] = encoder.fit_transform(data['ESTU_ESTADOINVESTIGACION'])
    data['ESTU_GENERACION-E'] = encoder.fit_transform(data['ESTU_GENERACION-E'])

    # data['DESEMP_INGLES'] = data['DESEMP_INGLES'].map({'A-':0, 'A1':1, 'A2': 3, 'B1':4, 'B+':5})

    #Llenar datos nulos
    data['FAMI_ESTRATOVIVIENDA'].fillna(1, inplace=True)
    data['FAMI_TIENEINTERNET'].fillna(0, inplace=True)
    data['FAMI_TIENESERVICIOTV'].fillna(0, inplace=True)
    data['FAMI_TIENECOMPUTADOR'].fillna(0, inplace=True)
    data['FAMI_TIENELAVADORA'].fillna(0, inplace=True)
    data['FAMI_TIENEHORNOMICROOGAS'].fillna(0, inplace=True)
    data['FAMI_TIENEAUTOMOVIL'].fillna(0, inplace=True)
    data['FAMI_TIENEMOTOCICLETA'].fillna(0, inplace=True)
    data['FAMI_TIENECONSOLAVIDEOJUEGOS'].fillna(0, inplace=True)
    data['COLE_BILINGUE'].fillna(0, inplace=True)
    data['ESTU_GENERO'].fillna(2, inplace=True)
    data['ESTU_TIENEETNIA'].fillna(0, inplace=True)
    data['COLE_GENERO'].fillna(2, inplace=True)
    data['COLE_NATURALEZA'].fillna(0, inplace=True)
    data['COLE_AREA_UBICACION'].fillna(0, inplace=True)
    data['ESTU_PRIVADO_LIBERTAD'].fillna(0, inplace=True)

    #reemplazar guiones por datos nulos
    data['FAMI_CUARTOSHOGAR'] = data['FAMI_CUARTOSHOGAR'].replace('-', np.nan)
    data['ESTU_COD_RESIDE_MCPIO'] = data['ESTU_COD_RESIDE_MCPIO'].replace('-', np.nan)
    data['ESTU_COD_RESIDE_DEPTO'] = data['ESTU_COD_RESIDE_DEPTO'].replace('-', np.nan)
    data['COLE_CODIGO_ICFES'] = data['COLE_CODIGO_ICFES'].replace('-', np.nan)
    data['COLE_COD_MCPIO_UBICACION'] = data['COLE_COD_MCPIO_UBICACION'].replace('-', np.nan)
    data['COLE_COD_DEPTO_UBICACION'] = data['COLE_COD_DEPTO_UBICACION'].replace('-', np.nan)
    data['ESTU_COD_MCPIO_PRESENTACION'] = data['ESTU_COD_MCPIO_PRESENTACION'].replace('-', np.nan)
    data['ESTU_COD_DEPTO_PRESENTACION'] = data['ESTU_COD_DEPTO_PRESENTACION'].replace('-', np.nan)
    data['ESTU_NSE_INDIVIDUAL'] = data['ESTU_NSE_INDIVIDUAL'].replace('-', np.nan)
    data['COLE_SEDE_PRINCIPAL'] = data['COLE_SEDE_PRINCIPAL'].replace('-', np.nan)
    data['ESTU_NSE_ESTABLECIMIENTO'] = data['ESTU_NSE_ESTABLECIMIENTO'].replace('-', np.nan)

    #Rellenar valores nulos con la mediana
    imputer = SimpleImputer(strategy='median')

    #rellenar valores nulos
    data['ESTU_COD_RESIDE_MCPIO'] = imputer.fit_transform(data[['ESTU_COD_RESIDE_MCPIO']])
    data['ESTU_COD_RESIDE_DEPTO'] = imputer.fit_transform(data[['ESTU_COD_RESIDE_DEPTO']])
    data['FAMI_CUARTOSHOGAR'] = imputer.fit_transform(data[['FAMI_CUARTOSHOGAR']])
    data['COLE_CODIGO_ICFES'] = imputer.fit_transform(data[['COLE_CODIGO_ICFES']])
    data['COLE_SEDE_PRINCIPAL'] = imputer.fit_transform(data[['COLE_SEDE_PRINCIPAL']])
    data['COLE_COD_MCPIO_UBICACION'] = imputer.fit_transform(data[['COLE_COD_MCPIO_UBICACION']])
    data['COLE_COD_DEPTO_UBICACION'] = imputer.fit_transform(data[['COLE_COD_DEPTO_UBICACION']])
    data['ESTU_COD_MCPIO_PRESENTACION'] = imputer.fit_transform(data[['ESTU_COD_MCPIO_PRESENTACION']])
    data['ESTU_COD_DEPTO_PRESENTACION'] = imputer.fit_transform(data[['ESTU_COD_DEPTO_PRESENTACION']])
    data['ESTU_NSE_INDIVIDUAL'] = imputer.fit_transform(data[['ESTU_NSE_INDIVIDUAL']])
    data['ESTU_NSE_ESTABLECIMIENTO'] = imputer.fit_transform(data[['ESTU_NSE_ESTABLECIMIENTO']])

    return encoder, data

def evaluarModelo(data, encoder=None):
    if encoder is None:
        encoder, data = procesarDatos(data)

    # Variables independientes (características)
    x = data[['ESTU_TIPODOCUMENTO', 'ESTU_NACIONALIDAD', 'ESTU_GENERO', 'ESTU_FECHANACIMIENTO', 'PERIODO', 'ESTU_CONSECUTIVO',
            'ESTU_ESTUDIANTE', 'ESTU_TIENEETNIA', 'ESTU_PAIS_RESIDE', 'ESTU_ETNIA', 'ESTU_DEPTO_RESIDE', 'ESTU_COD_RESIDE_DEPTO',
            'ESTU_MCPIO_RESIDE','ESTU_COD_RESIDE_MCPIO', 'FAMI_ESTRATOVIVIENDA', 'FAMI_PERSONASHOGAR', 'FAMI_CUARTOSHOGAR',
            'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE', 'FAMI_TRABAJOLABORPADRE', 'FAMI_TRABAJOLABORMADRE','FAMI_TIENEINTERNET',
            'FAMI_TIENESERVICIOTV', 'FAMI_TIENECOMPUTADOR', 'FAMI_TIENELAVADORA', 'FAMI_TIENEHORNOMICROOGAS', 'FAMI_TIENEAUTOMOVIL',
            'FAMI_TIENEMOTOCICLETA', 'FAMI_TIENECONSOLAVIDEOJUEGOS', 'FAMI_NUMLIBROS', 'FAMI_COMELECHEDERIVADOS',
            'FAMI_COMECARNEPESCADOHUEVO', 'FAMI_COMECEREALFRUTOSLEGUMBRE', 'FAMI_SITUACIONECONOMICA', 'ESTU_DEDICACIONLECTURADIARIA',
            'ESTU_DEDICACIONINTERNET', 'ESTU_HORASSEMANATRABAJA', 'ESTU_TIPOREMUNERACION', 'COLE_CODIGO_ICFES',
            'COLE_COD_DANE_ESTABLECIMIENTO', 'COLE_NOMBRE_ESTABLECIMIENTO', 'COLE_GENERO', 'COLE_CALENDARIO', 'COLE_NATURALEZA',
            'COLE_BILINGUE', 'COLE_CARACTER', 'COLE_COD_DANE_SEDE', 'COLE_NOMBRE_SEDE', 'COLE_SEDE_PRINCIPAL', 'COLE_AREA_UBICACION',
            'COLE_JORNADA', 'COLE_COD_MCPIO_UBICACION', 'COLE_MCPIO_UBICACION', 'COLE_COD_DEPTO_UBICACION', 'COLE_DEPTO_UBICACION',
            'ESTU_PRIVADO_LIBERTAD', 'ESTU_COD_MCPIO_PRESENTACION', 'ESTU_MCPIO_PRESENTACION', 'ESTU_DEPTO_PRESENTACION', 
            'ESTU_COD_DEPTO_PRESENTACION', 'ESTU_INSE_INDIVIDUAL', 'ESTU_NSE_INDIVIDUAL', 'ESTU_NSE_ESTABLECIMIENTO',
            'ESTU_ESTADOINVESTIGACION', 'ESTU_GENERACION-E']]

    # Variable objetivo
    y = data['PUNT_GLOBAL']

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Crear y entrenar el modelo Ridge
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)

    # Realizar predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # Definir los intervalos deseados
    intervalos = [(i, i + tamanoIntervalo) for i in range(0, 500, tamanoIntervalo)]

    # Redondear las predicciones al intervalo más cercano
    y_pred_rounded = np.round(y_pred / tamanoIntervalo) * tamanoIntervalo

    # Evaluar el modelo con intervalos
    correctos = 0
    total = len(y_test)
    predicciones_por_intervalo = {intervalo: 0 for intervalo in intervalos}

    for pred, true in zip(y_pred_rounded, y_test):
        for intervalo in intervalos:
            if intervalo[0] <= true <= intervalo[1] and intervalo[0] <= pred <= intervalo[1]:
                correctos += 1
                predicciones_por_intervalo[intervalo] += 1
                break

    precision = correctos / total
    print(f'Alpha: {alpha}')
    print(f'Precisión en intervalos: {precision}')

    # Imprimir el número de predicciones en cada intervalo
    for intervalo, count in predicciones_por_intervalo.items():
        print(f'Intervalo {intervalo}: {count} predicciones')

    return encoder, model

def modeloPrediccion(data, model, encoder):
    # Leer los datos a predecir
    encoder, data = procesarDatos(data, encoder=encoder)

    # Variables independientes (características)
    x = data[['ESTU_TIPODOCUMENTO', 'ESTU_NACIONALIDAD', 'ESTU_GENERO', 'ESTU_FECHANACIMIENTO', 'PERIODO', 'ESTU_CONSECUTIVO',
            'ESTU_ESTUDIANTE', 'ESTU_TIENEETNIA', 'ESTU_PAIS_RESIDE', 'ESTU_ETNIA', 'ESTU_DEPTO_RESIDE', 'ESTU_COD_RESIDE_DEPTO',
            'ESTU_MCPIO_RESIDE','ESTU_COD_RESIDE_MCPIO', 'FAMI_ESTRATOVIVIENDA', 'FAMI_PERSONASHOGAR', 'FAMI_CUARTOSHOGAR',
            'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE', 'FAMI_TRABAJOLABORPADRE', 'FAMI_TRABAJOLABORMADRE','FAMI_TIENEINTERNET',
            'FAMI_TIENESERVICIOTV', 'FAMI_TIENECOMPUTADOR', 'FAMI_TIENELAVADORA', 'FAMI_TIENEHORNOMICROOGAS', 'FAMI_TIENEAUTOMOVIL',
            'FAMI_TIENEMOTOCICLETA', 'FAMI_TIENECONSOLAVIDEOJUEGOS', 'FAMI_NUMLIBROS', 'FAMI_COMELECHEDERIVADOS',
            'FAMI_COMECARNEPESCADOHUEVO', 'FAMI_COMECEREALFRUTOSLEGUMBRE', 'FAMI_SITUACIONECONOMICA', 'ESTU_DEDICACIONLECTURADIARIA',
            'ESTU_DEDICACIONINTERNET', 'ESTU_HORASSEMANATRABAJA', 'ESTU_TIPOREMUNERACION', 'COLE_CODIGO_ICFES',
            'COLE_COD_DANE_ESTABLECIMIENTO', 'COLE_NOMBRE_ESTABLECIMIENTO', 'COLE_GENERO', 'COLE_CALENDARIO', 'COLE_NATURALEZA',
            'COLE_BILINGUE', 'COLE_CARACTER', 'COLE_COD_DANE_SEDE', 'COLE_NOMBRE_SEDE', 'COLE_SEDE_PRINCIPAL', 'COLE_AREA_UBICACION',
            'COLE_JORNADA', 'COLE_COD_MCPIO_UBICACION', 'COLE_MCPIO_UBICACION', 'COLE_COD_DEPTO_UBICACION', 'COLE_DEPTO_UBICACION',
            'ESTU_PRIVADO_LIBERTAD', 'ESTU_COD_MCPIO_PRESENTACION', 'ESTU_MCPIO_PRESENTACION', 'ESTU_DEPTO_PRESENTACION', 
            'ESTU_COD_DEPTO_PRESENTACION', 'ESTU_INSE_INDIVIDUAL', 'ESTU_NSE_INDIVIDUAL', 'ESTU_NSE_ESTABLECIMIENTO',
            'ESTU_ESTADOINVESTIGACION', 'ESTU_GENERACION-E']]

    # Realizar predicciones con el modelo entrenado
    predicciones = model.predict(x)

    # Redondear las predicciones al intervalo más cercano
    predicciones_rounded = np.round(predicciones / tamanoIntervalo) * tamanoIntervalo

    # Definir los intervalos deseados
    intervalos = [(i, i + tamanoIntervalo) for i in range(0, 500, tamanoIntervalo)]

    # Realizar predicciones con el modelo entrenado
    predicciones = model.predict(x)

    # Redondear las predicciones al intervalo más cercano
    predicciones_rounded = np.round(predicciones / tamanoIntervalo) * tamanoIntervalo

    # Imprimir el intervalo al que pertenece cada predicción
    count = 1
    for pred in predicciones_rounded:
        for intervalo in intervalos:
            if intervalo[0] <= pred <= intervalo[1]:
                intervalo_str = f'Intervalo ({intervalo[0]}, {intervalo[1]})'
                print(f'Predicción: {count} está en {intervalo_str}')
                count+=1
                break

warnings.filterwarnings("ignore", "is_categorical_dtype")
warnings.filterwarnings("ignore", "use_inf_as_na")

alpha = 0.01
tamanoIntervalo = 50

# leer los datos a entrenar y predecir
datos_entrenamiento = pd.read_csv('Data/datos.csv', delimiter=';', low_memory=False)
datos_prediccion = pd.read_csv('predicciones/PersonasPredecir.csv', delimiter=';')

# evaluar el modelo
encoder_entrenamiento, modelo_entrenado = evaluarModelo(datos_entrenamiento)
# predecir usando el modelo entrenado
modeloPrediccion(datos_prediccion, modelo_entrenado, encoder_entrenamiento)