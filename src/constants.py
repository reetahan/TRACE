
DISTRICT_TO_BOROUGH_MAPPING = {
        str(d): b for d, b in
        [(d,'M') for d in range(1,7)] +
        [(d,'X') for d in range(7,13)] +
        [(d,'K') for d in range(13,24)] +
        [(d,'Q') for d in range(24,31)] +
        [(31,'R'), (32,'K')]
    }

UTILITY_PENALTY = -0.1
LEARNING_RATE = 0.1

# pct_top_{k} columns are handled separately in _to_em_dataframes().
EM_COLUMN_MAP: dict[str, str] = {
    'subdivision':       'Residential District',
    'school_id':         'School DBN',
    'n_applicants':      'Total Applicants by Residential District',
    'n_true_applicants': 'True Applicants by Residential District',
    'ratio':             'Ratio',
    'rank':              'Rank',
    'n_students':        'Total Applicants',
    'capacity':          'Capacity',
    'utilization':       'Utilization',
    'pct_unmatched':     'Unmatched',
}

CHILE_PROVINCE_TO_REGION_MAPPING = {
    'Arica': 'Región de Arica y Parinacota', 'Parinacota': 'Región de Arica y Parinacota', 'Iquique': 'Región de Tarapacá', 
    'Tamarugal': 'Región de Tarapacá', 'Tocopilla': 'Región de Antofagasta', 'El Loa': 'Región de Antofagasta', 
    'Antofagasta': 'Región de Antofagasta', 'Chañaral': 'Región de Atacama', 'Copiapó': 'Región de Atacama', 'Huasco': 
    'Región de Atacama', 'Elqui': 'Región de Coquimbo', 'Limarí': 'Región de Coquimbo', 'Choapa': 'Región de Coquimbo', 
    'Petorca': 'Región de Valparaíso', 'Los Andes': 'Región de Valparaíso', 'San Felipe': 'Región de Valparaíso', 
    'Quillota': 'Región de Valparaíso', 'Marga Marga': 'Región de Valparaíso', 'Valparaíso': 'Región de Valparaíso', 
    'San Antonio': 'Región de Valparaíso', 'Cachapoal': "Región del Libertador Bernardo O'Higgins", 
    'Colchagua': "Región del Libertador Bernardo O'Higgins", 'Cardenal Caro': "Región del Libertador Bernardo O'Higgins", 
    'Curicó': 'Región del Maule', 'Talca': 'Región del Maule', 'Linares': 'Región del Maule', 'Cauquenes': 'Región del Maule',
    'Diguillín': 'Región de Ñuble', 'Punilla': 'Región de Ñuble', 'Itata': 'Región de Ñuble', 'Bío-Bío': 'Región del Bío-Bío',
    'Concepción': 'Región del Bío-Bío', 'Arauco': 'Región del Bío-Bío', 'Malleco': 'Región de La Araucanía', 
    'Cautín': 'Región de La Araucanía', 'Valdivia': 'Región de Los Ríos', 'Ranco': 'Región de Los Ríos', 
    'Osorno': 'Región de Los Lagos', 'Llanquihue': 'Región de Los Lagos', 'Chiloe': 'Región de Los Lagos', 
    'Palena': 'Región de Los Lagos', 'Coyhaique': 'Región de Aysén del Gral.Ibañez del Campo', 
    'Aysén': 'Región de Aysén del Gral.Ibañez del Campo', 'General Carrera': 'Región de Aysén del Gral.Ibañez del Campo',
    'Capitán Prat': 'Región de Aysén del Gral.Ibañez del Campo', 'Ultima Esperanza': 'Región de Magallanes y Antártica Chilena',
    'Magallanes': 'Región de Magallanes y Antártica Chilena', 'Tierra del Fuego': 'Región de Magallanes y Antártica Chilena', 
    'Antártica Chilena': 'Región de Magallanes y Antártica Chilena', 'Santiago': 'Región Metropolitana de Santiago', 
    'Cordillera': 'Región Metropolitana de Santiago', 'Chacabuco': 'Región Metropolitana de Santiago', 'Maipo': 'Región Metropolitana de Santiago', 
    'Talagante': 'Región Metropolitana de Santiago', 'Melipilla': 'Región Metropolitana de Santiago', 
    'Isla de Pascua': 'Región de Valparaíso'
}