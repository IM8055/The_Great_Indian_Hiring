from skopt import space

PARAMSPACE = {
    'RandomForestRegressor': [space.Integer(200, 2000, name='n_estimators'),
                              space.Integer(3, 50, name='max_depth'),
                              space.Integer(2, 10, name='min_samples_split'),
                              space.Integer(1, 15, name='min_samples_leaf'),
                              space.Real(.01, 1, prior='uniform', name='max_features')
                              ],
    'ElasticNet': [space.Real(0.0, 1, prior='uniform', name='alpha'),
                   space.Real(0.0, 1, prior='uniform', name='l1_ratio'),
                   space.Integer(1000, 3000, name='max_iter'),
                   space.Categorical(['True', 'False'], name='normalize')
                   ],
    'GradientBoostingRegressor': [space.Real(0.1, 1, prior='uniform', name='learning_rate'),
                                  space.Integer(100, 2000, name='n_estimators'),
                                  space.Real(0.0, 1, prior='uniform', name='subsample'),
                                  space.Real(0.1, 1, prior='uniform', name='min_samples_split'),
                                  space.Real(0, 0.5, prior='uniform', name='min_samples_leaf'),
                                  space.Real(0.1, 0.5, prior='uniform', name='min_weight_fraction_leaf'),
                                  space.Integer(3, 1000, name='max_depth'),
                                  space.Real(0.0, 1, prior='uniform', name='max_features'),
                                  ],
    'HuberRegressor': [space.Real(1.1, 100, prior='uniform', name='epsilon'),
                       space.Integer(100, 2000, name='max_iter'),
                       space.Real(0.0001, 100, prior='uniform', name='alpha')],
    'KNeighborsRegressor': [space.Integer(5, 50, name='n_neighbors'),
                            space.Categorical(['uniform', 'distance'], name='weights'),
                            space.Integer(30, 100, name='leaf_size'),
                            space.Integer(1, 2, name='p')]

}

PARAMNAMES = {
    'RandomForestRegressor': ['n_estimators',
                              'max_depth',
                              'min_samples_split',
                              'min_samples_leaf',
                              'max_features'
                              ],
    'ElasticNet': ['alpha',
                   'l1_ratio',
                   'max_iter',
                   'normalize'
                   ],
    'GradientBoostingRegressor': ['learning_rate',
                                  'n_estimators',
                                  'subsample',
                                  'min_samples_split',
                                  'min_samples_leaf',
                                  'min_weight_fraction_leaf',
                                  'max_depth',
                                  'max_features'],
    'HuberRegressor': ['epsilon',
                       'max_iter',
                       'alpha'],
    'KNeighborsRegressor': ['n_neighbors',
                            'weights',
                            'leaf_size',
                            'p']
}
