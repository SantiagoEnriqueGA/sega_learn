# Table of Contents

* [\_\_init\_\_](#__init__)
  * [\_\_all\_\_](#__init__.__all__)
* [auto](#auto)
  * [\_\_all\_\_](#auto.__all__)
* [auto.classifier](#auto.classifier)
  * [accuracy](#auto.classifier.accuracy)
  * [precision](#auto.classifier.precision)
  * [recall](#auto.classifier.recall)
  * [f1](#auto.classifier.f1)
  * [AutoClassifier](#auto.classifier.AutoClassifier)
    * [\_\_init\_\_](#auto.classifier.AutoClassifier.__init__)
    * [fit](#auto.classifier.AutoClassifier.fit)
    * [predict](#auto.classifier.AutoClassifier.predict)
    * [evaluate](#auto.classifier.AutoClassifier.evaluate)
    * [get\_model](#auto.classifier.AutoClassifier.get_model)
    * [summary](#auto.classifier.AutoClassifier.summary)
* [auto.regressor](#auto.regressor)
  * [r\_squared](#auto.regressor.r_squared)
  * [root\_mean\_squared\_error](#auto.regressor.root_mean_squared_error)
  * [mean\_absolute\_percentage\_error](#auto.regressor.mean_absolute_percentage_error)
  * [AutoRegressor](#auto.regressor.AutoRegressor)
    * [\_\_init\_\_](#auto.regressor.AutoRegressor.__init__)
    * [fit](#auto.regressor.AutoRegressor.fit)
    * [predict](#auto.regressor.AutoRegressor.predict)
    * [evaluate](#auto.regressor.AutoRegressor.evaluate)
    * [get\_model](#auto.regressor.AutoRegressor.get_model)
    * [summary](#auto.regressor.AutoRegressor.summary)
* [clustering](#clustering)
  * [\_\_all\_\_](#clustering.__all__)
* [clustering.clustering](#clustering.clustering)
  * [KMeans](#clustering.clustering.KMeans)
    * [\_\_init\_\_](#clustering.clustering.KMeans.__init__)
    * [\_handle\_categorical](#clustering.clustering.KMeans._handle_categorical)
    * [\_convert\_to\_ndarray](#clustering.clustering.KMeans._convert_to_ndarray)
    * [initialize\_centroids](#clustering.clustering.KMeans.initialize_centroids)
    * [assign\_clusters](#clustering.clustering.KMeans.assign_clusters)
    * [update\_centroids](#clustering.clustering.KMeans.update_centroids)
    * [fit](#clustering.clustering.KMeans.fit)
    * [predict](#clustering.clustering.KMeans.predict)
    * [elbow\_method](#clustering.clustering.KMeans.elbow_method)
    * [calinski\_harabasz\_index](#clustering.clustering.KMeans.calinski_harabasz_index)
    * [davies\_bouldin\_index](#clustering.clustering.KMeans.davies_bouldin_index)
    * [silhouette\_score](#clustering.clustering.KMeans.silhouette_score)
    * [find\_optimal\_clusters](#clustering.clustering.KMeans.find_optimal_clusters)
  * [DBSCAN](#clustering.clustering.DBSCAN)
    * [\_\_init\_\_](#clustering.clustering.DBSCAN.__init__)
    * [\_handle\_categorical](#clustering.clustering.DBSCAN._handle_categorical)
    * [\_convert\_to\_ndarray](#clustering.clustering.DBSCAN._convert_to_ndarray)
    * [\_custom\_distance\_matrix](#clustering.clustering.DBSCAN._custom_distance_matrix)
    * [fit](#clustering.clustering.DBSCAN.fit)
    * [predict](#clustering.clustering.DBSCAN.predict)
    * [fit\_predict](#clustering.clustering.DBSCAN.fit_predict)
    * [silhouette\_score](#clustering.clustering.DBSCAN.silhouette_score)
    * [auto\_eps](#clustering.clustering.DBSCAN.auto_eps)
* [clustering.\_dbscan\_jit\_utils](#clustering._dbscan_jit_utils)
  * [\_identify\_core\_points](#clustering._dbscan_jit_utils._identify_core_points)
  * [\_assign\_clusters](#clustering._dbscan_jit_utils._assign_clusters)
* [linear\_models](#linear_models)
  * [\_\_all\_\_](#linear_models.__all__)
* [linear\_models.classifiers](#linear_models.classifiers)
  * [make\_sample\_data](#linear_models.classifiers.make_sample_data)
  * [\_validate\_data](#linear_models.classifiers._validate_data)
  * [LinearDiscriminantAnalysis](#linear_models.classifiers.LinearDiscriminantAnalysis)
    * [\_\_init\_\_](#linear_models.classifiers.LinearDiscriminantAnalysis.__init__)
    * [fit](#linear_models.classifiers.LinearDiscriminantAnalysis.fit)
    * [\_fit\_svd](#linear_models.classifiers.LinearDiscriminantAnalysis._fit_svd)
    * [\_fit\_lsqr](#linear_models.classifiers.LinearDiscriminantAnalysis._fit_lsqr)
    * [\_fit\_eigen](#linear_models.classifiers.LinearDiscriminantAnalysis._fit_eigen)
    * [predict](#linear_models.classifiers.LinearDiscriminantAnalysis.predict)
    * [decision\_function](#linear_models.classifiers.LinearDiscriminantAnalysis.decision_function)
  * [QuadraticDiscriminantAnalysis](#linear_models.classifiers.QuadraticDiscriminantAnalysis)
    * [\_\_init\_\_](#linear_models.classifiers.QuadraticDiscriminantAnalysis.__init__)
    * [fit](#linear_models.classifiers.QuadraticDiscriminantAnalysis.fit)
    * [predict](#linear_models.classifiers.QuadraticDiscriminantAnalysis.predict)
    * [decision\_function](#linear_models.classifiers.QuadraticDiscriminantAnalysis.decision_function)
  * [Perceptron](#linear_models.classifiers.Perceptron)
    * [\_\_init\_\_](#linear_models.classifiers.Perceptron.__init__)
    * [fit](#linear_models.classifiers.Perceptron.fit)
    * [predict](#linear_models.classifiers.Perceptron.predict)
  * [LogisticRegression](#linear_models.classifiers.LogisticRegression)
    * [\_\_init\_\_](#linear_models.classifiers.LogisticRegression.__init__)
    * [fit](#linear_models.classifiers.LogisticRegression.fit)
    * [predict](#linear_models.classifiers.LogisticRegression.predict)
    * [\_sigmoid](#linear_models.classifiers.LogisticRegression._sigmoid)
* [linear\_models.regressors](#linear_models.regressors)
  * [\_validate\_data](#linear_models.regressors._validate_data)
  * [OrdinaryLeastSquares](#linear_models.regressors.OrdinaryLeastSquares)
    * [\_\_init\_\_](#linear_models.regressors.OrdinaryLeastSquares.__init__)
    * [\_\_str\_\_](#linear_models.regressors.OrdinaryLeastSquares.__str__)
    * [fit](#linear_models.regressors.OrdinaryLeastSquares.fit)
    * [predict](#linear_models.regressors.OrdinaryLeastSquares.predict)
    * [get\_formula](#linear_models.regressors.OrdinaryLeastSquares.get_formula)
  * [Ridge](#linear_models.regressors.Ridge)
    * [\_\_init\_\_](#linear_models.regressors.Ridge.__init__)
    * [\_\_str\_\_](#linear_models.regressors.Ridge.__str__)
    * [fit](#linear_models.regressors.Ridge.fit)
    * [predict](#linear_models.regressors.Ridge.predict)
    * [get\_formula](#linear_models.regressors.Ridge.get_formula)
  * [Lasso](#linear_models.regressors.Lasso)
    * [\_\_init\_\_](#linear_models.regressors.Lasso.__init__)
    * [\_\_str\_\_](#linear_models.regressors.Lasso.__str__)
    * [fit](#linear_models.regressors.Lasso.fit)
    * [predict](#linear_models.regressors.Lasso.predict)
    * [get\_formula](#linear_models.regressors.Lasso.get_formula)
  * [Bayesian](#linear_models.regressors.Bayesian)
    * [\_\_init\_\_](#linear_models.regressors.Bayesian.__init__)
    * [\_\_str\_\_](#linear_models.regressors.Bayesian.__str__)
    * [fit](#linear_models.regressors.Bayesian.fit)
    * [tune](#linear_models.regressors.Bayesian.tune)
    * [predict](#linear_models.regressors.Bayesian.predict)
    * [get\_formula](#linear_models.regressors.Bayesian.get_formula)
  * [RANSAC](#linear_models.regressors.RANSAC)
    * [\_\_init\_\_](#linear_models.regressors.RANSAC.__init__)
    * [\_\_str\_\_](#linear_models.regressors.RANSAC.__str__)
    * [\_square\_loss](#linear_models.regressors.RANSAC._square_loss)
    * [\_mean\_square\_loss](#linear_models.regressors.RANSAC._mean_square_loss)
    * [fit](#linear_models.regressors.RANSAC.fit)
    * [predict](#linear_models.regressors.RANSAC.predict)
    * [get\_formula](#linear_models.regressors.RANSAC.get_formula)
  * [PassiveAggressiveRegressor](#linear_models.regressors.PassiveAggressiveRegressor)
    * [\_\_init\_\_](#linear_models.regressors.PassiveAggressiveRegressor.__init__)
    * [\_\_str\_\_](#linear_models.regressors.PassiveAggressiveRegressor.__str__)
    * [fit](#linear_models.regressors.PassiveAggressiveRegressor.fit)
    * [predict](#linear_models.regressors.PassiveAggressiveRegressor.predict)
    * [predict\_all\_steps](#linear_models.regressors.PassiveAggressiveRegressor.predict_all_steps)
    * [get\_formula](#linear_models.regressors.PassiveAggressiveRegressor.get_formula)
* [linear\_models.\_lasso\_jit\_utils](#linear_models._lasso_jit_utils)
  * [\_fit\_numba\_no\_intercept](#linear_models._lasso_jit_utils._fit_numba_no_intercept)
  * [\_fit\_numba\_intercept](#linear_models._lasso_jit_utils._fit_numba_intercept)
* [linear\_models.\_ridge\_jit\_utils](#linear_models._ridge_jit_utils)
  * [\_fit\_numba\_no\_intercept](#linear_models._ridge_jit_utils._fit_numba_no_intercept)
  * [\_fit\_numba\_intercept](#linear_models._ridge_jit_utils._fit_numba_intercept)
* [nearest\_neighbors](#nearest_neighbors)
  * [\_\_all\_\_](#nearest_neighbors.__all__)
* [nearest\_neighbors.base](#nearest_neighbors.base)
  * [KNeighborsBase](#nearest_neighbors.base.KNeighborsBase)
    * [\_\_init\_\_](#nearest_neighbors.base.KNeighborsBase.__init__)
    * [fit](#nearest_neighbors.base.KNeighborsBase.fit)
    * [get\_distance\_indices](#nearest_neighbors.base.KNeighborsBase.get_distance_indices)
    * [\_data\_precision](#nearest_neighbors.base.KNeighborsBase._data_precision)
    * [\_check\_data](#nearest_neighbors.base.KNeighborsBase._check_data)
    * [\_one\_hot\_encode](#nearest_neighbors.base.KNeighborsBase._one_hot_encode)
    * [\_compute\_distances](#nearest_neighbors.base.KNeighborsBase._compute_distances)
    * [\_compute\_distances\_euclidean](#nearest_neighbors.base.KNeighborsBase._compute_distances_euclidean)
    * [\_compute\_distances\_manhattan](#nearest_neighbors.base.KNeighborsBase._compute_distances_manhattan)
    * [\_compute\_distances\_minkowski](#nearest_neighbors.base.KNeighborsBase._compute_distances_minkowski)
    * [predict](#nearest_neighbors.base.KNeighborsBase.predict)
* [nearest\_neighbors.knn\_classifier](#nearest_neighbors.knn_classifier)
  * [KNeighborsClassifier](#nearest_neighbors.knn_classifier.KNeighborsClassifier)
    * [predict](#nearest_neighbors.knn_classifier.KNeighborsClassifier.predict)
* [nearest\_neighbors.knn\_regressor](#nearest_neighbors.knn_regressor)
  * [KNeighborsRegressor](#nearest_neighbors.knn_regressor.KNeighborsRegressor)
    * [predict](#nearest_neighbors.knn_regressor.KNeighborsRegressor.predict)
* [nearest\_neighbors.\_nearest\_neighbors\_jit\_utils](#nearest_neighbors._nearest_neighbors_jit_utils)
  * [\_jit\_compute\_distances\_euclidean](#nearest_neighbors._nearest_neighbors_jit_utils._jit_compute_distances_euclidean)
  * [\_jit\_compute\_distances\_manhattan](#nearest_neighbors._nearest_neighbors_jit_utils._jit_compute_distances_manhattan)
  * [\_jit\_compute\_distances\_minkowski](#nearest_neighbors._nearest_neighbors_jit_utils._jit_compute_distances_minkowski)
  * [\_numba\_predict\_regressor](#nearest_neighbors._nearest_neighbors_jit_utils._numba_predict_regressor)
  * [\_numba\_predict\_classifier](#nearest_neighbors._nearest_neighbors_jit_utils._numba_predict_classifier)
* [neural\_networks](#neural_networks)
  * [\_\_all\_\_](#neural_networks.__all__)
* [neural\_networks.activations](#neural_networks.activations)
  * [Activation](#neural_networks.activations.Activation)
    * [relu](#neural_networks.activations.Activation.relu)
    * [relu\_derivative](#neural_networks.activations.Activation.relu_derivative)
    * [leaky\_relu](#neural_networks.activations.Activation.leaky_relu)
    * [leaky\_relu\_derivative](#neural_networks.activations.Activation.leaky_relu_derivative)
    * [tanh](#neural_networks.activations.Activation.tanh)
    * [tanh\_derivative](#neural_networks.activations.Activation.tanh_derivative)
    * [sigmoid](#neural_networks.activations.Activation.sigmoid)
    * [sigmoid\_derivative](#neural_networks.activations.Activation.sigmoid_derivative)
    * [softmax](#neural_networks.activations.Activation.softmax)
* [neural\_networks.animation](#neural_networks.animation)
  * [TrainingAnimator](#neural_networks.animation.TrainingAnimator)
    * [\_\_init\_\_](#neural_networks.animation.TrainingAnimator.__init__)
    * [initialize](#neural_networks.animation.TrainingAnimator.initialize)
    * [update\_metrics](#neural_networks.animation.TrainingAnimator.update_metrics)
    * [animate\_training\_metrics](#neural_networks.animation.TrainingAnimator.animate_training_metrics)
    * [setup\_training\_video](#neural_networks.animation.TrainingAnimator.setup_training_video)
    * [add\_training\_frame](#neural_networks.animation.TrainingAnimator.add_training_frame)
    * [finish\_training\_video](#neural_networks.animation.TrainingAnimator.finish_training_video)
* [neural\_networks.cupy\_utils](#neural_networks.cupy_utils)
  * [fused\_dropout](#neural_networks.cupy_utils.fused_dropout)
  * [apply\_dropout](#neural_networks.cupy_utils.apply_dropout)
  * [fused\_relu](#neural_networks.cupy_utils.fused_relu)
  * [fused\_sigmoid](#neural_networks.cupy_utils.fused_sigmoid)
  * [fused\_leaky\_relu](#neural_networks.cupy_utils.fused_leaky_relu)
  * [forward\_cupy](#neural_networks.cupy_utils.forward_cupy)
  * [backward\_cupy](#neural_networks.cupy_utils.backward_cupy)
  * [logsumexp](#neural_networks.cupy_utils.logsumexp)
  * [calculate\_cross\_entropy\_loss](#neural_networks.cupy_utils.calculate_cross_entropy_loss)
  * [calculate\_bce\_with\_logits\_loss](#neural_networks.cupy_utils.calculate_bce_with_logits_loss)
  * [calculate\_loss\_from\_outputs\_binary](#neural_networks.cupy_utils.calculate_loss_from_outputs_binary)
  * [calculate\_loss\_from\_outputs\_multi](#neural_networks.cupy_utils.calculate_loss_from_outputs_multi)
  * [evaluate\_batch](#neural_networks.cupy_utils.evaluate_batch)
* [neural\_networks.layers](#neural_networks.layers)
  * [DenseLayer](#neural_networks.layers.DenseLayer)
    * [\_\_init\_\_](#neural_networks.layers.DenseLayer.__init__)
    * [zero\_grad](#neural_networks.layers.DenseLayer.zero_grad)
    * [forward](#neural_networks.layers.DenseLayer.forward)
    * [backward](#neural_networks.layers.DenseLayer.backward)
    * [activate](#neural_networks.layers.DenseLayer.activate)
    * [activation\_derivative](#neural_networks.layers.DenseLayer.activation_derivative)
  * [FlattenLayer](#neural_networks.layers.FlattenLayer)
    * [\_\_init\_\_](#neural_networks.layers.FlattenLayer.__init__)
    * [forward](#neural_networks.layers.FlattenLayer.forward)
    * [backward](#neural_networks.layers.FlattenLayer.backward)
  * [ConvLayer](#neural_networks.layers.ConvLayer)
    * [\_\_init\_\_](#neural_networks.layers.ConvLayer.__init__)
    * [zero\_grad](#neural_networks.layers.ConvLayer.zero_grad)
    * [\_im2col](#neural_networks.layers.ConvLayer._im2col)
    * [\_col2im](#neural_networks.layers.ConvLayer._col2im)
    * [forward](#neural_networks.layers.ConvLayer.forward)
    * [backward](#neural_networks.layers.ConvLayer.backward)
    * [activate](#neural_networks.layers.ConvLayer.activate)
  * [RNNLayer](#neural_networks.layers.RNNLayer)
    * [\_\_init\_\_](#neural_networks.layers.RNNLayer.__init__)
* [neural\_networks.layers\_cupy](#neural_networks.layers_cupy)
  * [CuPyDenseLayer](#neural_networks.layers_cupy.CuPyDenseLayer)
    * [\_\_init\_\_](#neural_networks.layers_cupy.CuPyDenseLayer.__init__)
    * [zero\_grad](#neural_networks.layers_cupy.CuPyDenseLayer.zero_grad)
    * [activate](#neural_networks.layers_cupy.CuPyDenseLayer.activate)
    * [activation\_derivative](#neural_networks.layers_cupy.CuPyDenseLayer.activation_derivative)
  * [CuPyActivation](#neural_networks.layers_cupy.CuPyActivation)
    * [relu](#neural_networks.layers_cupy.CuPyActivation.relu)
    * [relu\_derivative](#neural_networks.layers_cupy.CuPyActivation.relu_derivative)
    * [leaky\_relu](#neural_networks.layers_cupy.CuPyActivation.leaky_relu)
    * [leaky\_relu\_derivative](#neural_networks.layers_cupy.CuPyActivation.leaky_relu_derivative)
    * [tanh](#neural_networks.layers_cupy.CuPyActivation.tanh)
    * [tanh\_derivative](#neural_networks.layers_cupy.CuPyActivation.tanh_derivative)
    * [sigmoid](#neural_networks.layers_cupy.CuPyActivation.sigmoid)
    * [sigmoid\_derivative](#neural_networks.layers_cupy.CuPyActivation.sigmoid_derivative)
    * [softmax](#neural_networks.layers_cupy.CuPyActivation.softmax)
* [neural\_networks.layers\_jit](#neural_networks.layers_jit)
  * [spec](#neural_networks.layers_jit.spec)
  * [JITDenseLayer](#neural_networks.layers_jit.JITDenseLayer)
    * [\_\_init\_\_](#neural_networks.layers_jit.JITDenseLayer.__init__)
    * [zero\_grad](#neural_networks.layers_jit.JITDenseLayer.zero_grad)
    * [forward](#neural_networks.layers_jit.JITDenseLayer.forward)
    * [backward](#neural_networks.layers_jit.JITDenseLayer.backward)
    * [activate](#neural_networks.layers_jit.JITDenseLayer.activate)
    * [activation\_derivative](#neural_networks.layers_jit.JITDenseLayer.activation_derivative)
  * [flatten\_spec](#neural_networks.layers_jit.flatten_spec)
  * [JITFlattenLayer](#neural_networks.layers_jit.JITFlattenLayer)
    * [\_\_init\_\_](#neural_networks.layers_jit.JITFlattenLayer.__init__)
    * [forward](#neural_networks.layers_jit.JITFlattenLayer.forward)
    * [backward](#neural_networks.layers_jit.JITFlattenLayer.backward)
  * [conv\_spec](#neural_networks.layers_jit.conv_spec)
  * [JITConvLayer](#neural_networks.layers_jit.JITConvLayer)
    * [\_\_init\_\_](#neural_networks.layers_jit.JITConvLayer.__init__)
    * [zero\_grad](#neural_networks.layers_jit.JITConvLayer.zero_grad)
    * [\_im2col](#neural_networks.layers_jit.JITConvLayer._im2col)
    * [\_col2im](#neural_networks.layers_jit.JITConvLayer._col2im)
    * [forward](#neural_networks.layers_jit.JITConvLayer.forward)
    * [backward](#neural_networks.layers_jit.JITConvLayer.backward)
    * [activate](#neural_networks.layers_jit.JITConvLayer.activate)
    * [activation\_derivative](#neural_networks.layers_jit.JITConvLayer.activation_derivative)
  * [JITRNNLayer](#neural_networks.layers_jit.JITRNNLayer)
    * [\_\_init\_\_](#neural_networks.layers_jit.JITRNNLayer.__init__)
* [neural\_networks.loss](#neural_networks.loss)
  * [\_validate\_shapes](#neural_networks.loss._validate_shapes)
  * [CrossEntropyLoss](#neural_networks.loss.CrossEntropyLoss)
    * [\_\_call\_\_](#neural_networks.loss.CrossEntropyLoss.__call__)
  * [BCEWithLogitsLoss](#neural_networks.loss.BCEWithLogitsLoss)
    * [\_\_call\_\_](#neural_networks.loss.BCEWithLogitsLoss.__call__)
  * [MeanSquaredErrorLoss](#neural_networks.loss.MeanSquaredErrorLoss)
    * [\_\_call\_\_](#neural_networks.loss.MeanSquaredErrorLoss.__call__)
  * [MeanAbsoluteErrorLoss](#neural_networks.loss.MeanAbsoluteErrorLoss)
    * [\_\_call\_\_](#neural_networks.loss.MeanAbsoluteErrorLoss.__call__)
  * [HuberLoss](#neural_networks.loss.HuberLoss)
    * [\_\_call\_\_](#neural_networks.loss.HuberLoss.__call__)
* [neural\_networks.loss\_cupy](#neural_networks.loss_cupy)
  * [CuPyCrossEntropyLoss](#neural_networks.loss_cupy.CuPyCrossEntropyLoss)
    * [\_\_call\_\_](#neural_networks.loss_cupy.CuPyCrossEntropyLoss.__call__)
  * [CuPyBCEWithLogitsLoss](#neural_networks.loss_cupy.CuPyBCEWithLogitsLoss)
    * [\_\_call\_\_](#neural_networks.loss_cupy.CuPyBCEWithLogitsLoss.__call__)
* [neural\_networks.loss\_jit](#neural_networks.loss_jit)
  * [CACHE](#neural_networks.loss_jit.CACHE)
  * [\_validate\_shapes](#neural_networks.loss_jit._validate_shapes)
  * [JITCrossEntropyLoss](#neural_networks.loss_jit.JITCrossEntropyLoss)
    * [\_\_init\_\_](#neural_networks.loss_jit.JITCrossEntropyLoss.__init__)
    * [calculate\_loss](#neural_networks.loss_jit.JITCrossEntropyLoss.calculate_loss)
  * [calculate\_cross\_entropy\_loss](#neural_networks.loss_jit.calculate_cross_entropy_loss)
  * [JITBCEWithLogitsLoss](#neural_networks.loss_jit.JITBCEWithLogitsLoss)
    * [\_\_init\_\_](#neural_networks.loss_jit.JITBCEWithLogitsLoss.__init__)
    * [calculate\_loss](#neural_networks.loss_jit.JITBCEWithLogitsLoss.calculate_loss)
  * [calculate\_bce\_with\_logits\_loss](#neural_networks.loss_jit.calculate_bce_with_logits_loss)
  * [JITMeanSquaredErrorLoss](#neural_networks.loss_jit.JITMeanSquaredErrorLoss)
    * [calculate\_loss](#neural_networks.loss_jit.JITMeanSquaredErrorLoss.calculate_loss)
  * [JITMeanAbsoluteErrorLoss](#neural_networks.loss_jit.JITMeanAbsoluteErrorLoss)
    * [calculate\_loss](#neural_networks.loss_jit.JITMeanAbsoluteErrorLoss.calculate_loss)
  * [JITHuberLoss](#neural_networks.loss_jit.JITHuberLoss)
    * [\_\_init\_\_](#neural_networks.loss_jit.JITHuberLoss.__init__)
    * [calculate\_loss](#neural_networks.loss_jit.JITHuberLoss.calculate_loss)
* [neural\_networks.neuralNetworkBase](#neural_networks.neuralNetworkBase)
  * [NeuralNetworkBase](#neural_networks.neuralNetworkBase.NeuralNetworkBase)
    * [\_\_init\_\_](#neural_networks.neuralNetworkBase.NeuralNetworkBase.__init__)
    * [initialize\_layers](#neural_networks.neuralNetworkBase.NeuralNetworkBase.initialize_layers)
    * [forward](#neural_networks.neuralNetworkBase.NeuralNetworkBase.forward)
    * [backward](#neural_networks.neuralNetworkBase.NeuralNetworkBase.backward)
    * [train](#neural_networks.neuralNetworkBase.NeuralNetworkBase.train)
    * [evaluate](#neural_networks.neuralNetworkBase.NeuralNetworkBase.evaluate)
    * [predict](#neural_networks.neuralNetworkBase.NeuralNetworkBase.predict)
    * [calculate\_loss](#neural_networks.neuralNetworkBase.NeuralNetworkBase.calculate_loss)
    * [apply\_dropout](#neural_networks.neuralNetworkBase.NeuralNetworkBase.apply_dropout)
    * [compute\_l2\_reg](#neural_networks.neuralNetworkBase.NeuralNetworkBase.compute_l2_reg)
    * [calculate\_precision\_recall\_f1](#neural_networks.neuralNetworkBase.NeuralNetworkBase.calculate_precision_recall_f1)
    * [create\_scheduler](#neural_networks.neuralNetworkBase.NeuralNetworkBase.create_scheduler)
    * [plot\_metrics](#neural_networks.neuralNetworkBase.NeuralNetworkBase.plot_metrics)
* [neural\_networks.neuralNetworkBaseBackend](#neural_networks.neuralNetworkBaseBackend)
  * [BaseBackendNeuralNetwork](#neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork)
    * [\_\_init\_\_](#neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.__init__)
    * [initialize\_new\_layers](#neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.initialize_new_layers)
    * [forward](#neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.forward)
    * [backward](#neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.backward)
    * [fit](#neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.fit)
    * [train](#neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.train)
    * [evaluate](#neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.evaluate)
    * [predict](#neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.predict)
    * [calculate\_loss](#neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.calculate_loss)
    * [\_create\_optimizer](#neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork._create_optimizer)
    * [tune\_hyperparameters](#neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.tune_hyperparameters)
    * [train\_with\_animation\_capture](#neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.train_with_animation_capture)
* [neural\_networks.neuralNetworkCuPyBackend](#neural_networks.neuralNetworkCuPyBackend)
  * [CuPyBackendNeuralNetwork](#neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork)
    * [\_\_init\_\_](#neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.__init__)
    * [initialize\_new\_layers](#neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.initialize_new_layers)
    * [apply\_dropout](#neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.apply_dropout)
    * [forward](#neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.forward)
    * [backward](#neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.backward)
    * [\_process\_batches\_async](#neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork._process_batches_async)
    * [is\_not\_instance\_of\_classes](#neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.is_not_instance_of_classes)
    * [train](#neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.train)
    * [evaluate](#neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.evaluate)
    * [\_evaluate\_cupy](#neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork._evaluate_cupy)
    * [predict](#neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.predict)
    * [calculate\_loss](#neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.calculate_loss)
    * [\_create\_optimizer](#neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork._create_optimizer)
* [neural\_networks.neuralNetworkNumbaBackend](#neural_networks.neuralNetworkNumbaBackend)
  * [NumbaBackendNeuralNetwork](#neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork)
    * [\_\_init\_\_](#neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.__init__)
    * [store\_init\_layers](#neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.store_init_layers)
    * [restore\_layers](#neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.restore_layers)
    * [initialize\_new\_layers](#neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.initialize_new_layers)
    * [forward](#neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.forward)
    * [backward](#neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.backward)
    * [is\_not\_instance\_of\_classes](#neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.is_not_instance_of_classes)
    * [train](#neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.train)
    * [evaluate](#neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.evaluate)
    * [predict](#neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.predict)
    * [calculate\_loss](#neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.calculate_loss)
    * [\_create\_optimizer](#neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork._create_optimizer)
    * [tune\_hyperparameters](#neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.tune_hyperparameters)
    * [\_get\_jit\_loss\_calculator](#neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork._get_jit_loss_calculator)
    * [compile\_numba\_functions](#neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.compile_numba_functions)
* [neural\_networks.numba\_utils](#neural_networks.numba_utils)
  * [CACHE](#neural_networks.numba_utils.CACHE)
  * [calculate\_loss\_from\_outputs\_binary](#neural_networks.numba_utils.calculate_loss_from_outputs_binary)
  * [calculate\_loss\_from\_outputs\_multi](#neural_networks.numba_utils.calculate_loss_from_outputs_multi)
  * [calculate\_cross\_entropy\_loss](#neural_networks.numba_utils.calculate_cross_entropy_loss)
  * [calculate\_bce\_with\_logits\_loss](#neural_networks.numba_utils.calculate_bce_with_logits_loss)
  * [\_compute\_l2\_reg](#neural_networks.numba_utils._compute_l2_reg)
  * [evaluate\_batch](#neural_networks.numba_utils.evaluate_batch)
  * [calculate\_mse\_loss](#neural_networks.numba_utils.calculate_mse_loss)
  * [calculate\_mae\_loss](#neural_networks.numba_utils.calculate_mae_loss)
  * [calculate\_huber\_loss](#neural_networks.numba_utils.calculate_huber_loss)
  * [relu](#neural_networks.numba_utils.relu)
  * [relu\_derivative](#neural_networks.numba_utils.relu_derivative)
  * [leaky\_relu](#neural_networks.numba_utils.leaky_relu)
  * [leaky\_relu\_derivative](#neural_networks.numba_utils.leaky_relu_derivative)
  * [tanh](#neural_networks.numba_utils.tanh)
  * [tanh\_derivative](#neural_networks.numba_utils.tanh_derivative)
  * [sigmoid](#neural_networks.numba_utils.sigmoid)
  * [sigmoid\_derivative](#neural_networks.numba_utils.sigmoid_derivative)
  * [softmax](#neural_networks.numba_utils.softmax)
  * [sum\_reduce](#neural_networks.numba_utils.sum_reduce)
  * [sum\_axis0](#neural_networks.numba_utils.sum_axis0)
  * [apply\_dropout\_jit](#neural_networks.numba_utils.apply_dropout_jit)
  * [compute\_l2\_reg](#neural_networks.numba_utils.compute_l2_reg)
  * [one\_hot\_encode](#neural_networks.numba_utils.one_hot_encode)
  * [process\_batches\_binary](#neural_networks.numba_utils.process_batches_binary)
  * [process\_batches\_multi](#neural_networks.numba_utils.process_batches_multi)
  * [process\_batches\_regression\_jit](#neural_networks.numba_utils.process_batches_regression_jit)
  * [evaluate\_jit](#neural_networks.numba_utils.evaluate_jit)
  * [evaluate\_regression\_jit](#neural_networks.numba_utils.evaluate_regression_jit)
* [neural\_networks.optimizers](#neural_networks.optimizers)
  * [AdamOptimizer](#neural_networks.optimizers.AdamOptimizer)
    * [\_\_init\_\_](#neural_networks.optimizers.AdamOptimizer.__init__)
    * [initialize](#neural_networks.optimizers.AdamOptimizer.initialize)
    * [update](#neural_networks.optimizers.AdamOptimizer.update)
  * [SGDOptimizer](#neural_networks.optimizers.SGDOptimizer)
    * [\_\_init\_\_](#neural_networks.optimizers.SGDOptimizer.__init__)
    * [initialize](#neural_networks.optimizers.SGDOptimizer.initialize)
    * [update](#neural_networks.optimizers.SGDOptimizer.update)
  * [AdadeltaOptimizer](#neural_networks.optimizers.AdadeltaOptimizer)
    * [\_\_init\_\_](#neural_networks.optimizers.AdadeltaOptimizer.__init__)
    * [initialize](#neural_networks.optimizers.AdadeltaOptimizer.initialize)
    * [update](#neural_networks.optimizers.AdadeltaOptimizer.update)
* [neural\_networks.optimizers\_cupy](#neural_networks.optimizers_cupy)
  * [CuPyAdamOptimizer](#neural_networks.optimizers_cupy.CuPyAdamOptimizer)
    * [\_\_init\_\_](#neural_networks.optimizers_cupy.CuPyAdamOptimizer.__init__)
    * [initialize](#neural_networks.optimizers_cupy.CuPyAdamOptimizer.initialize)
    * [update\_layers](#neural_networks.optimizers_cupy.CuPyAdamOptimizer.update_layers)
  * [CuPySGDOptimizer](#neural_networks.optimizers_cupy.CuPySGDOptimizer)
    * [\_\_init\_\_](#neural_networks.optimizers_cupy.CuPySGDOptimizer.__init__)
    * [initialize](#neural_networks.optimizers_cupy.CuPySGDOptimizer.initialize)
    * [update\_layers](#neural_networks.optimizers_cupy.CuPySGDOptimizer.update_layers)
  * [CuPyAdadeltaOptimizer](#neural_networks.optimizers_cupy.CuPyAdadeltaOptimizer)
    * [\_\_init\_\_](#neural_networks.optimizers_cupy.CuPyAdadeltaOptimizer.__init__)
    * [initialize](#neural_networks.optimizers_cupy.CuPyAdadeltaOptimizer.initialize)
    * [update\_layers](#neural_networks.optimizers_cupy.CuPyAdadeltaOptimizer.update_layers)
* [neural\_networks.optimizers\_jit](#neural_networks.optimizers_jit)
  * [CACHE](#neural_networks.optimizers_jit.CACHE)
  * [spec\_adam](#neural_networks.optimizers_jit.spec_adam)
  * [JITAdamOptimizer](#neural_networks.optimizers_jit.JITAdamOptimizer)
    * [\_\_init\_\_](#neural_networks.optimizers_jit.JITAdamOptimizer.__init__)
    * [initialize](#neural_networks.optimizers_jit.JITAdamOptimizer.initialize)
    * [update](#neural_networks.optimizers_jit.JITAdamOptimizer.update)
    * [update\_layers](#neural_networks.optimizers_jit.JITAdamOptimizer.update_layers)
  * [adam\_update\_layers](#neural_networks.optimizers_jit.adam_update_layers)
  * [spec\_sgd](#neural_networks.optimizers_jit.spec_sgd)
  * [JITSGDOptimizer](#neural_networks.optimizers_jit.JITSGDOptimizer)
    * [\_\_init\_\_](#neural_networks.optimizers_jit.JITSGDOptimizer.__init__)
    * [initialize](#neural_networks.optimizers_jit.JITSGDOptimizer.initialize)
    * [update](#neural_networks.optimizers_jit.JITSGDOptimizer.update)
    * [update\_layers](#neural_networks.optimizers_jit.JITSGDOptimizer.update_layers)
  * [sgd\_update\_layers](#neural_networks.optimizers_jit.sgd_update_layers)
  * [spec\_adadelta](#neural_networks.optimizers_jit.spec_adadelta)
  * [JITAdadeltaOptimizer](#neural_networks.optimizers_jit.JITAdadeltaOptimizer)
    * [\_\_init\_\_](#neural_networks.optimizers_jit.JITAdadeltaOptimizer.__init__)
    * [initialize](#neural_networks.optimizers_jit.JITAdadeltaOptimizer.initialize)
    * [update](#neural_networks.optimizers_jit.JITAdadeltaOptimizer.update)
    * [update\_layers](#neural_networks.optimizers_jit.JITAdadeltaOptimizer.update_layers)
  * [adadelta\_update\_layers](#neural_networks.optimizers_jit.adadelta_update_layers)
* [neural\_networks.schedulers](#neural_networks.schedulers)
  * [lr\_scheduler\_step](#neural_networks.schedulers.lr_scheduler_step)
    * [\_\_init\_\_](#neural_networks.schedulers.lr_scheduler_step.__init__)
    * [\_\_repr\_\_](#neural_networks.schedulers.lr_scheduler_step.__repr__)
    * [step](#neural_networks.schedulers.lr_scheduler_step.step)
    * [reduce](#neural_networks.schedulers.lr_scheduler_step.reduce)
  * [lr\_scheduler\_exp](#neural_networks.schedulers.lr_scheduler_exp)
    * [\_\_init\_\_](#neural_networks.schedulers.lr_scheduler_exp.__init__)
    * [\_\_repr\_\_](#neural_networks.schedulers.lr_scheduler_exp.__repr__)
    * [step](#neural_networks.schedulers.lr_scheduler_exp.step)
    * [reduce](#neural_networks.schedulers.lr_scheduler_exp.reduce)
  * [lr\_scheduler\_plateau](#neural_networks.schedulers.lr_scheduler_plateau)
    * [\_\_init\_\_](#neural_networks.schedulers.lr_scheduler_plateau.__init__)
    * [\_\_repr\_\_](#neural_networks.schedulers.lr_scheduler_plateau.__repr__)
    * [step](#neural_networks.schedulers.lr_scheduler_plateau.step)
* [neural\_networks\_cupy\_dev](#neural_networks_cupy_dev)
  * [\_\_all\_\_](#neural_networks_cupy_dev.__all__)
* [neural\_networks\_cupy\_dev.cupy\_utils](#neural_networks_cupy_dev.cupy_utils)
  * [fused\_dropout](#neural_networks_cupy_dev.cupy_utils.fused_dropout)
  * [apply\_dropout](#neural_networks_cupy_dev.cupy_utils.apply_dropout)
  * [fused\_relu](#neural_networks_cupy_dev.cupy_utils.fused_relu)
  * [fused\_sigmoid](#neural_networks_cupy_dev.cupy_utils.fused_sigmoid)
  * [fused\_leaky\_relu](#neural_networks_cupy_dev.cupy_utils.fused_leaky_relu)
  * [forward\_cupy](#neural_networks_cupy_dev.cupy_utils.forward_cupy)
  * [backward\_cupy](#neural_networks_cupy_dev.cupy_utils.backward_cupy)
  * [logsumexp](#neural_networks_cupy_dev.cupy_utils.logsumexp)
  * [calculate\_cross\_entropy\_loss](#neural_networks_cupy_dev.cupy_utils.calculate_cross_entropy_loss)
  * [calculate\_bce\_with\_logits\_loss](#neural_networks_cupy_dev.cupy_utils.calculate_bce_with_logits_loss)
  * [calculate\_loss\_from\_outputs\_binary](#neural_networks_cupy_dev.cupy_utils.calculate_loss_from_outputs_binary)
  * [calculate\_loss\_from\_outputs\_multi](#neural_networks_cupy_dev.cupy_utils.calculate_loss_from_outputs_multi)
  * [evaluate\_batch](#neural_networks_cupy_dev.cupy_utils.evaluate_batch)
* [neural\_networks\_cupy\_dev.loss](#neural_networks_cupy_dev.loss)
  * [CrossEntropyLoss](#neural_networks_cupy_dev.loss.CrossEntropyLoss)
    * [\_\_call\_\_](#neural_networks_cupy_dev.loss.CrossEntropyLoss.__call__)
  * [BCEWithLogitsLoss](#neural_networks_cupy_dev.loss.BCEWithLogitsLoss)
    * [\_\_call\_\_](#neural_networks_cupy_dev.loss.BCEWithLogitsLoss.__call__)
* [neural\_networks\_cupy\_dev.neuralNetwork](#neural_networks_cupy_dev.neuralNetwork)
  * [NeuralNetwork](#neural_networks_cupy_dev.neuralNetwork.NeuralNetwork)
    * [\_\_init\_\_](#neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.__init__)
    * [apply\_dropout](#neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.apply_dropout)
    * [forward](#neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.forward)
    * [backward](#neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.backward)
    * [\_process\_batches\_async](#neural_networks_cupy_dev.neuralNetwork.NeuralNetwork._process_batches_async)
    * [train](#neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.train)
    * [calculate\_loss](#neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.calculate_loss)
    * [evaluate](#neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.evaluate)
    * [\_evaluate\_cupy](#neural_networks_cupy_dev.neuralNetwork.NeuralNetwork._evaluate_cupy)
    * [predict](#neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.predict)
    * [\_create\_optimizer](#neural_networks_cupy_dev.neuralNetwork.NeuralNetwork._create_optimizer)
    * [create\_scheduler](#neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.create_scheduler)
  * [Layer](#neural_networks_cupy_dev.neuralNetwork.Layer)
    * [\_\_init\_\_](#neural_networks_cupy_dev.neuralNetwork.Layer.__init__)
    * [zero\_grad](#neural_networks_cupy_dev.neuralNetwork.Layer.zero_grad)
    * [activate](#neural_networks_cupy_dev.neuralNetwork.Layer.activate)
    * [activation\_derivative](#neural_networks_cupy_dev.neuralNetwork.Layer.activation_derivative)
  * [Activation](#neural_networks_cupy_dev.neuralNetwork.Activation)
    * [relu](#neural_networks_cupy_dev.neuralNetwork.Activation.relu)
    * [relu\_derivative](#neural_networks_cupy_dev.neuralNetwork.Activation.relu_derivative)
    * [leaky\_relu](#neural_networks_cupy_dev.neuralNetwork.Activation.leaky_relu)
    * [leaky\_relu\_derivative](#neural_networks_cupy_dev.neuralNetwork.Activation.leaky_relu_derivative)
    * [tanh](#neural_networks_cupy_dev.neuralNetwork.Activation.tanh)
    * [tanh\_derivative](#neural_networks_cupy_dev.neuralNetwork.Activation.tanh_derivative)
    * [sigmoid](#neural_networks_cupy_dev.neuralNetwork.Activation.sigmoid)
    * [sigmoid\_derivative](#neural_networks_cupy_dev.neuralNetwork.Activation.sigmoid_derivative)
    * [softmax](#neural_networks_cupy_dev.neuralNetwork.Activation.softmax)
* [neural\_networks\_cupy\_dev.optimizers](#neural_networks_cupy_dev.optimizers)
  * [AdamOptimizer](#neural_networks_cupy_dev.optimizers.AdamOptimizer)
    * [\_\_init\_\_](#neural_networks_cupy_dev.optimizers.AdamOptimizer.__init__)
    * [initialize](#neural_networks_cupy_dev.optimizers.AdamOptimizer.initialize)
    * [update\_layers](#neural_networks_cupy_dev.optimizers.AdamOptimizer.update_layers)
  * [SGDOptimizer](#neural_networks_cupy_dev.optimizers.SGDOptimizer)
    * [\_\_init\_\_](#neural_networks_cupy_dev.optimizers.SGDOptimizer.__init__)
    * [initialize](#neural_networks_cupy_dev.optimizers.SGDOptimizer.initialize)
    * [update\_layers](#neural_networks_cupy_dev.optimizers.SGDOptimizer.update_layers)
  * [AdadeltaOptimizer](#neural_networks_cupy_dev.optimizers.AdadeltaOptimizer)
    * [\_\_init\_\_](#neural_networks_cupy_dev.optimizers.AdadeltaOptimizer.__init__)
    * [initialize](#neural_networks_cupy_dev.optimizers.AdadeltaOptimizer.initialize)
    * [update\_layers](#neural_networks_cupy_dev.optimizers.AdadeltaOptimizer.update_layers)
* [neural\_networks\_cupy\_dev.schedulers](#neural_networks_cupy_dev.schedulers)
  * [lr\_scheduler\_step](#neural_networks_cupy_dev.schedulers.lr_scheduler_step)
    * [\_\_init\_\_](#neural_networks_cupy_dev.schedulers.lr_scheduler_step.__init__)
    * [\_\_repr\_\_](#neural_networks_cupy_dev.schedulers.lr_scheduler_step.__repr__)
    * [step](#neural_networks_cupy_dev.schedulers.lr_scheduler_step.step)
    * [reduce](#neural_networks_cupy_dev.schedulers.lr_scheduler_step.reduce)
  * [lr\_scheduler\_exp](#neural_networks_cupy_dev.schedulers.lr_scheduler_exp)
    * [\_\_init\_\_](#neural_networks_cupy_dev.schedulers.lr_scheduler_exp.__init__)
    * [\_\_repr\_\_](#neural_networks_cupy_dev.schedulers.lr_scheduler_exp.__repr__)
    * [step](#neural_networks_cupy_dev.schedulers.lr_scheduler_exp.step)
    * [reduce](#neural_networks_cupy_dev.schedulers.lr_scheduler_exp.reduce)
  * [lr\_scheduler\_plateau](#neural_networks_cupy_dev.schedulers.lr_scheduler_plateau)
    * [\_\_init\_\_](#neural_networks_cupy_dev.schedulers.lr_scheduler_plateau.__init__)
    * [\_\_repr\_\_](#neural_networks_cupy_dev.schedulers.lr_scheduler_plateau.__repr__)
    * [step](#neural_networks_cupy_dev.schedulers.lr_scheduler_plateau.step)
* [neural\_networks\_numba\_dev](#neural_networks_numba_dev)
  * [\_\_all\_\_](#neural_networks_numba_dev.__all__)
* [neural\_networks\_numba\_dev.layers\_jit\_unified](#neural_networks_numba_dev.layers_jit_unified)
  * [layer\_spec](#neural_networks_numba_dev.layers_jit_unified.layer_spec)
  * [JITLayer](#neural_networks_numba_dev.layers_jit_unified.JITLayer)
    * [\_\_init\_\_](#neural_networks_numba_dev.layers_jit_unified.JITLayer.__init__)
    * [zero\_grad](#neural_networks_numba_dev.layers_jit_unified.JITLayer.zero_grad)
    * [forward](#neural_networks_numba_dev.layers_jit_unified.JITLayer.forward)
    * [backward](#neural_networks_numba_dev.layers_jit_unified.JITLayer.backward)
    * [\_forward\_dense](#neural_networks_numba_dev.layers_jit_unified.JITLayer._forward_dense)
    * [\_backward\_dense](#neural_networks_numba_dev.layers_jit_unified.JITLayer._backward_dense)
    * [\_forward\_conv](#neural_networks_numba_dev.layers_jit_unified.JITLayer._forward_conv)
    * [\_backward\_conv](#neural_networks_numba_dev.layers_jit_unified.JITLayer._backward_conv)
    * [\_forward\_flatten](#neural_networks_numba_dev.layers_jit_unified.JITLayer._forward_flatten)
    * [\_backward\_flatten](#neural_networks_numba_dev.layers_jit_unified.JITLayer._backward_flatten)
    * [activate](#neural_networks_numba_dev.layers_jit_unified.JITLayer.activate)
    * [activation\_derivative](#neural_networks_numba_dev.layers_jit_unified.JITLayer.activation_derivative)
    * [\_im2col](#neural_networks_numba_dev.layers_jit_unified.JITLayer._im2col)
    * [\_col2im](#neural_networks_numba_dev.layers_jit_unified.JITLayer._col2im)
* [neural\_networks\_numba\_dev.layer\_jit\_utils](#neural_networks_numba_dev.layer_jit_utils)
  * [CACHE](#neural_networks_numba_dev.layer_jit_utils.CACHE)
  * [forward\_dense](#neural_networks_numba_dev.layer_jit_utils.forward_dense)
  * [backward\_dense](#neural_networks_numba_dev.layer_jit_utils.backward_dense)
  * [activate](#neural_networks_numba_dev.layer_jit_utils.activate)
  * [activation\_derivative](#neural_networks_numba_dev.layer_jit_utils.activation_derivative)
* [neural\_networks\_numba\_dev.loss\_jit](#neural_networks_numba_dev.loss_jit)
  * [CACHE](#neural_networks_numba_dev.loss_jit.CACHE)
  * [JITCrossEntropyLoss](#neural_networks_numba_dev.loss_jit.JITCrossEntropyLoss)
    * [\_\_init\_\_](#neural_networks_numba_dev.loss_jit.JITCrossEntropyLoss.__init__)
    * [calculate\_loss](#neural_networks_numba_dev.loss_jit.JITCrossEntropyLoss.calculate_loss)
  * [calculate\_cross\_entropy\_loss](#neural_networks_numba_dev.loss_jit.calculate_cross_entropy_loss)
  * [JITBCEWithLogitsLoss](#neural_networks_numba_dev.loss_jit.JITBCEWithLogitsLoss)
    * [\_\_init\_\_](#neural_networks_numba_dev.loss_jit.JITBCEWithLogitsLoss.__init__)
    * [calculate\_loss](#neural_networks_numba_dev.loss_jit.JITBCEWithLogitsLoss.calculate_loss)
  * [calculate\_bce\_with\_logits\_loss](#neural_networks_numba_dev.loss_jit.calculate_bce_with_logits_loss)
* [neural\_networks\_numba\_dev.neuralNetworkBase](#neural_networks_numba_dev.neuralNetworkBase)
  * [NeuralNetworkBase](#neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase)
    * [\_\_init\_\_](#neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.__init__)
    * [initialize\_layers](#neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.initialize_layers)
    * [forward](#neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.forward)
    * [backward](#neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.backward)
    * [train](#neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.train)
    * [evaluate](#neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.evaluate)
    * [predict](#neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.predict)
    * [calculate\_loss](#neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.calculate_loss)
    * [apply\_dropout](#neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.apply_dropout)
    * [compute\_l2\_reg](#neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.compute_l2_reg)
    * [calculate\_precision\_recall\_f1](#neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.calculate_precision_recall_f1)
    * [create\_scheduler](#neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.create_scheduler)
    * [plot\_metrics](#neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.plot_metrics)
* [neural\_networks\_numba\_dev.neuralNetworkNumbaBackend](#neural_networks_numba_dev.neuralNetworkNumbaBackend)
  * [NumbaBackendNeuralNetwork](#neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork)
    * [\_\_init\_\_](#neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.__init__)
    * [store\_init\_layers](#neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.store_init_layers)
    * [restore\_layers](#neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.restore_layers)
    * [initialize\_new\_layers](#neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.initialize_new_layers)
    * [forward](#neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.forward)
    * [backward](#neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.backward)
    * [is\_not\_instance\_of\_classes](#neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.is_not_instance_of_classes)
    * [train](#neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.train)
    * [evaluate](#neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.evaluate)
    * [predict](#neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.predict)
    * [calculate\_loss](#neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.calculate_loss)
    * [\_create\_optimizer](#neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork._create_optimizer)
    * [tune\_hyperparameters](#neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.tune_hyperparameters)
    * [compile\_numba\_functions](#neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.compile_numba_functions)
* [neural\_networks\_numba\_dev.numba\_utils](#neural_networks_numba_dev.numba_utils)
  * [CACHE](#neural_networks_numba_dev.numba_utils.CACHE)
  * [calculate\_loss\_from\_outputs\_binary](#neural_networks_numba_dev.numba_utils.calculate_loss_from_outputs_binary)
  * [calculate\_loss\_from\_outputs\_multi](#neural_networks_numba_dev.numba_utils.calculate_loss_from_outputs_multi)
  * [calculate\_cross\_entropy\_loss](#neural_networks_numba_dev.numba_utils.calculate_cross_entropy_loss)
  * [calculate\_bce\_with\_logits\_loss](#neural_networks_numba_dev.numba_utils.calculate_bce_with_logits_loss)
  * [\_compute\_l2\_reg](#neural_networks_numba_dev.numba_utils._compute_l2_reg)
  * [evaluate\_batch](#neural_networks_numba_dev.numba_utils.evaluate_batch)
  * [relu](#neural_networks_numba_dev.numba_utils.relu)
  * [relu\_derivative](#neural_networks_numba_dev.numba_utils.relu_derivative)
  * [leaky\_relu](#neural_networks_numba_dev.numba_utils.leaky_relu)
  * [leaky\_relu\_derivative](#neural_networks_numba_dev.numba_utils.leaky_relu_derivative)
  * [tanh](#neural_networks_numba_dev.numba_utils.tanh)
  * [tanh\_derivative](#neural_networks_numba_dev.numba_utils.tanh_derivative)
  * [sigmoid](#neural_networks_numba_dev.numba_utils.sigmoid)
  * [sigmoid\_derivative](#neural_networks_numba_dev.numba_utils.sigmoid_derivative)
  * [softmax](#neural_networks_numba_dev.numba_utils.softmax)
  * [sum\_reduce](#neural_networks_numba_dev.numba_utils.sum_reduce)
  * [sum\_axis0](#neural_networks_numba_dev.numba_utils.sum_axis0)
  * [apply\_dropout\_jit](#neural_networks_numba_dev.numba_utils.apply_dropout_jit)
  * [compute\_l2\_reg](#neural_networks_numba_dev.numba_utils.compute_l2_reg)
  * [one\_hot\_encode](#neural_networks_numba_dev.numba_utils.one_hot_encode)
  * [process\_batches\_binary](#neural_networks_numba_dev.numba_utils.process_batches_binary)
  * [process\_batches\_multi](#neural_networks_numba_dev.numba_utils.process_batches_multi)
  * [evaluate\_jit](#neural_networks_numba_dev.numba_utils.evaluate_jit)
* [neural\_networks\_numba\_dev.optimizers\_jit](#neural_networks_numba_dev.optimizers_jit)
  * [CACHE](#neural_networks_numba_dev.optimizers_jit.CACHE)
  * [spec\_adam](#neural_networks_numba_dev.optimizers_jit.spec_adam)
  * [JITAdamOptimizer](#neural_networks_numba_dev.optimizers_jit.JITAdamOptimizer)
    * [\_\_init\_\_](#neural_networks_numba_dev.optimizers_jit.JITAdamOptimizer.__init__)
    * [initialize](#neural_networks_numba_dev.optimizers_jit.JITAdamOptimizer.initialize)
    * [update](#neural_networks_numba_dev.optimizers_jit.JITAdamOptimizer.update)
    * [update\_layers](#neural_networks_numba_dev.optimizers_jit.JITAdamOptimizer.update_layers)
  * [dense\_adam\_update\_layers](#neural_networks_numba_dev.optimizers_jit.dense_adam_update_layers)
  * [conv\_adam\_update\_layers](#neural_networks_numba_dev.optimizers_jit.conv_adam_update_layers)
  * [spec\_sgd](#neural_networks_numba_dev.optimizers_jit.spec_sgd)
  * [JITSGDOptimizer](#neural_networks_numba_dev.optimizers_jit.JITSGDOptimizer)
    * [\_\_init\_\_](#neural_networks_numba_dev.optimizers_jit.JITSGDOptimizer.__init__)
    * [initialize](#neural_networks_numba_dev.optimizers_jit.JITSGDOptimizer.initialize)
    * [update](#neural_networks_numba_dev.optimizers_jit.JITSGDOptimizer.update)
    * [update\_layers](#neural_networks_numba_dev.optimizers_jit.JITSGDOptimizer.update_layers)
  * [dense\_sgd\_update\_layers](#neural_networks_numba_dev.optimizers_jit.dense_sgd_update_layers)
  * [conv\_sgd\_update\_layers](#neural_networks_numba_dev.optimizers_jit.conv_sgd_update_layers)
  * [spec\_adadelta](#neural_networks_numba_dev.optimizers_jit.spec_adadelta)
  * [JITAdadeltaOptimizer](#neural_networks_numba_dev.optimizers_jit.JITAdadeltaOptimizer)
    * [\_\_init\_\_](#neural_networks_numba_dev.optimizers_jit.JITAdadeltaOptimizer.__init__)
    * [initialize](#neural_networks_numba_dev.optimizers_jit.JITAdadeltaOptimizer.initialize)
    * [update](#neural_networks_numba_dev.optimizers_jit.JITAdadeltaOptimizer.update)
    * [update\_layers](#neural_networks_numba_dev.optimizers_jit.JITAdadeltaOptimizer.update_layers)
  * [dense\_adadelta\_update\_layers](#neural_networks_numba_dev.optimizers_jit.dense_adadelta_update_layers)
  * [conv\_adadelta\_update\_layers](#neural_networks_numba_dev.optimizers_jit.conv_adadelta_update_layers)
* [neural\_networks\_numba\_dev.schedulers](#neural_networks_numba_dev.schedulers)
  * [lr\_scheduler\_step](#neural_networks_numba_dev.schedulers.lr_scheduler_step)
    * [\_\_init\_\_](#neural_networks_numba_dev.schedulers.lr_scheduler_step.__init__)
    * [\_\_repr\_\_](#neural_networks_numba_dev.schedulers.lr_scheduler_step.__repr__)
    * [step](#neural_networks_numba_dev.schedulers.lr_scheduler_step.step)
    * [reduce](#neural_networks_numba_dev.schedulers.lr_scheduler_step.reduce)
  * [lr\_scheduler\_exp](#neural_networks_numba_dev.schedulers.lr_scheduler_exp)
    * [\_\_init\_\_](#neural_networks_numba_dev.schedulers.lr_scheduler_exp.__init__)
    * [\_\_repr\_\_](#neural_networks_numba_dev.schedulers.lr_scheduler_exp.__repr__)
    * [step](#neural_networks_numba_dev.schedulers.lr_scheduler_exp.step)
    * [reduce](#neural_networks_numba_dev.schedulers.lr_scheduler_exp.reduce)
  * [lr\_scheduler\_plateau](#neural_networks_numba_dev.schedulers.lr_scheduler_plateau)
    * [\_\_init\_\_](#neural_networks_numba_dev.schedulers.lr_scheduler_plateau.__init__)
    * [\_\_repr\_\_](#neural_networks_numba_dev.schedulers.lr_scheduler_plateau.__repr__)
    * [step](#neural_networks_numba_dev.schedulers.lr_scheduler_plateau.step)
* [svm](#svm)
  * [\_\_all\_\_](#svm.__all__)
* [svm.baseSVM](#svm.baseSVM)
  * [BaseSVM](#svm.baseSVM.BaseSVM)
    * [\_\_init\_\_](#svm.baseSVM.BaseSVM.__init__)
    * [fit](#svm.baseSVM.BaseSVM.fit)
    * [\_fit](#svm.baseSVM.BaseSVM._fit)
    * [\_compute\_kernel](#svm.baseSVM.BaseSVM._compute_kernel)
    * [decision\_function](#svm.baseSVM.BaseSVM.decision_function)
    * [predict](#svm.baseSVM.BaseSVM.predict)
    * [score](#svm.baseSVM.BaseSVM.score)
    * [get\_params](#svm.baseSVM.BaseSVM.get_params)
    * [set\_params](#svm.baseSVM.BaseSVM.set_params)
    * [\_\_sklearn\_is\_fitted\_\_](#svm.baseSVM.BaseSVM.__sklearn_is_fitted__)
* [svm.generalizedSVM](#svm.generalizedSVM)
  * [GeneralizedSVR](#svm.generalizedSVM.GeneralizedSVR)
    * [\_\_init\_\_](#svm.generalizedSVM.GeneralizedSVR.__init__)
    * [\_fit](#svm.generalizedSVM.GeneralizedSVR._fit)
    * [predict](#svm.generalizedSVM.GeneralizedSVR.predict)
    * [decision\_function](#svm.generalizedSVM.GeneralizedSVR.decision_function)
    * [score](#svm.generalizedSVM.GeneralizedSVR.score)
  * [GeneralizedSVC](#svm.generalizedSVM.GeneralizedSVC)
    * [\_\_init\_\_](#svm.generalizedSVM.GeneralizedSVC.__init__)
    * [\_fit](#svm.generalizedSVM.GeneralizedSVC._fit)
    * [\_predict\_binary](#svm.generalizedSVM.GeneralizedSVC._predict_binary)
    * [\_predict\_multiclass](#svm.generalizedSVM.GeneralizedSVC._predict_multiclass)
    * [\_score\_binary](#svm.generalizedSVM.GeneralizedSVC._score_binary)
    * [\_score\_multiclass](#svm.generalizedSVM.GeneralizedSVC._score_multiclass)
    * [decision\_function](#svm.generalizedSVM.GeneralizedSVC.decision_function)
* [svm.linerarSVM](#svm.linerarSVM)
  * [LinearSVC](#svm.linerarSVM.LinearSVC)
    * [\_\_init\_\_](#svm.linerarSVM.LinearSVC.__init__)
    * [\_fit](#svm.linerarSVM.LinearSVC._fit)
    * [\_predict\_binary](#svm.linerarSVM.LinearSVC._predict_binary)
    * [\_predict\_multiclass](#svm.linerarSVM.LinearSVC._predict_multiclass)
    * [decision\_function](#svm.linerarSVM.LinearSVC.decision_function)
    * [\_score\_binary](#svm.linerarSVM.LinearSVC._score_binary)
    * [\_score\_multiclass](#svm.linerarSVM.LinearSVC._score_multiclass)
  * [LinearSVR](#svm.linerarSVM.LinearSVR)
    * [\_\_init\_\_](#svm.linerarSVM.LinearSVR.__init__)
    * [\_fit](#svm.linerarSVM.LinearSVR._fit)
    * [predict](#svm.linerarSVM.LinearSVR.predict)
    * [decision\_function](#svm.linerarSVM.LinearSVR.decision_function)
    * [score](#svm.linerarSVM.LinearSVR.score)
* [svm.oneClassSVM](#svm.oneClassSVM)
  * [OneClassSVM](#svm.oneClassSVM.OneClassSVM)
    * [\_\_init\_\_](#svm.oneClassSVM.OneClassSVM.__init__)
    * [\_fit](#svm.oneClassSVM.OneClassSVM._fit)
    * [decision\_function](#svm.oneClassSVM.OneClassSVM.decision_function)
    * [predict](#svm.oneClassSVM.OneClassSVM.predict)
    * [score](#svm.oneClassSVM.OneClassSVM.score)
    * [\_\_sklearn\_is\_fitted\_\_](#svm.oneClassSVM.OneClassSVM.__sklearn_is_fitted__)
* [svm.\_LinearSVM\_jit\_utils](#svm._LinearSVM_jit_utils)
  * [\_linearSVC\_minibatches](#svm._LinearSVM_jit_utils._linearSVC_minibatches)
  * [\_linearSVR\_minibatches](#svm._LinearSVM_jit_utils._linearSVR_minibatches)
* [time\_series](#time_series)
  * [\_\_all\_\_](#time_series.__all__)
* [time\_series.arima](#time_series.arima)
  * [ARIMA](#time_series.arima.ARIMA)
    * [\_\_init\_\_](#time_series.arima.ARIMA.__init__)
    * [\_\_name\_\_](#time_series.arima.ARIMA.__name__)
    * [\_\_str\_\_](#time_series.arima.ARIMA.__str__)
    * [fit](#time_series.arima.ARIMA.fit)
    * [forecast](#time_series.arima.ARIMA.forecast)
    * [\_compute\_residuals](#time_series.arima.ARIMA._compute_residuals)
    * [\_compute\_ar\_part](#time_series.arima.ARIMA._compute_ar_part)
    * [\_compute\_ma\_part](#time_series.arima.ARIMA._compute_ma_part)
    * [\_difference\_series](#time_series.arima.ARIMA._difference_series)
    * [\_fit\_ar\_model](#time_series.arima.ARIMA._fit_ar_model)
    * [\_fit\_ma\_model](#time_series.arima.ARIMA._fit_ma_model)
    * [\_combine\_ar\_ma](#time_series.arima.ARIMA._combine_ar_ma)
    * [\_forecast\_arima](#time_series.arima.ARIMA._forecast_arima)
    * [\_inverse\_difference](#time_series.arima.ARIMA._inverse_difference)
    * [suggest\_order](#time_series.arima.ARIMA.suggest_order)
    * [find\_best\_order](#time_series.arima.ARIMA.find_best_order)
  * [SARIMA](#time_series.arima.SARIMA)
    * [\_\_init\_\_](#time_series.arima.SARIMA.__init__)
    * [\_\_name\_\_](#time_series.arima.SARIMA.__name__)
    * [\_\_str\_\_](#time_series.arima.SARIMA.__str__)
    * [fit](#time_series.arima.SARIMA.fit)
    * [forecast](#time_series.arima.SARIMA.forecast)
    * [\_seasonal\_difference](#time_series.arima.SARIMA._seasonal_difference)
    * [\_inverse\_seasonal\_difference](#time_series.arima.SARIMA._inverse_seasonal_difference)
    * [suggest\_order](#time_series.arima.SARIMA.suggest_order)
    * [find\_best\_order](#time_series.arima.SARIMA.find_best_order)
  * [SARIMAX](#time_series.arima.SARIMAX)
    * [\_\_init\_\_](#time_series.arima.SARIMAX.__init__)
    * [\_\_name\_\_](#time_series.arima.SARIMAX.__name__)
    * [\_\_str\_\_](#time_series.arima.SARIMAX.__str__)
    * [fit](#time_series.arima.SARIMAX.fit)
    * [forecast](#time_series.arima.SARIMAX.forecast)
    * [suggest\_order](#time_series.arima.SARIMAX.suggest_order)
    * [find\_best\_order](#time_series.arima.SARIMAX.find_best_order)
* [time\_series.decomposition](#time_series.decomposition)
  * [\_centered\_moving\_average](#time_series.decomposition._centered_moving_average)
  * [AdditiveDecomposition](#time_series.decomposition.AdditiveDecomposition)
    * [\_\_init\_\_](#time_series.decomposition.AdditiveDecomposition.__init__)
    * [\_\_name\_\_](#time_series.decomposition.AdditiveDecomposition.__name__)
    * [\_\_str\_\_](#time_series.decomposition.AdditiveDecomposition.__str__)
    * [fit](#time_series.decomposition.AdditiveDecomposition.fit)
    * [get\_components](#time_series.decomposition.AdditiveDecomposition.get_components)
    * [reconstruct](#time_series.decomposition.AdditiveDecomposition.reconstruct)
  * [MultiplicativeDecomposition](#time_series.decomposition.MultiplicativeDecomposition)
    * [\_\_init\_\_](#time_series.decomposition.MultiplicativeDecomposition.__init__)
    * [\_\_name\_\_](#time_series.decomposition.MultiplicativeDecomposition.__name__)
    * [\_\_str\_\_](#time_series.decomposition.MultiplicativeDecomposition.__str__)
    * [fit](#time_series.decomposition.MultiplicativeDecomposition.fit)
    * [get\_components](#time_series.decomposition.MultiplicativeDecomposition.get_components)
    * [reconstruct](#time_series.decomposition.MultiplicativeDecomposition.reconstruct)
* [time\_series.exponential\_smoothing](#time_series.exponential_smoothing)
  * [mean\_squared\_error](#time_series.exponential_smoothing.mean_squared_error)
  * [SimpleExponentialSmoothing](#time_series.exponential_smoothing.SimpleExponentialSmoothing)
    * [\_\_init\_\_](#time_series.exponential_smoothing.SimpleExponentialSmoothing.__init__)
    * [\_\_name\_\_](#time_series.exponential_smoothing.SimpleExponentialSmoothing.__name__)
    * [\_\_str\_\_](#time_series.exponential_smoothing.SimpleExponentialSmoothing.__str__)
    * [fit](#time_series.exponential_smoothing.SimpleExponentialSmoothing.fit)
    * [forecast](#time_series.exponential_smoothing.SimpleExponentialSmoothing.forecast)
  * [DoubleExponentialSmoothing](#time_series.exponential_smoothing.DoubleExponentialSmoothing)
    * [\_\_init\_\_](#time_series.exponential_smoothing.DoubleExponentialSmoothing.__init__)
    * [\_\_name\_\_](#time_series.exponential_smoothing.DoubleExponentialSmoothing.__name__)
    * [\_\_str\_\_](#time_series.exponential_smoothing.DoubleExponentialSmoothing.__str__)
    * [fit](#time_series.exponential_smoothing.DoubleExponentialSmoothing.fit)
    * [forecast](#time_series.exponential_smoothing.DoubleExponentialSmoothing.forecast)
    * [find\_best\_alpha\_beta](#time_series.exponential_smoothing.DoubleExponentialSmoothing.find_best_alpha_beta)
  * [TripleExponentialSmoothing](#time_series.exponential_smoothing.TripleExponentialSmoothing)
    * [\_\_init\_\_](#time_series.exponential_smoothing.TripleExponentialSmoothing.__init__)
    * [\_\_name\_\_](#time_series.exponential_smoothing.TripleExponentialSmoothing.__name__)
    * [\_\_str\_\_](#time_series.exponential_smoothing.TripleExponentialSmoothing.__str__)
    * [\_initial\_seasonal\_components](#time_series.exponential_smoothing.TripleExponentialSmoothing._initial_seasonal_components)
    * [fit](#time_series.exponential_smoothing.TripleExponentialSmoothing.fit)
    * [forecast](#time_series.exponential_smoothing.TripleExponentialSmoothing.forecast)
    * [find\_best\_alpha\_beta\_gamma](#time_series.exponential_smoothing.TripleExponentialSmoothing.find_best_alpha_beta_gamma)
* [time\_series.forecasting](#time_series.forecasting)
  * [ForecastingPipeline](#time_series.forecasting.ForecastingPipeline)
    * [\_\_init\_\_](#time_series.forecasting.ForecastingPipeline.__init__)
    * [add\_preprocessor](#time_series.forecasting.ForecastingPipeline.add_preprocessor)
    * [remove\_preprocessor](#time_series.forecasting.ForecastingPipeline.remove_preprocessor)
    * [add\_evaluator](#time_series.forecasting.ForecastingPipeline.add_evaluator)
    * [remove\_evaluator](#time_series.forecasting.ForecastingPipeline.remove_evaluator)
    * [add\_model](#time_series.forecasting.ForecastingPipeline.add_model)
    * [remove\_model](#time_series.forecasting.ForecastingPipeline.remove_model)
    * [fit](#time_series.forecasting.ForecastingPipeline.fit)
    * [predict](#time_series.forecasting.ForecastingPipeline.predict)
    * [evaluate](#time_series.forecasting.ForecastingPipeline.evaluate)
    * [summary](#time_series.forecasting.ForecastingPipeline.summary)
* [time\_series.moving\_average](#time_series.moving_average)
  * [SimpleMovingAverage](#time_series.moving_average.SimpleMovingAverage)
    * [\_\_init\_\_](#time_series.moving_average.SimpleMovingAverage.__init__)
    * [\_\_name\_\_](#time_series.moving_average.SimpleMovingAverage.__name__)
    * [\_\_str\_\_](#time_series.moving_average.SimpleMovingAverage.__str__)
    * [fit](#time_series.moving_average.SimpleMovingAverage.fit)
    * [get\_smoothed](#time_series.moving_average.SimpleMovingAverage.get_smoothed)
    * [forecast](#time_series.moving_average.SimpleMovingAverage.forecast)
  * [WeightedMovingAverage](#time_series.moving_average.WeightedMovingAverage)
    * [\_\_init\_\_](#time_series.moving_average.WeightedMovingAverage.__init__)
    * [\_\_name\_\_](#time_series.moving_average.WeightedMovingAverage.__name__)
    * [\_\_str\_\_](#time_series.moving_average.WeightedMovingAverage.__str__)
    * [fit](#time_series.moving_average.WeightedMovingAverage.fit)
    * [get\_smoothed](#time_series.moving_average.WeightedMovingAverage.get_smoothed)
    * [forecast](#time_series.moving_average.WeightedMovingAverage.forecast)
  * [ExponentialMovingAverage](#time_series.moving_average.ExponentialMovingAverage)
    * [\_\_init\_\_](#time_series.moving_average.ExponentialMovingAverage.__init__)
    * [\_\_name\_\_](#time_series.moving_average.ExponentialMovingAverage.__name__)
    * [\_\_str\_\_](#time_series.moving_average.ExponentialMovingAverage.__str__)
    * [fit](#time_series.moving_average.ExponentialMovingAverage.fit)
    * [get\_smoothed](#time_series.moving_average.ExponentialMovingAverage.get_smoothed)
    * [forecast](#time_series.moving_average.ExponentialMovingAverage.forecast)
* [trees](#trees)
  * [\_\_all\_\_](#trees.__all__)
* [trees.adaBoostClassifier](#trees.adaBoostClassifier)
  * [AdaBoostClassifier](#trees.adaBoostClassifier.AdaBoostClassifier)
    * [\_\_init\_\_](#trees.adaBoostClassifier.AdaBoostClassifier.__init__)
    * [\_supports\_sample\_weight](#trees.adaBoostClassifier.AdaBoostClassifier._supports_sample_weight)
    * [\_fit](#trees.adaBoostClassifier.AdaBoostClassifier._fit)
    * [fit](#trees.adaBoostClassifier.AdaBoostClassifier.fit)
    * [decision\_function](#trees.adaBoostClassifier.AdaBoostClassifier.decision_function)
    * [predict\_proba](#trees.adaBoostClassifier.AdaBoostClassifier.predict_proba)
    * [predict](#trees.adaBoostClassifier.AdaBoostClassifier.predict)
    * [get\_stats](#trees.adaBoostClassifier.AdaBoostClassifier.get_stats)
    * [\_calculate\_metrics](#trees.adaBoostClassifier.AdaBoostClassifier._calculate_metrics)
* [trees.adaBoostRegressor](#trees.adaBoostRegressor)
  * [AdaBoostRegressor](#trees.adaBoostRegressor.AdaBoostRegressor)
    * [\_\_init\_\_](#trees.adaBoostRegressor.AdaBoostRegressor.__init__)
    * [\_supports\_sample\_weight](#trees.adaBoostRegressor.AdaBoostRegressor._supports_sample_weight)
    * [\_fit](#trees.adaBoostRegressor.AdaBoostRegressor._fit)
    * [fit](#trees.adaBoostRegressor.AdaBoostRegressor.fit)
    * [predict](#trees.adaBoostRegressor.AdaBoostRegressor.predict)
    * [get\_stats](#trees.adaBoostRegressor.AdaBoostRegressor.get_stats)
    * [\_calculate\_metrics](#trees.adaBoostRegressor.AdaBoostRegressor._calculate_metrics)
* [trees.gradientBoostedClassifier](#trees.gradientBoostedClassifier)
  * [GradientBoostedClassifier](#trees.gradientBoostedClassifier.GradientBoostedClassifier)
    * [\_\_init\_\_](#trees.gradientBoostedClassifier.GradientBoostedClassifier.__init__)
    * [get\_params](#trees.gradientBoostedClassifier.GradientBoostedClassifier.get_params)
    * [\_validate\_input](#trees.gradientBoostedClassifier.GradientBoostedClassifier._validate_input)
    * [\_init\_predict](#trees.gradientBoostedClassifier.GradientBoostedClassifier._init_predict)
    * [fit](#trees.gradientBoostedClassifier.GradientBoostedClassifier.fit)
    * [decision\_function](#trees.gradientBoostedClassifier.GradientBoostedClassifier.decision_function)
    * [predict\_proba](#trees.gradientBoostedClassifier.GradientBoostedClassifier.predict_proba)
    * [predict](#trees.gradientBoostedClassifier.GradientBoostedClassifier.predict)
    * [calculate\_metrics](#trees.gradientBoostedClassifier.GradientBoostedClassifier.calculate_metrics)
    * [get\_stats](#trees.gradientBoostedClassifier.GradientBoostedClassifier.get_stats)
* [trees.gradientBoostedRegressor](#trees.gradientBoostedRegressor)
  * [GradientBoostedRegressor](#trees.gradientBoostedRegressor.GradientBoostedRegressor)
    * [\_\_init\_\_](#trees.gradientBoostedRegressor.GradientBoostedRegressor.__init__)
    * [get\_params](#trees.gradientBoostedRegressor.GradientBoostedRegressor.get_params)
    * [fit](#trees.gradientBoostedRegressor.GradientBoostedRegressor.fit)
    * [predict](#trees.gradientBoostedRegressor.GradientBoostedRegressor.predict)
    * [calculate\_metrics](#trees.gradientBoostedRegressor.GradientBoostedRegressor.calculate_metrics)
    * [get\_stats](#trees.gradientBoostedRegressor.GradientBoostedRegressor.get_stats)
* [trees.isolationForest](#trees.isolationForest)
  * [IsolationUtils](#trees.isolationForest.IsolationUtils)
    * [compute\_avg\_path\_length](#trees.isolationForest.IsolationUtils.compute_avg_path_length)
  * [IsolationTree](#trees.isolationForest.IsolationTree)
    * [\_\_init\_\_](#trees.isolationForest.IsolationTree.__init__)
    * [fit](#trees.isolationForest.IsolationTree.fit)
    * [path\_length](#trees.isolationForest.IsolationTree.path_length)
  * [IsolationForest](#trees.isolationForest.IsolationForest)
    * [\_\_init\_\_](#trees.isolationForest.IsolationForest.__init__)
    * [fit](#trees.isolationForest.IsolationForest.fit)
    * [\_fit\_tree](#trees.isolationForest.IsolationForest._fit_tree)
    * [anomaly\_score](#trees.isolationForest.IsolationForest.anomaly_score)
    * [predict](#trees.isolationForest.IsolationForest.predict)
    * [\_\_sklearn\_is\_fitted\_\_](#trees.isolationForest.IsolationForest.__sklearn_is_fitted__)
* [trees.randomForestClassifier](#trees.randomForestClassifier)
  * [\_fit\_tree](#trees.randomForestClassifier._fit_tree)
  * [\_classify\_oob](#trees.randomForestClassifier._classify_oob)
  * [RandomForestClassifier](#trees.randomForestClassifier.RandomForestClassifier)
    * [\_\_init\_\_](#trees.randomForestClassifier.RandomForestClassifier.__init__)
    * [get\_params](#trees.randomForestClassifier.RandomForestClassifier.get_params)
    * [fit](#trees.randomForestClassifier.RandomForestClassifier.fit)
    * [calculate\_metrics](#trees.randomForestClassifier.RandomForestClassifier.calculate_metrics)
    * [predict](#trees.randomForestClassifier.RandomForestClassifier.predict)
    * [predict\_proba](#trees.randomForestClassifier.RandomForestClassifier.predict_proba)
    * [get\_stats](#trees.randomForestClassifier.RandomForestClassifier.get_stats)
* [trees.randomForestRegressor](#trees.randomForestRegressor)
  * [\_fit\_single\_tree](#trees.randomForestRegressor._fit_single_tree)
  * [RandomForestRegressor](#trees.randomForestRegressor.RandomForestRegressor)
    * [\_\_init\_\_](#trees.randomForestRegressor.RandomForestRegressor.__init__)
    * [get\_params](#trees.randomForestRegressor.RandomForestRegressor.get_params)
    * [fit](#trees.randomForestRegressor.RandomForestRegressor.fit)
    * [predict](#trees.randomForestRegressor.RandomForestRegressor.predict)
    * [get\_stats](#trees.randomForestRegressor.RandomForestRegressor.get_stats)
    * [calculate\_metrics](#trees.randomForestRegressor.RandomForestRegressor.calculate_metrics)
* [trees.treeClassifier](#trees.treeClassifier)
  * [ClassifierTreeUtility](#trees.treeClassifier.ClassifierTreeUtility)
    * [\_\_init\_\_](#trees.treeClassifier.ClassifierTreeUtility.__init__)
    * [entropy](#trees.treeClassifier.ClassifierTreeUtility.entropy)
    * [partition\_classes](#trees.treeClassifier.ClassifierTreeUtility.partition_classes)
    * [information\_gain](#trees.treeClassifier.ClassifierTreeUtility.information_gain)
    * [best\_split](#trees.treeClassifier.ClassifierTreeUtility.best_split)
  * [ClassifierTree](#trees.treeClassifier.ClassifierTree)
    * [\_\_init\_\_](#trees.treeClassifier.ClassifierTree.__init__)
    * [fit](#trees.treeClassifier.ClassifierTree.fit)
    * [learn](#trees.treeClassifier.ClassifierTree.learn)
    * [classify](#trees.treeClassifier.ClassifierTree.classify)
    * [predict](#trees.treeClassifier.ClassifierTree.predict)
    * [predict\_proba](#trees.treeClassifier.ClassifierTree.predict_proba)
* [trees.treeRegressor](#trees.treeRegressor)
  * [RegressorTreeUtility](#trees.treeRegressor.RegressorTreeUtility)
    * [\_\_init\_\_](#trees.treeRegressor.RegressorTreeUtility.__init__)
    * [calculate\_variance](#trees.treeRegressor.RegressorTreeUtility.calculate_variance)
    * [calculate\_leaf\_value](#trees.treeRegressor.RegressorTreeUtility.calculate_leaf_value)
    * [best\_split](#trees.treeRegressor.RegressorTreeUtility.best_split)
  * [RegressorTree](#trees.treeRegressor.RegressorTree)
    * [\_\_init\_\_](#trees.treeRegressor.RegressorTree.__init__)
    * [fit](#trees.treeRegressor.RegressorTree.fit)
    * [predict](#trees.treeRegressor.RegressorTree.predict)
    * [\_traverse\_tree](#trees.treeRegressor.RegressorTree._traverse_tree)
    * [\_learn\_recursive](#trees.treeRegressor.RegressorTree._learn_recursive)
* [utils](#utils)
  * [\_\_all\_\_](#utils.__all__)
* [utils.animator](#utils.animator)
  * [AnimationBase](#utils.animator.AnimationBase)
    * [\_\_init\_\_](#utils.animator.AnimationBase.__init__)
    * [setup\_plot](#utils.animator.AnimationBase.setup_plot)
    * [update\_model](#utils.animator.AnimationBase.update_model)
    * [update\_plot](#utils.animator.AnimationBase.update_plot)
    * [animate](#utils.animator.AnimationBase.animate)
    * [save](#utils.animator.AnimationBase.save)
    * [show](#utils.animator.AnimationBase.show)
  * [ForcastingAnimation](#utils.animator.ForcastingAnimation)
    * [\_\_init\_\_](#utils.animator.ForcastingAnimation.__init__)
    * [setup\_plot](#utils.animator.ForcastingAnimation.setup_plot)
    * [update\_model](#utils.animator.ForcastingAnimation.update_model)
    * [update\_plot](#utils.animator.ForcastingAnimation.update_plot)
  * [RegressionAnimation](#utils.animator.RegressionAnimation)
    * [\_\_init\_\_](#utils.animator.RegressionAnimation.__init__)
    * [setup\_plot](#utils.animator.RegressionAnimation.setup_plot)
    * [update\_model](#utils.animator.RegressionAnimation.update_model)
    * [update\_plot](#utils.animator.RegressionAnimation.update_plot)
  * [ClassificationAnimation](#utils.animator.ClassificationAnimation)
    * [\_\_init\_\_](#utils.animator.ClassificationAnimation.__init__)
    * [setup\_plot](#utils.animator.ClassificationAnimation.setup_plot)
    * [update\_model](#utils.animator.ClassificationAnimation.update_model)
    * [update\_plot](#utils.animator.ClassificationAnimation.update_plot)
* [utils.dataAugmentation](#utils.dataAugmentation)
  * [\_Utils](#utils.dataAugmentation._Utils)
    * [check\_class\_balance](#utils.dataAugmentation._Utils.check_class_balance)
    * [separate\_samples](#utils.dataAugmentation._Utils.separate_samples)
    * [get\_class\_distribution](#utils.dataAugmentation._Utils.get_class_distribution)
    * [get\_minority\_majority\_classes](#utils.dataAugmentation._Utils.get_minority_majority_classes)
    * [validate\_Xy](#utils.dataAugmentation._Utils.validate_Xy)
  * [SMOTE](#utils.dataAugmentation.SMOTE)
    * [\_\_init\_\_](#utils.dataAugmentation.SMOTE.__init__)
    * [fit\_resample](#utils.dataAugmentation.SMOTE.fit_resample)
  * [RandomOverSampler](#utils.dataAugmentation.RandomOverSampler)
    * [\_\_init\_\_](#utils.dataAugmentation.RandomOverSampler.__init__)
    * [fit\_resample](#utils.dataAugmentation.RandomOverSampler.fit_resample)
  * [RandomUnderSampler](#utils.dataAugmentation.RandomUnderSampler)
    * [\_\_init\_\_](#utils.dataAugmentation.RandomUnderSampler.__init__)
    * [fit\_resample](#utils.dataAugmentation.RandomUnderSampler.fit_resample)
  * [Augmenter](#utils.dataAugmentation.Augmenter)
    * [\_\_init\_\_](#utils.dataAugmentation.Augmenter.__init__)
    * [augment](#utils.dataAugmentation.Augmenter.augment)
* [utils.dataPrep](#utils.dataPrep)
  * [DataPrep](#utils.dataPrep.DataPrep)
    * [one\_hot\_encode](#utils.dataPrep.DataPrep.one_hot_encode)
    * [find\_categorical\_columns](#utils.dataPrep.DataPrep.find_categorical_columns)
    * [write\_data](#utils.dataPrep.DataPrep.write_data)
    * [prepare\_data](#utils.dataPrep.DataPrep.prepare_data)
    * [df\_to\_ndarray](#utils.dataPrep.DataPrep.df_to_ndarray)
    * [k\_split](#utils.dataPrep.DataPrep.k_split)
* [utils.dataPreprocessing](#utils.dataPreprocessing)
  * [one\_hot\_encode](#utils.dataPreprocessing.one_hot_encode)
  * [\_find\_categorical\_columns](#utils.dataPreprocessing._find_categorical_columns)
  * [normalize](#utils.dataPreprocessing.normalize)
  * [Scaler](#utils.dataPreprocessing.Scaler)
    * [\_\_init\_\_](#utils.dataPreprocessing.Scaler.__init__)
    * [fit](#utils.dataPreprocessing.Scaler.fit)
    * [transform](#utils.dataPreprocessing.Scaler.transform)
    * [fit\_transform](#utils.dataPreprocessing.Scaler.fit_transform)
    * [inverse\_transform](#utils.dataPreprocessing.Scaler.inverse_transform)
* [utils.dataSplitting](#utils.dataSplitting)
  * [train\_test\_split](#utils.dataSplitting.train_test_split)
* [utils.decomposition](#utils.decomposition)
  * [PCA](#utils.decomposition.PCA)
    * [\_\_init\_\_](#utils.decomposition.PCA.__init__)
    * [fit](#utils.decomposition.PCA.fit)
    * [transform](#utils.decomposition.PCA.transform)
    * [fit\_transform](#utils.decomposition.PCA.fit_transform)
    * [get\_explained\_variance\_ratio](#utils.decomposition.PCA.get_explained_variance_ratio)
    * [get\_components](#utils.decomposition.PCA.get_components)
    * [inverse\_transform](#utils.decomposition.PCA.inverse_transform)
  * [SVD](#utils.decomposition.SVD)
    * [\_\_init\_\_](#utils.decomposition.SVD.__init__)
    * [fit](#utils.decomposition.SVD.fit)
    * [transform](#utils.decomposition.SVD.transform)
    * [fit\_transform](#utils.decomposition.SVD.fit_transform)
    * [get\_singular\_values](#utils.decomposition.SVD.get_singular_values)
    * [get\_singular\_vectors](#utils.decomposition.SVD.get_singular_vectors)
* [utils.makeData](#utils.makeData)
  * [make\_regression](#utils.makeData.make_regression)
  * [make\_classification](#utils.makeData.make_classification)
  * [make\_blobs](#utils.makeData.make_blobs)
  * [make\_time\_series](#utils.makeData.make_time_series)
* [utils.metrics](#utils.metrics)
  * [Metrics](#utils.metrics.Metrics)
    * [mean\_squared\_error](#utils.metrics.Metrics.mean_squared_error)
    * [r\_squared](#utils.metrics.Metrics.r_squared)
    * [mean\_absolute\_error](#utils.metrics.Metrics.mean_absolute_error)
    * [root\_mean\_squared\_error](#utils.metrics.Metrics.root_mean_squared_error)
    * [mean\_absolute\_percentage\_error](#utils.metrics.Metrics.mean_absolute_percentage_error)
    * [mean\_percentage\_error](#utils.metrics.Metrics.mean_percentage_error)
    * [accuracy](#utils.metrics.Metrics.accuracy)
    * [precision](#utils.metrics.Metrics.precision)
    * [recall](#utils.metrics.Metrics.recall)
    * [f1\_score](#utils.metrics.Metrics.f1_score)
    * [log\_loss](#utils.metrics.Metrics.log_loss)
    * [confusion\_matrix](#utils.metrics.Metrics.confusion_matrix)
    * [show\_confusion\_matrix](#utils.metrics.Metrics.show_confusion_matrix)
    * [classification\_report](#utils.metrics.Metrics.classification_report)
    * [show\_classification\_report](#utils.metrics.Metrics.show_classification_report)
* [utils.modelSelection](#utils.modelSelection)
  * [ModelSelectionUtility](#utils.modelSelection.ModelSelectionUtility)
    * [get\_param\_combinations](#utils.modelSelection.ModelSelectionUtility.get_param_combinations)
    * [cross\_validate](#utils.modelSelection.ModelSelectionUtility.cross_validate)
  * [GridSearchCV](#utils.modelSelection.GridSearchCV)
    * [\_\_init\_\_](#utils.modelSelection.GridSearchCV.__init__)
    * [fit](#utils.modelSelection.GridSearchCV.fit)
  * [RandomSearchCV](#utils.modelSelection.RandomSearchCV)
    * [\_\_init\_\_](#utils.modelSelection.RandomSearchCV.__init__)
    * [fit](#utils.modelSelection.RandomSearchCV.fit)
  * [segaSearchCV](#utils.modelSelection.segaSearchCV)
    * [\_\_init\_\_](#utils.modelSelection.segaSearchCV.__init__)
    * [fit](#utils.modelSelection.segaSearchCV.fit)
* [utils.polynomialTransform](#utils.polynomialTransform)
  * [PolynomialTransform](#utils.polynomialTransform.PolynomialTransform)
    * [\_\_init\_\_](#utils.polynomialTransform.PolynomialTransform.__init__)
    * [fit](#utils.polynomialTransform.PolynomialTransform.fit)
    * [transform](#utils.polynomialTransform.PolynomialTransform.transform)
    * [fit\_transform](#utils.polynomialTransform.PolynomialTransform.fit_transform)
* [utils.voting](#utils.voting)
  * [VotingRegressor](#utils.voting.VotingRegressor)
    * [\_\_init\_\_](#utils.voting.VotingRegressor.__init__)
    * [predict](#utils.voting.VotingRegressor.predict)
    * [get\_params](#utils.voting.VotingRegressor.get_params)
    * [show\_models](#utils.voting.VotingRegressor.show_models)
  * [VotingClassifier](#utils.voting.VotingClassifier)
    * [\_\_init\_\_](#utils.voting.VotingClassifier.__init__)
    * [predict](#utils.voting.VotingClassifier.predict)
    * [get\_params](#utils.voting.VotingClassifier.get_params)
    * [show\_models](#utils.voting.VotingClassifier.show_models)
  * [ForecastRegressor](#utils.voting.ForecastRegressor)
    * [\_\_init\_\_](#utils.voting.ForecastRegressor.__init__)
    * [forecast](#utils.voting.ForecastRegressor.forecast)
    * [get\_params](#utils.voting.ForecastRegressor.get_params)
    * [show\_models](#utils.voting.ForecastRegressor.show_models)

<a id="__init__"></a>

# \_\_init\_\_

<a id="__init__.__all__"></a>

#### \_\_all\_\_

<a id="auto"></a>

# auto

<a id="auto.__all__"></a>

#### \_\_all\_\_

<a id="auto.classifier"></a>

# auto.classifier

<a id="auto.classifier.accuracy"></a>

#### accuracy

<a id="auto.classifier.precision"></a>

#### precision

<a id="auto.classifier.recall"></a>

#### recall

<a id="auto.classifier.f1"></a>

#### f1

<a id="auto.classifier.AutoClassifier"></a>

## AutoClassifier Objects

```python
class AutoClassifier()
```

A class to automatically select and evaluate the best classification model.

Includes optional automated hyperparameter tuning using GridSearchCV or RandomSearchCV.

<a id="auto.classifier.AutoClassifier.__init__"></a>

#### \_\_init\_\_

```python
def __init__(all_kernels=False,
             tune_hyperparameters=False,
             tuning_method="random",
             tuning_iterations=10,
             cv=3,
             tuning_metric="f1")
```

Initializes the AutoClassifier.

Args:
    all_kernels (bool): If True, include all SVM kernels. Default False.
    tune_hyperparameters (bool): If True, perform hyperparameter tuning. Default False.
    tuning_method (str): Method for tuning ('random' or 'grid'). Default 'random'.
    tuning_iterations (int): Number of iterations for Random Search. Default 10.
    cv (int): Number of cross-validation folds for tuning. Default 3.
    tuning_metric (str): Metric to optimize ('accuracy', 'precision', 'recall', 'f1'). Default 'f1'.

<a id="auto.classifier.AutoClassifier.fit"></a>

#### fit

```python
def fit(X_train,
        y_train,
        X_test=None,
        y_test=None,
        custom_metrics=None,
        verbose=False)
```

Fits the classification models, optionally performing hyperparameter tuning.

Args:
    X_train: (np.ndarray) - Training feature data.
    y_train: (np.ndarray) - Training target data.
    X_test: (np.ndarray), optional - Testing feature data. Default None.
    y_test: (np.ndarray), optional - Testing target data. Default None.
    custom_metrics: (dict: str -> callable), optional - Custom metrics for evaluation.
    verbose: (bool), optional - If True, prints progress. Default False.

Returns:
    results: (list) - A list of dictionaries containing model performance metrics.
    predictions: (dict) - A dictionary of predictions for each model on the test/train set.

<a id="auto.classifier.AutoClassifier.predict"></a>

#### predict

```python
def predict(X, model=None)
```

Generates predictions using fitted models.

Args:
    X: (np.ndarray) - Input feature data.
    model: (str), optional - Specific model name. Default None (predict with all).

Returns:
    dict or np.ndarray: Predictions for specified model(s).

<a id="auto.classifier.AutoClassifier.evaluate"></a>

#### evaluate

```python
def evaluate(y_true, custom_metrics=None, model=None)
```

Evaluates the performance using stored predictions.

Args:
    y_true: (np.ndarray) - True target values.
    custom_metrics: (dict), optional - Custom metrics. Default None.
    model: (str), optional - Specific model name. Default None (evaluate all).

Returns:
    dict: Evaluation metrics for the specified model(s).

<a id="auto.classifier.AutoClassifier.get_model"></a>

#### get\_model

```python
def get_model(model_name)
```

Returns the final fitted model instance (potentially tuned).

Args:
    model_name (str): The name of the model.

Returns:
    model_instance: The fitted model instance.

<a id="auto.classifier.AutoClassifier.summary"></a>

#### summary

```python
def summary()
```

Prints a summary of model performance, including tuning results if available.

<a id="auto.regressor"></a>

# auto.regressor

<a id="auto.regressor.r_squared"></a>

#### r\_squared

<a id="auto.regressor.root_mean_squared_error"></a>

#### root\_mean\_squared\_error

<a id="auto.regressor.mean_absolute_percentage_error"></a>

#### mean\_absolute\_percentage\_error

<a id="auto.regressor.AutoRegressor"></a>

## AutoRegressor Objects

```python
class AutoRegressor()
```

A class to automatically select and evaluate the best regression model.

Includes optional automated hyperparameter tuning using GridSearchCV or RandomSearchCV.

<a id="auto.regressor.AutoRegressor.__init__"></a>

#### \_\_init\_\_

```python
def __init__(all_kernels=False,
             tune_hyperparameters=False,
             tuning_method="random",
             tuning_iterations=10,
             cv=3,
             tuning_metric="r2")
```

Initializes the AutoRegressor.

Args:
    all_kernels (bool): If True, include all SVM kernels. Default False.
    tune_hyperparameters (bool): If True, perform hyperparameter tuning. Default False.
    tuning_method (str): Method for tuning ('random' or 'grid'). Default 'random'.
    tuning_iterations (int): Number of iterations for Random Search. Default 10.
    cv (int): Number of cross-validation folds for tuning. Default 3.
    tuning_metric (str): Metric to optimize ('r2', 'neg_mean_squared_error', 'rmse', 'mae', 'mape'). Default 'r2'.
                       Note: for minimization use 'neg_mean_squared_error', 'rmse', 'mae', 'mape'.

<a id="auto.regressor.AutoRegressor.fit"></a>

#### fit

```python
def fit(X_train,
        y_train,
        X_test=None,
        y_test=None,
        custom_metrics=None,
        verbose=False)
```

Fits the regression models, optionally performing hyperparameter tuning.

Args:
    X_train: (np.ndarray) - Training feature data.
    y_train: (np.ndarray) - Training target data.
    X_test: (np.ndarray), optional - Testing feature data. Default None.
    y_test: (np.ndarray), optional - Testing target data. Default None.
    custom_metrics: (dict: str -> callable), optional - Custom metrics.
    verbose: (bool), optional - If True, prints progress. Default False.

Returns:
    results: (list) - Performance metrics for each model.
    predictions: (dict) - Predictions for each model on the test/train set.

<a id="auto.regressor.AutoRegressor.predict"></a>

#### predict

```python
def predict(X, model=None)
```

Generates predictions using fitted models.

Args:
    X: (np.ndarray) - Input feature data.
    model: (str), optional - Specific model name. Default None (predict with all).

Returns:
    dict or np.ndarray: Predictions for specified model(s).

<a id="auto.regressor.AutoRegressor.evaluate"></a>

#### evaluate

```python
def evaluate(y_true, custom_metrics=None, model=None)
```

Evaluates performance using stored predictions.

Args:
    y_true: (np.ndarray) - True target values.
    custom_metrics: (dict), optional - Custom metrics. Default None.
    model: (str), optional - Specific model name. Default None (evaluate all).

Returns:
    dict: Evaluation metrics.

<a id="auto.regressor.AutoRegressor.get_model"></a>

#### get\_model

```python
def get_model(model_name)
```

Returns the final fitted model instance (potentially tuned).

<a id="auto.regressor.AutoRegressor.summary"></a>

#### summary

```python
def summary()
```

Prints a summary of model performance, including tuning results.

<a id="clustering"></a>

# clustering

<a id="clustering.__all__"></a>

#### \_\_all\_\_

<a id="clustering.clustering"></a>

# clustering.clustering

<a id="clustering.clustering.KMeans"></a>

## KMeans Objects

```python
class KMeans()
```

This class implements the K-Means clustering algorithm along with methods for evaluating the optimal number of clusters and visualizing the clustering results.

Args:
    X: The data matrix (numpy array).
    n_clusters: The number of clusters.
    max_iter: The maximum number of iterations.
    tol: The tolerance to declare convergence.

Methods:
    - __init__: Initializes the KMeans object with parameters such as the data matrix, number of clusters, maximum iterations, and convergence tolerance.
    - _handle_categorical: Handles categorical columns in the input data by one-hot encoding.
    - _convert_to_ndarray: Converts input data to a NumPy ndarray and handles categorical columns.
    - initialize_centroids: Randomly initializes the centroids for KMeans clustering.
    - assign_clusters: Assigns clusters based on the nearest centroid.
    - update_centroids: Updates centroids based on the current cluster assignments.
    - fit: Fits the KMeans model to the data by iteratively updating centroids and cluster assignments until convergence.
    - predict: Predicts the closest cluster each sample in new_X belongs to.
    - elbow_method: Implements the elbow method to determine the optimal number of clusters.
    - calinski_harabasz_index: Calculates the Calinski-Harabasz Index for evaluating clustering performance.
    - davies_bouldin_index: Calculates the Davies-Bouldin Index for evaluating clustering performance.
    - silhouette_score: Calculates the Silhouette Score for evaluating clustering performance.
    - find_optimal_clusters: Implements methods to find the optimal number of clusters using the elbow method,
                         Calinski-Harabasz Index, Davies-Bouldin Index, and Silhouette Score. It also plots the evaluation
                         metrics to aid in determining the optimal k value.

<a id="clustering.clustering.KMeans.__init__"></a>

#### \_\_init\_\_

```python
def __init__(X, n_clusters=3, max_iter=300, tol=1e-4)
```

Initialize the KMeans object.

Args:
    X: The data matrix (numpy array, pandas DataFrame, or list).
    n_clusters: The number of clusters.
    max_iter: The maximum number of iterations.
    tol: The tolerance to declare convergence.

<a id="clustering.clustering.KMeans._handle_categorical"></a>

#### \_handle\_categorical

```python
def _handle_categorical(X)
```

Handle categorical columns by one-hot encoding.

Args:
    X: The input data with potential categorical columns.

Returns:
    X_processed: The processed data with categorical columns encoded.

<a id="clustering.clustering.KMeans._convert_to_ndarray"></a>

#### \_convert\_to\_ndarray

```python
def _convert_to_ndarray(X)
```

Convert input data to a NumPy ndarray and handle categorical columns.

Args:
    X: The input data, which can be a list, DataFrame, or ndarray.

Returns:
    X_ndarray: The converted and processed input data as a NumPy ndarray.

<a id="clustering.clustering.KMeans.initialize_centroids"></a>

#### initialize\_centroids

```python
def initialize_centroids()
```

Randomly initialize the centroids.

Returns:
    centroids: The initialized centroids.

<a id="clustering.clustering.KMeans.assign_clusters"></a>

#### assign\_clusters

```python
def assign_clusters(centroids)
```

Assign clusters based on the nearest centroid.

Args:
    centroids: The current centroids.

Returns:
    labels: The cluster assignments for each data point.

<a id="clustering.clustering.KMeans.update_centroids"></a>

#### update\_centroids

```python
def update_centroids()
```

Update the centroids based on the current cluster assignments.

Returns:
    centroids: The updated centroids.

<a id="clustering.clustering.KMeans.fit"></a>

#### fit

```python
def fit()
```

Fit the KMeans model to the data.

<a id="clustering.clustering.KMeans.predict"></a>

#### predict

```python
def predict(new_X)
```

Predict the closest cluster each sample in new_X belongs to.

Args:
    new_X: The data matrix to predict (numpy array).

Returns:
    labels: The predicted cluster labels.

<a id="clustering.clustering.KMeans.elbow_method"></a>

#### elbow\_method

```python
def elbow_method(max_k=10)
```

Implement the elbow method to determine the optimal number of clusters.

Args:
    max_k: The maximum number of clusters to test.

Returns:
    distortions: A list of distortions for each k.

<a id="clustering.clustering.KMeans.calinski_harabasz_index"></a>

#### calinski\_harabasz\_index

```python
def calinski_harabasz_index(X, labels, centroids)
```

Calculate the Calinski-Harabasz Index for evaluating clustering performance.

Args:
    X: The data matrix (numpy array).
    labels: The cluster labels for each data point.
    centroids: The centroids of the clusters.

Returns:
    ch_index: The computed Calinski-Harabasz Index.

<a id="clustering.clustering.KMeans.davies_bouldin_index"></a>

#### davies\_bouldin\_index

```python
def davies_bouldin_index(X, labels, centroids)
```

Calculate the Davies-Bouldin Index for evaluating clustering performance.

Args:
    X: The data matrix (numpy array).
    labels: The cluster labels for each data point.
    centroids: The centroids of the clusters.

Returns:
    db_index: The computed Davies-Bouldin Index.

<a id="clustering.clustering.KMeans.silhouette_score"></a>

#### silhouette\_score

```python
def silhouette_score(X, labels)
```

Calculate the silhouette score for evaluating clustering performance.

Args:
    X: The data matrix (numpy array).
    labels: The cluster labels for each data point.

Returns:
    silhouette_score: The computed silhouette score.

<a id="clustering.clustering.KMeans.find_optimal_clusters"></a>

#### find\_optimal\_clusters

```python
def find_optimal_clusters(max_k=10, true_k=None, save_dir=None)
```

Find the optimal number of clusters using various evaluation metrics and plot the results.

Args:
    X: The data matrix (numpy array).
    max_k: The maximum number of clusters to consider.
    true_k: The true number of clusters in the data.
    save_dir: The directory to save the plot (optional).

Returns:
    ch_optimal_k: The optimal number of clusters based on the Calinski-Harabasz Index.
    db_optimal_k: The optimal number of clusters based on the Davies-Bouldin Index.
    silhouette_optimal_k: The optimal number of clusters based on the Silhouette Score.

<a id="clustering.clustering.DBSCAN"></a>

## DBSCAN Objects

```python
class DBSCAN()
```

This class implements the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm.

Args:
    X: The data matrix (numpy array).
    eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

Methods:
    - __init__: Initializes the DBSCAN object with the input parameters.
    - fit: Fits the DBSCAN model to the data and assigns cluster labels.
    - predict: Predicts the cluster labels for new data points.
    - fit_predict: Fits the DBSCAN model and returns cluster labels.
    - silhouette_score: Calculates the Silhouette Score for evaluating clustering performance.
    - _handle_categorical: Handles categorical columns by one-hot encoding.
    - _convert_to_ndarray: Converts input data to a NumPy ndarray and handles categorical columns.
    - _custom_distance_matrix: Calculates the pairwise distance matrix using a custom distance calculation method.

<a id="clustering.clustering.DBSCAN.__init__"></a>

#### \_\_init\_\_

```python
def __init__(X, eps=0.5, min_samples=5, compile_numba=False)
```

Initialize the DBSCAN object.

Args:
    X: The data matrix (numpy array).
    eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
    compile_numba: Whether to compile the distance calculations using Numba for performance.
    If not compiled, the first call to the numba fitting function will take longer, but subsequent calls will be faster.

<a id="clustering.clustering.DBSCAN._handle_categorical"></a>

#### \_handle\_categorical

```python
def _handle_categorical(X)
```

Handle categorical columns by one-hot encoding.

Args:
    X: The input data with potential categorical columns.

Returns:
    X_processed: The processed data with categorical columns encoded.

<a id="clustering.clustering.DBSCAN._convert_to_ndarray"></a>

#### \_convert\_to\_ndarray

```python
def _convert_to_ndarray(X)
```

Convert input data to a NumPy ndarray and handle categorical columns.

Args:
    X: The input data, which can be a list, DataFrame, or ndarray.

Returns:
    X_ndarray: The converted and processed input data as a NumPy ndarray.

<a id="clustering.clustering.DBSCAN._custom_distance_matrix"></a>

#### \_custom\_distance\_matrix

```python
def _custom_distance_matrix(X1, X2, metric="euclidean")
```

Calculate the pairwise distance matrix between two sets of data points using a custom distance calculation method.

Args:
    X1: The first data matrix (numpy array).
    X2: The second data matrix (numpy array).
    metric: The distance metric to use ('euclidean', 'manhattan', or 'cosine').

Returns:
    dist_matrix: The pairwise distance matrix between data points in X1 and X2.

<a id="clustering.clustering.DBSCAN.fit"></a>

#### fit

```python
def fit(metric="euclidean", numba=False)
```

Fit the DBSCAN model to the data.

Algorithm Steps:
1. Calculate the distance matrix between all points in the dataset.
2. Identify core points based on the minimum number of neighbors within eps distance.
3. Assign cluster labels using depth-first search (DFS) starting from core points.

Args:
    metric: The distance metric to use ('euclidean', 'manhattan', or 'cosine').
    numba: Whether to use numba for faster computation.

Returns:
    labels: The cluster labels for each data point.

<a id="clustering.clustering.DBSCAN.predict"></a>

#### predict

```python
def predict(new_X)
```

Predict the cluster labels for new data points.

Note: DBSCAN does not naturally support predicting new data points.

Args:
    new_X: The data matrix to predict (numpy array).

Returns:
    labels: The predicted cluster labels (-1 for noise).

<a id="clustering.clustering.DBSCAN.fit_predict"></a>

#### fit\_predict

```python
def fit_predict(numba=False)
```

Fit the DBSCAN model to the data and return the cluster labels.

Returns:
    labels: The cluster labels for the data.

<a id="clustering.clustering.DBSCAN.silhouette_score"></a>

#### silhouette\_score

```python
def silhouette_score()
```

Calculate the silhouette score for evaluating clustering performance.

Returns:
    silhouette_score: The computed silhouette score.

<a id="clustering.clustering.DBSCAN.auto_eps"></a>

#### auto\_eps

```python
def auto_eps(min=0.1,
             max=1.1,
             precision=0.01,
             return_scores=False,
             verbose=False)
```

Find the optimal eps value for DBSCAN based on silhouette score.

Args:
    min: The minimum eps value to start the search.
    max: The maximum eps value to end the search.
    precision: The precision of the search.
    return_scores: Whether to return a dictionary of (eps, score) pairs.
    verbose: Whether to print the silhouette score for each eps value.

Returns:
    eps: The optimal eps value.
    scores_dict (optional): A dictionary of (eps, score) pairs if return_scores is True.

<a id="clustering._dbscan_jit_utils"></a>

# clustering.\_dbscan\_jit\_utils

<a id="clustering._dbscan_jit_utils._identify_core_points"></a>

#### \_identify\_core\_points

```python
@njit(parallel=True, fastmath=True)
def _identify_core_points(dist_matrix, eps, min_samples)
```

Identify core points based on the distance matrix, eps, and min_samples.

Args:
    dist_matrix: Pairwise distance matrix.
    eps: Maximum distance for neighbors.
    min_samples: Minimum number of neighbors to be a core point.

Returns:
    core_points: Boolean array indicating core points.

<a id="clustering._dbscan_jit_utils._assign_clusters"></a>

#### \_assign\_clusters

```python
@njit(parallel=False, fastmath=True)
def _assign_clusters(dist_matrix, core_points, eps)
```

Assign cluster labels using depth-first search (DFS) starting from core points.

Args:
    dist_matrix: Pairwise distance matrix.
    core_points: Boolean array indicating core points.
    eps: Maximum distance for neighbors.

Returns:
    labels: Cluster labels for each data point.

<a id="linear_models"></a>

# linear\_models

<a id="linear_models.__all__"></a>

#### \_\_all\_\_

<a id="linear_models.classifiers"></a>

# linear\_models.classifiers

<a id="linear_models.classifiers.make_sample_data"></a>

#### make\_sample\_data

```python
def make_sample_data(n_samples,
                     n_features,
                     cov_class_1,
                     cov_class_2,
                     shift=None,
                     seed=0)
```

Generates sample data for testing LDA and QDA models.

Args:
    n_samples: (int) - Number of samples per class.
    n_features: (int) - Number of features.
    cov_class_1: (np.ndarray) - Covariance matrix for class 1.
    cov_class_2: (np.ndarray) - Covariance matrix for class 2.
    shift: (list), optional - Shift applied to class 2 data (default is [1, 1]).
    seed: (int), optional - Random seed for reproducibility (default is 0).

Returns:
    X: (np.ndarray) - Generated feature data.
    y: (np.ndarray) - Generated target labels.

<a id="linear_models.classifiers._validate_data"></a>

#### \_validate\_data

```python
def _validate_data(X, y)
```

Validates input data.

Args:
    X : array-like of shape (n_samples, n_features): Training data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets): Target values.

Must be:
    - array-like
    - same number of samples
    - Not empty

<a id="linear_models.classifiers.LinearDiscriminantAnalysis"></a>

## LinearDiscriminantAnalysis Objects

```python
class LinearDiscriminantAnalysis()
```

Implements Linear Discriminant Analysis.

A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes' rule.

Args:
    solver: (str) - {'svd', 'lsqr', 'eigen'}, default='svd'. Solver to use for the LDA.
    priors: (np.ndarray), optional - Prior probabilities of the classes (default is None).

<a id="linear_models.classifiers.LinearDiscriminantAnalysis.__init__"></a>

#### \_\_init\_\_

```python
def __init__(solver="svd", priors=None)
```

Initializes the Linear Discriminant Analysis model.

Args:
    solver: (str) - {'svd', 'lsqr', 'eigen'}, default='svd'. Solver to use for the LDA.
    priors: (np.ndarray), optional - Prior probabilities of the classes (default is None).

<a id="linear_models.classifiers.LinearDiscriminantAnalysis.fit"></a>

#### fit

```python
def fit(X, y)
```

Fits the LDA model to the training data.

Args:
    X: (np.ndarray) - Training feature data.
    y: (np.ndarray) - Training target data.

<a id="linear_models.classifiers.LinearDiscriminantAnalysis._fit_svd"></a>

#### \_fit\_svd

```python
def _fit_svd(X, y)
```

Fits the LDA model using Singular Value Decomposition (SVD).

Args:
    X: (np.ndarray) - Training feature data.
    y: (np.ndarray) - Training target data.

<a id="linear_models.classifiers.LinearDiscriminantAnalysis._fit_lsqr"></a>

#### \_fit\_lsqr

```python
def _fit_lsqr(X, y)
```

Fits the LDA model using Least Squares (LSQR).

Args:
    X: (np.ndarray) - Training feature data.
    y: (np.ndarray) - Training target data.

<a id="linear_models.classifiers.LinearDiscriminantAnalysis._fit_eigen"></a>

#### \_fit\_eigen

```python
def _fit_eigen(X, y)
```

Fits the LDA model using eigenvalue decomposition.

Args:
    X: (np.ndarray) - Training feature data.
    y: (np.ndarray) - Training target data.

<a id="linear_models.classifiers.LinearDiscriminantAnalysis.predict"></a>

#### predict

```python
def predict(X)
```

Predicts class labels for the input data.

Args:
    X: (np.ndarray) - Test feature data.

Returns:
    predictions: (np.ndarray) - Predicted class labels.

<a id="linear_models.classifiers.LinearDiscriminantAnalysis.decision_function"></a>

#### decision\_function

```python
def decision_function(X)
```

Computes the log-likelihood of each class for the input data. The decision function is the log-likelihood of each class.

Args:
    X: (np.ndarray) - Test feature data.

Returns:
    scores: (np.ndarray) - Log-likelihood of each class for the input samples.

<a id="linear_models.classifiers.QuadraticDiscriminantAnalysis"></a>

## QuadraticDiscriminantAnalysis Objects

```python
class QuadraticDiscriminantAnalysis()
```

Implements Quadratic Discriminant Analysis.

The quadratic term allows for more flexibility in modeling the class conditional
A classifier with a quadratic decision boundary, generated by fitting class conditional densities to the data and using Bayes' rule.

Args:
    priors: (np.ndarray), optional - Prior probabilities of the classes (default is None).
    reg_param: (float), optional - Regularization parameter (default is 0.0).

<a id="linear_models.classifiers.QuadraticDiscriminantAnalysis.__init__"></a>

#### \_\_init\_\_

```python
def __init__(priors=None, reg_param=0.0)
```

Initialize the Quadratic Discriminant Analysis model with the specified prior probabilities and regularization parameter.

Args:
    priors: (np.ndarray), optional - Prior probabilities of the classes (default is None).
    reg_param: (float), optional - Regularization parameter (default is 0.0).

<a id="linear_models.classifiers.QuadraticDiscriminantAnalysis.fit"></a>

#### fit

```python
def fit(X, y)
```

Fit the model according to the given training data. Uses the means and covariance matrices of each class.

Args:
    X: (np.ndarray) - Training feature data.
    y: (np.ndarray) - Training target data.

<a id="linear_models.classifiers.QuadraticDiscriminantAnalysis.predict"></a>

#### predict

```python
def predict(X)
```

Perform classification on an array of test vectors X.

Args:
    X: (np.ndarray) - Test feature data.

Returns:
    predictions: (np.ndarray) - Predicted class labels.

<a id="linear_models.classifiers.QuadraticDiscriminantAnalysis.decision_function"></a>

#### decision\_function

```python
def decision_function(X)
```

Apply decision function to an array of samples.

The decision function is the log-likelihood of each class.

Args:
    X: (np.ndarray) - Test feature data.

Returns:
    scores: (np.ndarray) - Log-likelihood of each class for the input samples.

<a id="linear_models.classifiers.Perceptron"></a>

## Perceptron Objects

```python
class Perceptron()
```

Implements the Perceptron algorithm for binary and multiclass classification.

Args:
    max_iter: (int) - Maximum number of iterations (default is 1000).
    learning_rate: (float) - Learning rate for weight updates (default is 0.01).

<a id="linear_models.classifiers.Perceptron.__init__"></a>

#### \_\_init\_\_

```python
def __init__(max_iter=1000, learning_rate=0.01)
```

Initializes the classifier with the specified maximum number of iterations and learning rate.

Args:
    max_iter (int, optional): The maximum number of iterations for the training process. Defaults to 1000.
    learning_rate (float, optional): The learning rate for the optimization algorithm. Defaults to 0.01.

<a id="linear_models.classifiers.Perceptron.fit"></a>

#### fit

```python
def fit(X, y)
```

Fits the Perceptron model to the training data.

Args:
    X: (np.ndarray) - Training feature data of shape (n_samples, n_features).
    y: (np.ndarray) - Training target data of shape (n_samples,).

<a id="linear_models.classifiers.Perceptron.predict"></a>

#### predict

```python
def predict(X)
```

Predicts class labels for the input data.

Args:
    X: (np.ndarray) - Test feature data.

Returns:
    predictions: (np.ndarray) - Predicted class labels.

<a id="linear_models.classifiers.LogisticRegression"></a>

## LogisticRegression Objects

```python
class LogisticRegression()
```

Implements Logistic Regression using gradient descent. Supports binary and multiclass classification.

Args:
    learning_rate: (float) - Learning rate for gradient updates (default is 0.01).
    max_iter: (int) - Maximum number of iterations (default is 1000).

<a id="linear_models.classifiers.LogisticRegression.__init__"></a>

#### \_\_init\_\_

```python
def __init__(learning_rate=0.01, max_iter=1000)
```

Initializes the classifier with specified hyperparameters.

Args:
    learning_rate (float, optional): The step size for updating weights during training. Defaults to 0.01.
    max_iter (int, optional): The maximum number of iterations for the training process. Defaults to 1000.

<a id="linear_models.classifiers.LogisticRegression.fit"></a>

#### fit

```python
def fit(X, y)
```

Fits the Logistic Regression model to the training data.

Args:
    X: (np.ndarray) - Training feature data of shape (n_samples, n_features).
    y: (np.ndarray) - Training target data of shape (n_samples,).

<a id="linear_models.classifiers.LogisticRegression.predict"></a>

#### predict

```python
def predict(X)
```

Predicts class labels for the input data.

Args:
    X: (np.ndarray) - Test feature data.

Returns:
    predictions: (np.ndarray) - Predicted class labels.

<a id="linear_models.classifiers.LogisticRegression._sigmoid"></a>

#### \_sigmoid

```python
def _sigmoid(z)
```

Applies the sigmoid function.

<a id="linear_models.regressors"></a>

# linear\_models.regressors

<a id="linear_models.regressors._validate_data"></a>

#### \_validate\_data

```python
def _validate_data(X, y)
```

Validates input data.

Args:
    X : array-like of shape (n_samples, n_features): Training data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets): Target values.

Must be:
    - array-like
    - same number of samples
    - Not empty

<a id="linear_models.regressors.OrdinaryLeastSquares"></a>

## OrdinaryLeastSquares Objects

```python
class OrdinaryLeastSquares()
```

Ordinary Least Squares (OLS) linear regression model.

Attributes:
    coef_ : ndarray of shape (n_features,) or (n_features + 1,) - Estimated coefficients for the linear regression problem. If `fit_intercept` is True, the first element is the intercept.
    intercept_ : float - Independent term in the linear model. Set to 0.0 if `fit_intercept` is False.

Methods:
    fit(X, y): Fit the linear model to the data.
    predict(X): Predict using the linear model.
    get_formula(): Returns the formula of the model as a string.

<a id="linear_models.regressors.OrdinaryLeastSquares.__init__"></a>

#### \_\_init\_\_

```python
def __init__(fit_intercept=True) -> None
```

Initializes the OrdinaryLeastSquares object.

Args:
    fit_intercept: (bool) - Whether to calculate the intercept for this model (default is True).

<a id="linear_models.regressors.OrdinaryLeastSquares.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

Returns the string representation of the model.

<a id="linear_models.regressors.OrdinaryLeastSquares.fit"></a>

#### fit

```python
def fit(X, y)
```

Fits the linear regression model to the training data.

Args:
    X: (np.ndarray) - Training feature data of shape (n_samples, n_features).
    y: (np.ndarray) - Training target data of shape (n_samples,).

Returns:
    self: (OrdinaryLeastSquares) - The fitted linear regression model.

<a id="linear_models.regressors.OrdinaryLeastSquares.predict"></a>

#### predict

```python
def predict(X)
```

Predicts the target values using the linear model.

Args:
    X: (np.ndarray) - Feature data of shape (n_samples, n_features).

Returns:
    y_pred: (np.ndarray) - Predicted target values of shape (n_samples,).

<a id="linear_models.regressors.OrdinaryLeastSquares.get_formula"></a>

#### get\_formula

```python
def get_formula()
```

Returns the formula of the model as a string.

Returns:
    formula: (str) - The formula of the model.

<a id="linear_models.regressors.Ridge"></a>

## Ridge Objects

```python
class Ridge()
```

Fits the Ridge Regression model to the training data.

Ridge regression implements L2 regularization, which helps to prevent overfitting by adding a penalty term to the loss function.

Args:
    alpha: (float) - Regularization strength; must be a positive float (default is 1.0).
    fit_intercept: (bool), optional - Whether to calculate the intercept for this model (default is True).
    max_iter: (int), optional - Maximum number of iterations for the coordinate descent solver (default is 10000).
    tol: (float), optional - Tolerance for the optimization. The optimization stops when the change in the coefficients is less than this tolerance (default is 1e-4).

Attributes:
    coef_: (np.ndarray) - Estimated coefficients for the linear regression problem. If `fit_intercept` is True, the first element is the intercept.
    intercept_: (float) - Independent term in the linear model. Set to 0.0 if `fit_intercept` is False.

Methods:
    fit(X, y): Fits the Ridge Regression model to the training data.
    predict(X): Predicts using the Ridge Regression model.
    get_formula(): Returns the formula of the model as a string.

<a id="linear_models.regressors.Ridge.__init__"></a>

#### \_\_init\_\_

```python
def __init__(alpha=1.0,
             fit_intercept=True,
             max_iter=10000,
             tol=1e-4,
             compile_numba=False)
```

Initializes the Ridge Regression model.

Ridge regression implements L2 regularization, which helps to prevent overfitting by adding a penalty term to the loss function.

Args:
    alpha: (float) - Regularization strength; must be a positive float (default is 1.0).
    fit_intercept: (bool), optional - Whether to calculate the intercept for this model (default is True).
    max_iter: (int), optional - Maximum number of iterations for the coordinate descent solver (default is 10000).
    tol: (float), optional - Tolerance for the optimization. The optimization stops when the change in the coefficients is less than this tolerance (default is 1e-4).
    compile_numba: (bool), optional - Whether to precompile the numba functions (default is False). If True, the numba fitting functions will be compiled before use.

<a id="linear_models.regressors.Ridge.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

Returns the string representation of the model.

<a id="linear_models.regressors.Ridge.fit"></a>

#### fit

```python
def fit(X, y, numba=False)
```

Fits the Ridge Regression model to the training data.

Args:
    X: (np.ndarray) - Training feature data of shape (n_samples, n_features).
    y: (np.ndarray) - Training target data of shape (n_samples,).
    numba: (bool), optional - Whether to use numba for faster computation (default is False).

Returns:
    self: (Ridge) - The fitted Ridge Regression model.

<a id="linear_models.regressors.Ridge.predict"></a>

#### predict

```python
def predict(X)
```

Predicts the target values using the Ridge Regression model.

Args:
    X: (np.ndarray) - Feature data of shape (n_samples, n_features).

Returns:
    y_pred: (np.ndarray) - Predicted target values of shape (n_samples,).

<a id="linear_models.regressors.Ridge.get_formula"></a>

#### get\_formula

```python
def get_formula()
```

Computes the formula of the model.

Returns:
    formula: (str) - The formula of the model as a string.

<a id="linear_models.regressors.Lasso"></a>

## Lasso Objects

```python
class Lasso()
```

Fits the Lasso Regression model to the training data.

Lasso regression implements L1 regularization, which helps to prevent overfitting by adding a penalty term to the loss function.

Args:
    X_train: (np.ndarray) - Training feature data of shape (n_samples, n_features).
    y_train: (np.ndarray) - Training target data of shape (n_samples,).
    X_test: (np.ndarray), optional - Testing feature data (default is None).
    y_test: (np.ndarray), optional - Testing target data (default is None).
    custom_metrics: (dict: str -> callable), optional - Custom metrics for evaluation (default is None).
    verbose: (bool), optional - If True, prints progress (default is False).

Attributes:
    coef_: (np.ndarray) - Estimated coefficients for the linear regression problem. If `fit_intercept` is True, the first element is the intercept.
    intercept_: (float) - Independent term in the linear model. Set to 0.0 if `fit_intercept` is False.

Returns:
    results: (list) - A list of dictionaries containing model performance metrics.
    predictions: (dict) - A dictionary of predictions for each model.

<a id="linear_models.regressors.Lasso.__init__"></a>

#### \_\_init\_\_

```python
def __init__(alpha=1.0,
             fit_intercept=True,
             max_iter=10000,
             tol=1e-4,
             compile_numba=False)
```

Initializes the Lasso Regression model.

Lasso regression implements L1 regularization, which helps to prevent overfitting by adding a penalty term to the loss function.

Args:
    alpha: (float) - Regularization strength; must be a positive float (default is 1.0).
    fit_intercept: (bool), optional - Whether to calculate the intercept for this model (default is True).
    max_iter: (int), optional - Maximum number of iterations for the coordinate descent solver (default is 10000).
    tol: (float), optional - Tolerance for the optimization. The optimization stops when the change in the coefficients is less than this tolerance (default is 1e-4).
    compile_numba: (bool), optional - Whether to precompile the numba functions (default is False). If True, the numba fitting functions will be compiled before use.

<a id="linear_models.regressors.Lasso.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

Returns the string representation of the model.

<a id="linear_models.regressors.Lasso.fit"></a>

#### fit

```python
def fit(X, y, numba=False)
```

Fits the Lasso Regression model to the training data using coordinate descent.

Args:
    X: (np.ndarray) - Training feature data of shape (n_samples, n_features).
    y: (np.ndarray) - Training target data of shape (n_samples,).
    numba: (bool), optional - Whether to use numba for faster computation (default is False).

Returns:
    self: (Lasso) - The fitted Lasso Regression model.

<a id="linear_models.regressors.Lasso.predict"></a>

#### predict

```python
def predict(X)
```

Predicts the target values using the Lasso Regression model.

Args:
    X: (np.ndarray) - Feature data of shape (n_samples, n_features).

Returns:
    y_pred: (np.ndarray) - Predicted target values of shape (n_samples,).

<a id="linear_models.regressors.Lasso.get_formula"></a>

#### get\_formula

```python
def get_formula()
```

Computes the formula of the model.

Returns:
- formula : str: The formula of the model.

<a id="linear_models.regressors.Bayesian"></a>

## Bayesian Objects

```python
class Bayesian()
```

Fits the Bayesian Regression model to the training data using Coordinate Descent.

Args:
    X_train: (np.ndarray) - Training feature data.
    y_train: (np.ndarray) - Training target data.
    X_test: (np.ndarray), optional - Testing feature data (default is None).
    y_test: (np.ndarray), optional - Testing target data (default is None).
    max_iter: (int), optional - The maximum number of iterations to perform (default is 300).
    tol: (float), optional - The convergence threshold. The algorithm stops when the coefficients change less than this threshold (default is 0.001).
    alpha_1: (float), optional - The shape parameter for the prior on the weights (default is 1e-06).
    alpha_2: (float), optional - The scale parameter for the prior on the weights (default is 1e-06).
    lambda_1: (float), optional - The shape parameter for the prior on the noise (default is 1e-06).
    lambda_2: (float), optional - The scale parameter for the prior on the noise (default is 1e-06).
    fit_intercept: (bool), optional - Whether to calculate the intercept for this model (default is True).

Returns:
    intercept_: (float) - The intercept of the model.
    coef_: (np.ndarray) - Estimated coefficients for the linear regression problem. If `fit_intercept` is True, the first element is the intercept.
    n_iter_: (int) - The number of iterations performed.
    alpha_: (float) - The precision of the weights.
    lambda_: (float) - The precision of the noise.
    sigma_: (np.ndarray) - The posterior covariance of the weights.

<a id="linear_models.regressors.Bayesian.__init__"></a>

#### \_\_init\_\_

```python
def __init__(max_iter=300,
             tol=0.001,
             alpha_1=1e-06,
             alpha_2=1e-06,
             lambda_1=1e-06,
             lambda_2=1e-06,
             fit_intercept=None)
```

Implements Bayesian Regression using Coordinate Descent.

Bayesian regression applies both L1 and L2 regularization to prevent overfitting by adding penalty terms to the loss function.

Args:
    max_iter: (int) - The maximum number of iterations to perform (default is 300).
    tol: (float) - The convergence threshold. The algorithm stops when the coefficients change less than this threshold (default is 0.001).
    alpha_1: (float) - The shape parameter for the prior on the weights (default is 1e-06).
    alpha_2: (float) - The scale parameter for the prior on the weights (default is 1e-06).
    lambda_1: (float) - The shape parameter for the prior on the noise (default is 1e-06).
    lambda_2: (float) - The scale parameter for the prior on the noise (default is 1e-06).
    fit_intercept: (bool), optional - Whether to calculate the intercept for this model (default is True).

Returns:
    intercept_: (float) - The intercept of the model.
    coef_: (np.ndarray) - Estimated coefficients for the linear regression problem. If `fit_intercept` is True, the first element is the intercept.
    n_iter_: (int) - The number of iterations performed.
    alpha_: (float) - The precision of the weights.
    lambda_: (float) - The precision of the noise.
    sigma_: (np.ndarray) - The posterior covariance of the weights.

<a id="linear_models.regressors.Bayesian.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

Returns the string representation of the model.

<a id="linear_models.regressors.Bayesian.fit"></a>

#### fit

```python
def fit(X, y)
```

Fits the Bayesian Regression model to the training data.

Args:
    X: (np.ndarray) - Training feature data of shape (n_samples, n_features).
    y: (np.ndarray) - Training target data of shape (n_samples,).

Returns:
    self: (Bayesian) - The fitted Bayesian Regression model.

<a id="linear_models.regressors.Bayesian.tune"></a>

#### tune

```python
def tune(X, y, beta1=0.9, beta2=0.999, iter=1000)
```

Tunes the hyperparameters alpha_1, alpha_2, lambda_1, and lambda_2 using ADAM optimizer.

Args:
    X: (np.ndarray) - Training feature data of shape (n_samples, n_features).
    y: (np.ndarray) - Training target data of shape (n_samples,).
    beta1: (float), optional - The exponential decay rate for the first moment estimates (default is 0.9).
    beta2: (float), optional - The exponential decay rate for the second moment estimates (default is 0.999).
    iter: (int), optional - The maximum number of iterations to perform (default is 1000).

Returns:
    best_alpha_1: (float) - The best value of alpha_1.
    best_alpha_2: (float) - The best value of alpha_2.
    best_lambda_1: (float) - The best value of lambda_1.
    best_lambda_2: (float) - The best value of lambda_2.

<a id="linear_models.regressors.Bayesian.predict"></a>

#### predict

```python
def predict(X)
```

Predicts the target values using the Bayesian Regression model.

Args:
    X: (np.ndarray) - Feature data of shape (n_samples, n_features).

Returns:
    y_pred: (np.ndarray) - Predicted target values of shape (n_samples,).

<a id="linear_models.regressors.Bayesian.get_formula"></a>

#### get\_formula

```python
def get_formula()
```

Computes the formula of the model.

Returns:
    formula: (str) - The formula of the model as a string.

<a id="linear_models.regressors.RANSAC"></a>

## RANSAC Objects

```python
class RANSAC()
```

Fits the RANSAC (RANdom SAmple Consensus) algorithm for robust linear regression.

Args:
    X_train: (np.ndarray) - Training feature data.
    y_train: (np.ndarray) - Training target data.
    X_test: (np.ndarray), optional - Testing feature data (default is None).
    y_test: (np.ndarray), optional - Testing target data (default is None).
    n: (int), optional - Number of data points to estimate parameters (default is 10).
    k: (int), optional - Maximum iterations allowed (default is 100).
    t: (float), optional - Threshold value to determine if points are fit well, in terms of residuals (default is 0.05).
    d: (int), optional - Number of close data points required to assert model fits well (default is 10).
    model: (object), optional - The model to use for fitting. If None, uses Ordinary Least Squares (default is None).
    auto_scale_t: (bool), optional - Whether to automatically scale the threshold until a model is fit (default is False).
    scale_t_factor: (float), optional - Factor by which to scale the threshold until a model is fit (default is 2).
    auto_scale_n: (bool), optional - Whether to automatically scale the number of data points until a model is fit (default is False).
    scale_n_factor: (float), optional - Factor by which to scale the number of data points until a model is fit (default is 2).

Returns:
    best_fit: (object) - The best model fit.
    best_error: (float) - The best error achieved by the model.
    best_n: (int) - The best number of data points used to fit the model.
    best_t: (float) - The best threshold value used to determine if points are fit well, in terms of residuals.
    best_model: (object) - The best model fit.

<a id="linear_models.regressors.RANSAC.__init__"></a>

#### \_\_init\_\_

```python
def __init__(n=10,
             k=100,
             t=0.05,
             d=10,
             model=None,
             auto_scale_t=False,
             scale_t_factor=2,
             auto_scale_n=False,
             scale_n_factor=2)
```

Fits the RANSAC (RANdom SAmple Consensus) algorithm for robust linear regression.

Args:
    X_train: (np.ndarray) - Training feature data.
    y_train: (np.ndarray) - Training target data.
    X_test: (np.ndarray), optional - Testing feature data (default is None).
    y_test: (np.ndarray), optional - Testing target data (default is None).
    n: (int), optional - Number of data points to estimate parameters (default is 10).
    k: (int), optional - Maximum iterations allowed (default is 100).
    t: (float), optional - Threshold value to determine if points are fit well, in terms of residuals (default is 0.05).
    d: (int), optional - Number of close data points required to assert model fits well (default is 10).
    model: (object), optional - The model to use for fitting. If None, uses Ordinary Least Squares (default is None).
    auto_scale_t: (bool), optional - Whether to automatically scale the threshold until a model is fit (default is False).
    scale_t_factor: (float), optional - Factor by which to scale the threshold until a model is fit (default is 2).
    auto_scale_n: (bool), optional - Whether to automatically scale the number of data points until a model is fit (default is False).
    scale_n_factor: (float), optional - Factor by which to scale the number of data points until a model is fit (default is 2).

Returns:
    best_fit: (object) - The best model fit.
    best_error: (float) - The best error achieved by the model.
    best_n: (int) - The best number of data points used to fit the model.
    best_t: (float) - The best threshold value used to determine if points are fit well, in terms of residuals.
    best_model: (object) - The best model fit.

<a id="linear_models.regressors.RANSAC.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

Returns the string representation of the model.

<a id="linear_models.regressors.RANSAC._square_loss"></a>

#### \_square\_loss

```python
def _square_loss(y_true, y_pred)
```

Compute the square loss.

<a id="linear_models.regressors.RANSAC._mean_square_loss"></a>

#### \_mean\_square\_loss

```python
def _mean_square_loss(y_true, y_pred)
```

Compute the mean square loss.

<a id="linear_models.regressors.RANSAC.fit"></a>

#### fit

```python
def fit(X, y)
```

Fits the RANSAC model to the training data.

Args:
    X: (np.ndarray) - Training feature data of shape (n_samples, n_features).
    y: (np.ndarray) - Training target data of shape (n_samples,).

Returns:
    None

<a id="linear_models.regressors.RANSAC.predict"></a>

#### predict

```python
def predict(X)
```

Predicts the target values using the best fit model.

Args:
    X: (np.ndarray) - Feature data of shape (n_samples, n_features).

Returns:
    y_pred: (np.ndarray) - Predicted target values of shape (n_samples,).

<a id="linear_models.regressors.RANSAC.get_formula"></a>

#### get\_formula

```python
def get_formula()
```

Computes the formula of the model if fit, else returns "No model fit available".

<a id="linear_models.regressors.PassiveAggressiveRegressor"></a>

## PassiveAggressiveRegressor Objects

```python
class PassiveAggressiveRegressor()
```

Fits the Passive Aggressive Regression model to the training data.

Args:
    X: (np.ndarray) - Training feature data of shape (n_samples, n_features).
    y: (np.ndarray) - Training target data of shape (n_samples,).
    save_steps: (bool), optional - Whether to save the weights and intercept at each iteration (default is False).
    verbose: (bool), optional - If True, prints progress during training (default is False).

Attributes:
    coef_: (np.ndarray) - Estimated coefficients for the regression problem.
    intercept_: (float) - Independent term in the linear model.
    steps_: (list of tuples), optional - The weights and intercept at each iteration if `save_steps` is True.

<a id="linear_models.regressors.PassiveAggressiveRegressor.__init__"></a>

#### \_\_init\_\_

```python
def __init__(C=1.0, max_iter=1000, tol=1e-3)
```

Fits the Passive Aggressive Regression model to the training data.

Args:
    C: (float) - Regularization parameter/step size (default is 1.0).
    max_iter: (int) - The maximum number of passes over the training data (default is 1000).
    tol: (float) - The stopping criterion (default is 1e-3).

Attributes:
    coef_: (np.ndarray) - Estimated coefficients for the regression problem.
    intercept_: (float) - Independent term in the linear model.

<a id="linear_models.regressors.PassiveAggressiveRegressor.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

Returns the string representation of the model.

<a id="linear_models.regressors.PassiveAggressiveRegressor.fit"></a>

#### fit

```python
def fit(X, y, save_steps=False, verbose=False)
```

Fits the Passive Aggressive Regression model to the training data.

Args:
    X: (np.ndarray) - Training feature data of shape (n_samples, n_features).
    y: (np.ndarray) - Training target data of shape (n_samples,).
    save_steps: (bool), optional - Whether to save the weights and intercept at each iteration (default is False).
    verbose: (bool), optional - If True, prints progress during training (default is False).

Returns:
    None

<a id="linear_models.regressors.PassiveAggressiveRegressor.predict"></a>

#### predict

```python
def predict(X)
```

Predict using the linear model. Dot product of X and the coefficients.

<a id="linear_models.regressors.PassiveAggressiveRegressor.predict_all_steps"></a>

#### predict\_all\_steps

```python
def predict_all_steps(X)
```

Predict using the linear model at each iteration. (save_steps=True).

<a id="linear_models.regressors.PassiveAggressiveRegressor.get_formula"></a>

#### get\_formula

```python
def get_formula()
```

Computes the formula of the model.

Returns:
    formula : str: The formula of the model.

<a id="linear_models._lasso_jit_utils"></a>

# linear\_models.\_lasso\_jit\_utils

<a id="linear_models._lasso_jit_utils._fit_numba_no_intercept"></a>

#### \_fit\_numba\_no\_intercept

```python
@njit(parallel=True, fastmath=True)
def _fit_numba_no_intercept(X, y, alpha, max_iter, tol)
```

Fits the model to the data using coordinate descent with numba (no intercept) for Lasso.

Args:
    X: (np.ndarray) - Training feature data of shape (n_samples, n_features).
    y: (np.ndarray) - Target values of shape (n_samples,).
    alpha: (float) - Regularization strength.
    max_iter: (int) - Maximum number of iterations.
    tol: (float) - Tolerance for convergence.

Returns:
    coef_: (np.ndarray) - Estimated coefficients of shape (n_features,).

<a id="linear_models._lasso_jit_utils._fit_numba_intercept"></a>

#### \_fit\_numba\_intercept

```python
@njit(parallel=True, fastmath=True)
def _fit_numba_intercept(X, y, alpha, max_iter, tol)
```

Fits the model to the data using coordinate descent with numba (with intercept) for Lasso.

Args:
    X: (np.ndarray) - Training feature data of shape (n_samples, n_features).
    y: (np.ndarray) - Target values of shape (n_samples,).
    alpha: (float) - Regularization strength.
    max_iter: (int) - Maximum number of iterations.
    tol: (float) - Tolerance for convergence.

Returns:
    coef_: (np.ndarray) - Estimated coefficients of shape (n_features,).
    intercept_: (float) - Estimated intercept.

<a id="linear_models._ridge_jit_utils"></a>

# linear\_models.\_ridge\_jit\_utils

<a id="linear_models._ridge_jit_utils._fit_numba_no_intercept"></a>

#### \_fit\_numba\_no\_intercept

```python
@njit(parallel=True, fastmath=True)
def _fit_numba_no_intercept(X, y, alpha, max_iter, tol)
```

Fits the model to the data using coordinate descent with numba (no intercept).

Args:
    X: (np.ndarray) - Training feature data of shape (n_samples, n_features).
    y: (np.ndarray) - Target values of shape (n_samples,).
    alpha: (float) - Regularization strength.
    max_iter: (int) - Maximum number of iterations.
    tol: (float) - Tolerance for convergence.

Returns:
    coef_: (np.ndarray) - Estimated coefficients of shape (n_features,).

<a id="linear_models._ridge_jit_utils._fit_numba_intercept"></a>

#### \_fit\_numba\_intercept

```python
@njit(parallel=True, fastmath=True)
def _fit_numba_intercept(X, y, alpha, max_iter, tol)
```

Fits the model to the data using coordinate descent with numba (with intercept).

Args:
    X: (np.ndarray) - Training feature data of shape (n_samples, n_features).
    y: (np.ndarray) - Target values of shape (n_samples,).
    alpha: (float) - Regularization strength.
    max_iter: (int) - Maximum number of iterations.
    tol: (float) - Tolerance for convergence.

Returns:
    coef_: (np.ndarray) - Estimated coefficients of shape (n_features,).
    intercept_: (float) - Estimated intercept.

<a id="nearest_neighbors"></a>

# nearest\_neighbors

<a id="nearest_neighbors.__all__"></a>

#### \_\_all\_\_

<a id="nearest_neighbors.base"></a>

# nearest\_neighbors.base

<a id="nearest_neighbors.base.KNeighborsBase"></a>

## KNeighborsBase Objects

```python
class KNeighborsBase(ABC)
```

Abstract base class for implementing k-nearest neighbors (KNN) algorithms.

Provides common functionality for fitting data, computing distances, and managing configurations.

Attributes:
    n_neighbors (int): Number of neighbors to use for the KNN algorithm.
    distance_metric (str): Distance metric for calculating distances ('euclidean', 'manhattan', 'minkowski').
    one_hot_encode (bool): Whether to apply one-hot encoding to categorical columns.
    fp_precision (type): Floating point precision for calculations.
    numba (bool): Whether to use numba for performance optimization.
    X_train (np.ndarray): Training feature data.
    y_train (np.ndarray): Training target data.

Methods:
    __init__(n_neighbors=5, distance_metric="euclidean", one_hot_encode=False,
             fp_precision=np.float64, numba=False):
        Initializes the KNeighborsBase class with specified parameters.
    fit(X, y):
        Fits the model using the training data and target values.
    get_distance_indices(X):
        Computes distances and returns indices of the nearest points in the training data.
    _data_precision(X, y=None):
        Sets the floating point precision for the input data.
    _check_data(X, y):
        Validates input data to ensure it is numeric and consistent.
    _one_hot_encode(X):
        Applies one-hot encoding to categorical columns in the input data.
    _compute_distances(X):
        Computes distances between input data and training data using the specified distance metric.
    _compute_distances_euclidean(X):
        Computes distances using the Euclidean distance formula.
    _compute_distances_manhattan(X):
        Computes distances using the Manhattan distance formula.
    _compute_distances_minkowski(X, p=3):
        Computes distances using the Minkowski distance formula with specified order `p`.
    predict(X):
        Abstract method to be implemented by subclasses for making predictions based on input data.

<a id="nearest_neighbors.base.KNeighborsBase.__init__"></a>

#### \_\_init\_\_

```python
def __init__(n_neighbors=5,
             distance_metric="euclidean",
             one_hot_encode=False,
             fp_precision=np.float64,
             numba=False)
```

Initialize the KNeighborsBase class.

Args:
    n_neighbors: int, default=5. The number of neighbors to use for the KNN algorithm.
    distance_metric: str, default='euclidean'. The distance metric to use for calculating distances.
    one_hot_encode: bool, default=False. Whether to apply one-hot encoding to the categorical columns.
    fp_precision: data type, default=np.float64. The floating point precision to use for the calculations.
    numba: bool, default=True. Whether to use numba for speeding up the calculations.

<a id="nearest_neighbors.base.KNeighborsBase.fit"></a>

#### fit

```python
def fit(X, y)
```

Fit the model using the training data.

Args:
    X: array-like, shape (n_samples, n_features) - The training data.
    y: array-like, shape (n_samples,) - The target values.

<a id="nearest_neighbors.base.KNeighborsBase.get_distance_indices"></a>

#### get\_distance\_indices

```python
def get_distance_indices(X)
```

Compute the distances and return the indices of the nearest points im the training data.

Args:
    X: array-like, shape (n_samples, n_features) - The input data.

Returns:
    indices: array, shape (n_samples, n_neighbors) - The indices of the nearest neighbors.

<a id="nearest_neighbors.base.KNeighborsBase._data_precision"></a>

#### \_data\_precision

```python
def _data_precision(X, y=None)
```

Set the floating point precision for the input data.

Args:
    X: array-like, shape (n_samples, n_features) - The training data.
    y: array-like, shape (n_samples,) - The target values.

<a id="nearest_neighbors.base.KNeighborsBase._check_data"></a>

#### \_check\_data

```python
def _check_data(X, y)
```

Check if the input data is valid.

Args:
    X: array-like, shape (n_samples, n_features) - The input data.
    y: array-like, shape (n_samples,) - The target values.

<a id="nearest_neighbors.base.KNeighborsBase._one_hot_encode"></a>

#### \_one\_hot\_encode

```python
def _one_hot_encode(X)
```

Apply one-hot encoding to the categorical columns in the DataFrame.

<a id="nearest_neighbors.base.KNeighborsBase._compute_distances"></a>

#### \_compute\_distances

```python
def _compute_distances(X)
```

Helper method to call the appropriate distance computation method.

<a id="nearest_neighbors.base.KNeighborsBase._compute_distances_euclidean"></a>

#### \_compute\_distances\_euclidean

```python
def _compute_distances_euclidean(X)
```

Compute the distances between the training data and the input data.

This method uses the Euclidean distance formula.
Formula: d(x, y) = sqrt(sum((x_i - y_i)^2))

<a id="nearest_neighbors.base.KNeighborsBase._compute_distances_manhattan"></a>

#### \_compute\_distances\_manhattan

```python
def _compute_distances_manhattan(X)
```

Compute the distances between the training data and the input data.

This method uses the Manhattan distance formula.
Formula: d(x, y) = sum(|x_i - y_i|)

<a id="nearest_neighbors.base.KNeighborsBase._compute_distances_minkowski"></a>

#### \_compute\_distances\_minkowski

```python
def _compute_distances_minkowski(X, p=3)
```

Compute the distances between the training data and the input data.

This method uses the Minkowski distance formula.
Formula: d(x, y) = (sum(|x_i - y_i|^p))^(1/p)
where p is the order of the norm.

<a id="nearest_neighbors.base.KNeighborsBase.predict"></a>

#### predict

```python
@abstractmethod
def predict(X)
```

The @abstractmethod decorator indicates that this method must be implemented by any subclass of KNNBase.

<a id="nearest_neighbors.knn_classifier"></a>

# nearest\_neighbors.knn\_classifier

<a id="nearest_neighbors.knn_classifier.KNeighborsClassifier"></a>

## KNeighborsClassifier Objects

```python
class KNeighborsClassifier(KNeighborsBase)
```

K-Nearest Neighbors classifier.

This class implements the k-nearest neighbors algorithm for classification.

<a id="nearest_neighbors.knn_classifier.KNeighborsClassifier.predict"></a>

#### predict

```python
def predict(X)
```

Predict the class labels for the provided data.

Args:
    X: array-like, shape (n_samples, n_features) - The input data for which to predict the class labels.

Returns:
    predictions: array, shape (n_samples,) - The predicted class labels for the input data.

<a id="nearest_neighbors.knn_regressor"></a>

# nearest\_neighbors.knn\_regressor

<a id="nearest_neighbors.knn_regressor.KNeighborsRegressor"></a>

## KNeighborsRegressor Objects

```python
class KNeighborsRegressor(KNeighborsBase)
```

K-Nearest Neighbors classifier.

This class implements the k-nearest neighbors algorithm for regression.

<a id="nearest_neighbors.knn_regressor.KNeighborsRegressor.predict"></a>

#### predict

```python
def predict(X)
```

Predict the class labels for the provided data.

Args:
    X: array-like, shape (n_samples, n_features) - The input data for which to predict the class labels.

Returns:
    predictions: array, shape (n_samples,) - The predicted class labels for the input data.

<a id="nearest_neighbors._nearest_neighbors_jit_utils"></a>

# nearest\_neighbors.\_nearest\_neighbors\_jit\_utils

<a id="nearest_neighbors._nearest_neighbors_jit_utils._jit_compute_distances_euclidean"></a>

#### \_jit\_compute\_distances\_euclidean

```python
@njit(parallel=True, fastmath=True)
def _jit_compute_distances_euclidean(X, X_train)
```

<a id="nearest_neighbors._nearest_neighbors_jit_utils._jit_compute_distances_manhattan"></a>

#### \_jit\_compute\_distances\_manhattan

```python
@njit(parallel=True, fastmath=True)
def _jit_compute_distances_manhattan(X, X_train)
```

<a id="nearest_neighbors._nearest_neighbors_jit_utils._jit_compute_distances_minkowski"></a>

#### \_jit\_compute\_distances\_minkowski

```python
@njit(parallel=True, fastmath=True)
def _jit_compute_distances_minkowski(X, X_train, p)
```

<a id="nearest_neighbors._nearest_neighbors_jit_utils._numba_predict_regressor"></a>

#### \_numba\_predict\_regressor

```python
@njit(parallel=True, fastmath=True)
def _numba_predict_regressor(distances, y_train, n_neighbors)
```

Numba-optimized helper function for KNN regression predictions.

Args:
    distances (np.ndarray): 2D array of shape (n_samples, n_train_samples), precomputed distances.
    y_train (np.ndarray): 1D array of shape (n_train_samples,), training labels.
    n_neighbors (int): Number of nearest neighbors to consider.

Returns:
    np.ndarray: 1D array of shape (n_samples,), predicted values.

<a id="nearest_neighbors._nearest_neighbors_jit_utils._numba_predict_classifier"></a>

#### \_numba\_predict\_classifier

```python
@njit(parallel=True, fastmath=True)
def _numba_predict_classifier(distances, y_train, n_neighbors)
```

Numba-optimized helper function for KNN classification predictions.

Args:
    distances (np.ndarray): 2D array of shape (n_samples, n_train_samples), precomputed distances.
    y_train (np.ndarray): 1D array of shape (n_train_samples,), training labels.
    n_neighbors (int): Number of nearest neighbors to consider.

Returns:
    predictions (np.ndarray): 1D array of shape (n_samples,), predicted class labels.

<a id="neural_networks"></a>

# neural\_networks

<a id="neural_networks.__all__"></a>

#### \_\_all\_\_

<a id="neural_networks.activations"></a>

# neural\_networks.activations

<a id="neural_networks.activations.Activation"></a>

## Activation Objects

```python
class Activation()
```

This class contains various activation functions and their corresponding derivatives for use in neural networks.

Methods:
    relu: Rectified Linear Unit activation function. Returns the input directly if it's positive, otherwise returns 0.
    leaky_relu: Leaky ReLU activation function. A variant of ReLU that allows a small gradient when the input is negative.
    tanh: Hyperbolic tangent activation function. Maps input to range [-1, 1]. Commonly used for normalized input.
    sigmoid: Sigmoid activation function. Maps input to range [0, 1]. Commonly used for binary classification.
    softmax: Softmax activation function. Maps input into a probability distribution over multiple classes.

<a id="neural_networks.activations.Activation.relu"></a>

#### relu

```python
@staticmethod
def relu(z)
```

ReLU (Rectified Linear Unit) activation function: f(z) = max(0, z).

Returns the input directly if it's positive, otherwise returns 0.

<a id="neural_networks.activations.Activation.relu_derivative"></a>

#### relu\_derivative

```python
@staticmethod
def relu_derivative(z)
```

Derivative of the ReLU function: f'(z) = 1 if z > 0, else 0.

Returns 1 for positive input, and 0 for negative input.

<a id="neural_networks.activations.Activation.leaky_relu"></a>

#### leaky\_relu

```python
@staticmethod
def leaky_relu(z, alpha=0.01)
```

Leaky ReLU activation function: f(z) = z if z > 0, else alpha * z.

Allows a small, non-zero gradient when the input is negative to address the dying ReLU problem.

<a id="neural_networks.activations.Activation.leaky_relu_derivative"></a>

#### leaky\_relu\_derivative

```python
@staticmethod
def leaky_relu_derivative(z, alpha=0.01)
```

Derivative of the Leaky ReLU function: f'(z) = 1 if z > 0, else alpha.

Returns 1 for positive input, and alpha for negative input.

<a id="neural_networks.activations.Activation.tanh"></a>

#### tanh

```python
@staticmethod
def tanh(z)
```

Hyperbolic tangent (tanh) activation function: f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z)).

Maps input to the range [-1, 1], typically used for normalized input.

<a id="neural_networks.activations.Activation.tanh_derivative"></a>

#### tanh\_derivative

```python
@staticmethod
def tanh_derivative(z)
```

Derivative of the tanh function: f'(z) = 1 - tanh(z)^2.

Used for backpropagation through the tanh activation.

<a id="neural_networks.activations.Activation.sigmoid"></a>

#### sigmoid

```python
@staticmethod
def sigmoid(z)
```

Sigmoid activation function: f(z) = 1 / (1 + exp(-z)).

Maps input to the range [0, 1], commonly used for binary classification.

<a id="neural_networks.activations.Activation.sigmoid_derivative"></a>

#### sigmoid\_derivative

```python
@staticmethod
def sigmoid_derivative(z)
```

Derivative of the sigmoid function: f'(z) = sigmoid(z) * (1 - sigmoid(z)).

Used for backpropagation through the sigmoid activation.

<a id="neural_networks.activations.Activation.softmax"></a>

#### softmax

```python
@staticmethod
def softmax(z)
```

Softmax activation function: f(z)_i = exp(z_i) / sum(exp(z_j)) for all j.

Maps input into a probability distribution over multiple classes. Used for multiclass classification.

<a id="neural_networks.animation"></a>

# neural\_networks.animation

<a id="neural_networks.animation.TrainingAnimator"></a>

## TrainingAnimator Objects

```python
class TrainingAnimator()
```

A utility class to create and manage training animations.

This class provides callback functions that can be used during model training.

<a id="neural_networks.animation.TrainingAnimator.__init__"></a>

#### \_\_init\_\_

```python
def __init__(figure_size=(18, 10), dpi=100)
```

Initialize the animator with given figure size and DPI.

Args:
    figure_size: (tuple) - Size of the figure (width, height)
    dpi: (int) - DPI for rendering

<a id="neural_networks.animation.TrainingAnimator.initialize"></a>

#### initialize

```python
def initialize(metrics_to_track, has_validation=False)
```

Initialize the animation with specified metrics.

Args:
    metrics_to_track: (list) - List of metrics to track
    has_validation: (bool) - Whether validation metrics are available

<a id="neural_networks.animation.TrainingAnimator.update_metrics"></a>

#### update\_metrics

```python
def update_metrics(epoch_metrics, validation=False)
```

Update the stored metrics with new values.

Args:
    epoch_metrics (dict): Dictionary containing metric values
    validation (bool): Whether these are validation metrics

<a id="neural_networks.animation.TrainingAnimator.animate_training_metrics"></a>

#### animate\_training\_metrics

```python
def animate_training_metrics(interval=200,
                             blit=True,
                             save_path=None,
                             save_format="mp4",
                             fps=10,
                             dpi=300)
```

Create an animation of the training metrics.

Args:
    interval: (int) - Delay between frames in milliseconds
    blit: (bool) - Whether to use blitting for efficient animation
    save_path: (str - optional): Path to save the animation
    save_format: (str) - Format to save animation ('mp4', 'gif', etc.)
    fps: (int) - Frames per second for the saved video
    dpi: (int) - DPI for the saved animation

Returns:
    animation.FuncAnimation: Animation object

<a id="neural_networks.animation.TrainingAnimator.setup_training_video"></a>

#### setup\_training\_video

```python
def setup_training_video(filepath, fps=10, dpi=None)
```

Set up a video writer to capture training progress in real-time.

Args:
    filepath: (str) - Path to save the video
    fps: (int) - Frames per second
    dpi: (int, optional) - DPI for rendering

<a id="neural_networks.animation.TrainingAnimator.add_training_frame"></a>

#### add\_training\_frame

```python
def add_training_frame()
```

Add a frame to the training video.

<a id="neural_networks.animation.TrainingAnimator.finish_training_video"></a>

#### finish\_training\_video

```python
def finish_training_video(print_message=True)
```

Finish and save the training video.

<a id="neural_networks.cupy_utils"></a>

# neural\_networks.cupy\_utils

<a id="neural_networks.cupy_utils.fused_dropout"></a>

#### fused\_dropout

```python
@fuse()
def fused_dropout(x, dropout_rate, random_vals)
```

Apply fused dropout operation.

<a id="neural_networks.cupy_utils.apply_dropout"></a>

#### apply\_dropout

```python
def apply_dropout(X, dropout_rate)
```

Generate dropout mask and apply fused dropout.

<a id="neural_networks.cupy_utils.fused_relu"></a>

#### fused\_relu

```python
@fuse()
def fused_relu(x)
```

Apply fused ReLU activation.

<a id="neural_networks.cupy_utils.fused_sigmoid"></a>

#### fused\_sigmoid

```python
@fuse()
def fused_sigmoid(x)
```

Apply fused sigmoid activation.

<a id="neural_networks.cupy_utils.fused_leaky_relu"></a>

#### fused\_leaky\_relu

```python
@fuse()
def fused_leaky_relu(x, alpha=0.01)
```

Apply fused leaky ReLU activation.

<a id="neural_networks.cupy_utils.forward_cupy"></a>

#### forward\_cupy

```python
def forward_cupy(X, weights, biases, activations, dropout_rate, training,
                 is_binary)
```

Perform forward pass using CuPy with fused and in-place operations.

<a id="neural_networks.cupy_utils.backward_cupy"></a>

#### backward\_cupy

```python
def backward_cupy(layer_outputs, y, weights, activations, reg_lambda,
                  is_binary, dWs, dbs)
```

Perform backward pass using CuPy with fused derivative computations.

<a id="neural_networks.cupy_utils.logsumexp"></a>

#### logsumexp

```python
def logsumexp(a, axis=None, keepdims=False)
```

Compute log-sum-exp for numerical stability.

<a id="neural_networks.cupy_utils.calculate_cross_entropy_loss"></a>

#### calculate\_cross\_entropy\_loss

```python
def calculate_cross_entropy_loss(logits, targets)
```

Calculate cross-entropy loss for multi-class classification.

<a id="neural_networks.cupy_utils.calculate_bce_with_logits_loss"></a>

#### calculate\_bce\_with\_logits\_loss

```python
def calculate_bce_with_logits_loss(logits, targets)
```

Calculate binary cross-entropy loss with logits.

<a id="neural_networks.cupy_utils.calculate_loss_from_outputs_binary"></a>

#### calculate\_loss\_from\_outputs\_binary

```python
def calculate_loss_from_outputs_binary(outputs, y, weights, reg_lambda)
```

Calculate binary classification loss with L2 regularization.

<a id="neural_networks.cupy_utils.calculate_loss_from_outputs_multi"></a>

#### calculate\_loss\_from\_outputs\_multi

```python
def calculate_loss_from_outputs_multi(outputs, y, weights, reg_lambda)
```

Calculate multi-class classification loss with L2 regularization.

<a id="neural_networks.cupy_utils.evaluate_batch"></a>

#### evaluate\_batch

```python
def evaluate_batch(y_hat, y_true, is_binary)
```

Evaluate batch accuracy for binary or multi-class classification.

<a id="neural_networks.layers"></a>

# neural\_networks.layers

<a id="neural_networks.layers.DenseLayer"></a>

## DenseLayer Objects

```python
class DenseLayer()
```

Initializes a fully connected layer object, where each neuron is connected to all neurons in the previous layer.

Each layer consists of weights, biases, and an activation function.

Args:
    input_size (int): The size of the input to the layer.
    output_size (int): The size of the output from the layer.
    activation (str): The activation function to be used in the layer.

Attributes:
    weights (np.ndarray): Weights of the layer.
    biases (np.ndarray): Biases of the layer.
    activation (str): Activation function name.
    weight_gradients (np.ndarray): Gradients of the weights.
    bias_gradients (np.ndarray): Gradients of the biases.
    input_cache (np.ndarray): Cached input for backpropagation.
    output_cache (np.ndarray): Cached output for backpropagation.

Methods:
    zero_grad(): Resets the gradients of the weights and biases to zero.
    forward(X): Performs the forward pass of the layer.
    backward(dA, reg_lambda): Performs the backward pass of the layer.
    activate(Z): Applies the activation function.
    activation_derivative(Z): Applies the derivative of the activation function.

<a id="neural_networks.layers.DenseLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(input_size, output_size, activation="relu")
```

Initializes the layer with weights, biases, and activation function.

Args:
    input_size: (int) - The number of input features to the layer.
    output_size: (int) - The number of output features from the layer.
    activation: (str), optional - The activation function to use (default is "relu").

Attributes:
    weights: (np.ndarray) - The weight matrix initialized using He initialization for ReLU or Leaky ReLU, or standard initialization otherwise.
    biases: (np.ndarray) - The bias vector initialized to zeros.
    input_size: (int) - The number of input features to the layer.
    output_size: (int) - The number of output features from the layer.
    activation: (str) - The activation function to use.
    weight_gradients: (np.ndarray or None) - Gradients of the weights, initialized to None.
    bias_gradients: (np.ndarray or None) - Gradients of the biases, initialized to None.

<a id="neural_networks.layers.DenseLayer.zero_grad"></a>

#### zero\_grad

```python
def zero_grad()
```

Reset the gradients of the weights and biases to zero.

<a id="neural_networks.layers.DenseLayer.forward"></a>

#### forward

```python
def forward(X)
```

Forward pass of the layer.

<a id="neural_networks.layers.DenseLayer.backward"></a>

#### backward

```python
def backward(dA, reg_lambda)
```

Backward pass of the layer.

<a id="neural_networks.layers.DenseLayer.activate"></a>

#### activate

```python
def activate(Z)
```

Apply activation function.

<a id="neural_networks.layers.DenseLayer.activation_derivative"></a>

#### activation\_derivative

```python
def activation_derivative(Z)
```

Apply activation derivative.

<a id="neural_networks.layers.FlattenLayer"></a>

## FlattenLayer Objects

```python
class FlattenLayer()
```

A layer that flattens multi-dimensional input into a 2D array (batch_size, flattened_size).

Useful for transitioning from convolutional layers to dense layers.

Attributes:
    input_shape: (tuple) - Shape of the input data (excluding batch size).
    output_size: (int) - Size of the flattened output vector.
    input_cache: (np.ndarray) - Cached input for backpropagation.
    input_size: (int) - Size of the input (same as input_shape).
    output_size: (int) - Size of the output (same as output_size).

<a id="neural_networks.layers.FlattenLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__()
```

Initializes the layer with default attributes.

Attributes:
    input_shape: (tuple or None) - Shape of the input data, to be set dynamically during the forward pass.
    output_size: (int or None) - Size of the output data, to be set dynamically during the forward pass.
    input_cache: (any or None) - Cache to store input data for use during backpropagation.
    input_size: (int or None) - Flattened size of the input, calculated as channels * height * width.
    output_size: (int or None) - Flattened size of the output, same as input_size.

<a id="neural_networks.layers.FlattenLayer.forward"></a>

#### forward

```python
def forward(X)
```

Flattens the input tensor.

Args:
    X: (np.ndarray) - Input data of shape (batch_size, channels, height, width)
                   or any multi-dimensional shape after batch dimension.

Returns:
    np.ndarray: Flattened output of shape (batch_size, flattened_size)

<a id="neural_networks.layers.FlattenLayer.backward"></a>

#### backward

```python
def backward(dA, reg_lambda=0)
```

Reshapes the gradient back to the original input shape.

Args:
    dA (np.ndarray): Gradient of the loss with respect to the layer's output,
                    shape (batch_size, flattened_size)
    reg_lambda (float): Regularization parameter (unused in FlattenLayer).

Returns:
    np.ndarray: Gradient with respect to the input, reshaped to original input shape.

<a id="neural_networks.layers.ConvLayer"></a>

## ConvLayer Objects

```python
class ConvLayer()
```

A convolutional layer implementation for neural networks.

This layer performs 2D convolution operations, commonly used in convolutional neural networks (CNNs).
The implementation uses the im2col technique for efficient computation, transforming the convolution operation into matrix multiplication.
An optional activation function is applied element-wise to the output.

Args:
    in_channels (int): Number of input channels (depth of input volume).
    out_channels (int): Number of output channels (number of filters).
    kernel_size (int): Size of the convolutional kernel (square kernel assumed).
    stride (int, optional): Stride of the convolution. Default: 1.
    padding (int, optional): Zero-padding added to both sides of the input. Default: 0.
    activation (str, optional): Activation function to use. Options are "relu", "sigmoid", "tanh", or None. Default: "relu".

Attributes:
    in_channels (int): Number of input channels.
    out_channels (int): Number of output channels.
    kernel_size (int): Size of the square convolutional kernel.
    stride (int): Stride of the convolution.
    padding (int): Zero-padding added to both sides of the input.
    weights (numpy.ndarray): Learnable weights of shape (out_channels, in_channels, kernel_size, kernel_size).
    biases (numpy.ndarray): Learnable biases of shape (out_channels, 1).
    activation (str): Type of activation function.
    weight_gradients (numpy.ndarray): Gradients with respect to weights.
    bias_gradients (numpy.ndarray): Gradients with respect to biases.
    input_cache (numpy.ndarray): Cached input for use in backward pass.
    X_cols (numpy.ndarray): Cached column-transformed input.
    X_padded (numpy.ndarray): Cached padded input.
    h_out (int): Height of output feature maps.
    w_out (int): Width of output feature maps.
    input_size (int): Size of input (same as in_channels).
    output_size (int): Size of output (same as out_channels).

Methods:
    zero_grad(): Reset gradients to zero.
    _im2col(x, h_out, w_out): Convert image regions to columns for efficient convolution.
    forward(X): Perform forward pass of the convolutional layer.
    _col2im(dcol, x_shape): Convert column back to image format for the backward pass.
    backward(d_out, reg_lambda=0): Perform backward pass of the convolutional layer.

<a id="neural_networks.layers.ConvLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(in_channels,
             out_channels,
             kernel_size,
             stride=1,
             padding=0,
             activation="relu")
```

Initializes a convolutional layer object for neural networks.

This layer performs 2D convolution operations, commonly used in convolutional neural networks (CNNs).

Args:
    in_channels: (int) - Number of input channels (depth of input volume).
    out_channels: (int) - Number of output channels (number of filters).
    kernel_size: (int) - Size of the convolutional kernel (square kernel assumed).
    stride: (int), optional - Stride of the convolution (default is 1).
    padding: (int), optional - Zero-padding added to both sides of the input (default is 0).
    activation: (str), optional - Activation function to use (default is "relu").

Attributes:
    in_channels: (int) - Number of input channels.
    out_channels: (int) - Number of output channels.
    kernel_size: (int) - Size of the square convolutional kernel.
    stride: (int) - Stride of the convolution.
    padding: (int) - Zero-padding added to both sides of the input.
    weights: (np.ndarray) - Learnable weights of shape (out_channels, in_channels, kernel_size, kernel_size).
    biases: (np.ndarray) - Learnable biases of shape (out_channels, 1).
    activation: (str) - Type of activation function.
    weight_gradients: (np.ndarray or None) - Gradients with respect to weights, initialized to None.
    bias_gradients: (np.ndarray or None) - Gradients with respect to biases, initialized to None.
    input_cache: (np.ndarray or None) - Cached input for use in backward pass.
    input_size: (int) - Size of input (same as in_channels).
    output_size: (int) - Size of output (same as out_channels).

<a id="neural_networks.layers.ConvLayer.zero_grad"></a>

#### zero\_grad

```python
def zero_grad()
```

Reset the gradients of the weights and biases to zero.

<a id="neural_networks.layers.ConvLayer._im2col"></a>

#### \_im2col

```python
def _im2col(x, h_out, w_out)
```

Convert image regions to columns for efficient convolution.

This transforms the 4D input tensor into a 2D matrix where each column
contains a kernel-sized region of the input.

<a id="neural_networks.layers.ConvLayer._col2im"></a>

#### \_col2im

```python
def _col2im(dcol, x_shape)
```

Convert column back to image format for the backward pass.

<a id="neural_networks.layers.ConvLayer.forward"></a>

#### forward

```python
def forward(X)
```

Perform forward pass of the convolutional layer.

Args:
    X: numpy array with shape (batch_size, in_channels, height, width)

Returns:
    Output feature maps after convolution and activation.

<a id="neural_networks.layers.ConvLayer.backward"></a>

#### backward

```python
def backward(d_out, reg_lambda=0)
```

Optimized backward pass using im2col technique.

Args:
    d_out: (np.ndarray) - Gradient of the loss with respect to the layer output,
                      shape (batch_size, out_channels, h_out, w_out)
    reg_lambda: (float, optional) - Regularization parameter.

Returns:
    dX: Gradient with respect to the input X.

<a id="neural_networks.layers.ConvLayer.activate"></a>

#### activate

```python
def activate(Z)
```

Apply activation function.

<a id="neural_networks.layers.RNNLayer"></a>

## RNNLayer Objects

```python
class RNNLayer()
```

Will be implemented later.

<a id="neural_networks.layers.RNNLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(input_size, hidden_size, activation="tanh")
```

Will be implemented later.

<a id="neural_networks.layers_cupy"></a>

# neural\_networks.layers\_cupy

<a id="neural_networks.layers_cupy.CuPyDenseLayer"></a>

## CuPyDenseLayer Objects

```python
class CuPyDenseLayer()
```

Initializes a Layer object.

Args:
    input_size (int): The size of the input to the layer.
    output_size (int): The size of the output from the layer.
    activation (str): The activation function to be used in the layer.

<a id="neural_networks.layers_cupy.CuPyDenseLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(input_size, output_size, activation="relu")
```

Initializes the layer with weights, biases, and activation function.

Args:
    input_size: (int) - The number of input features to the layer.
    output_size: (int) - The number of output features from the layer.
    activation: (str), optional - The activation function to use (default is "relu").
    Supported values: "relu", "leaky_relu", or others.

Attributes:
    weights: (cp.ndarray) - The weight matrix initialized using He initialization for "relu" or "leaky_relu".
    biases: (cp.ndarray) - The bias vector initialized to zeros.
    weight_gradients: (cp.ndarray) - Gradients of the weights, initialized to zeros.
    bias_gradients: (cp.ndarray) - Gradients of the biases, initialized to zeros.
    input_size: (int) - The number of input features to the layer.
    output_size: (int) - The number of output features from the layer.
    activation: (str) - The activation function used in the layer.

<a id="neural_networks.layers_cupy.CuPyDenseLayer.zero_grad"></a>

#### zero\_grad

```python
def zero_grad()
```

Reset the gradients of the weights and biases to zero.

<a id="neural_networks.layers_cupy.CuPyDenseLayer.activate"></a>

#### activate

```python
def activate(Z)
```

Apply activation function.

<a id="neural_networks.layers_cupy.CuPyDenseLayer.activation_derivative"></a>

#### activation\_derivative

```python
def activation_derivative(Z)
```

Apply activation derivative.

<a id="neural_networks.layers_cupy.CuPyActivation"></a>

## CuPyActivation Objects

```python
class CuPyActivation()
```

Activation functions for neural networks using CuPy.

<a id="neural_networks.layers_cupy.CuPyActivation.relu"></a>

#### relu

```python
@staticmethod
def relu(z)
```

ReLU (Rectified Linear Unit) activation function: f(z) = max(0, z).

Returns the input directly if it's positive, otherwise returns 0.

<a id="neural_networks.layers_cupy.CuPyActivation.relu_derivative"></a>

#### relu\_derivative

```python
@staticmethod
def relu_derivative(z)
```

Derivative of the ReLU function: f'(z) = 1 if z > 0, else 0.

Returns 1 for positive input, and 0 for negative input.

<a id="neural_networks.layers_cupy.CuPyActivation.leaky_relu"></a>

#### leaky\_relu

```python
@staticmethod
def leaky_relu(z, alpha=0.01)
```

Leaky ReLU activation function: f(z) = z if z > 0, else alpha * z.

Allows a small, non-zero gradient when the input is negative to address the dying ReLU problem.

<a id="neural_networks.layers_cupy.CuPyActivation.leaky_relu_derivative"></a>

#### leaky\_relu\_derivative

```python
@staticmethod
def leaky_relu_derivative(z, alpha=0.01)
```

Derivative of the Leaky ReLU function: f'(z) = 1 if z > 0, else alpha.

Returns 1 for positive input, and alpha for negative input.

<a id="neural_networks.layers_cupy.CuPyActivation.tanh"></a>

#### tanh

```python
@staticmethod
def tanh(z)
```

Hyperbolic tangent (tanh) activation function: f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z)).

Maps input to the range [-1, 1], typically used for normalized input.

<a id="neural_networks.layers_cupy.CuPyActivation.tanh_derivative"></a>

#### tanh\_derivative

```python
@staticmethod
def tanh_derivative(z)
```

Derivative of the tanh function: f'(z) = 1 - tanh(z)^2.

Used for backpropagation through the tanh activation.

<a id="neural_networks.layers_cupy.CuPyActivation.sigmoid"></a>

#### sigmoid

```python
@staticmethod
def sigmoid(z)
```

Sigmoid activation function: f(z) = 1 / (1 + exp(-z)).

Maps input to the range [0, 1], commonly used for binary classification.

<a id="neural_networks.layers_cupy.CuPyActivation.sigmoid_derivative"></a>

#### sigmoid\_derivative

```python
@staticmethod
def sigmoid_derivative(z)
```

Derivative of the sigmoid function: f'(z) = sigmoid(z) * (1 - sigmoid(z)).

Used for backpropagation through the sigmoid activation.

<a id="neural_networks.layers_cupy.CuPyActivation.softmax"></a>

#### softmax

```python
@staticmethod
def softmax(z)
```

Softmax activation function: f(z)_i = exp(z_i) / sum(exp(z_j)) for all j.

Maps input into a probability distribution over multiple classes. Used for multiclass classification.

<a id="neural_networks.layers_jit"></a>

# neural\_networks.layers\_jit

<a id="neural_networks.layers_jit.spec"></a>

#### spec

<a id="neural_networks.layers_jit.JITDenseLayer"></a>

## JITDenseLayer Objects

```python
@jitclass(spec)
class JITDenseLayer()
```

Initializes a fully connected layer object, where each neuron is connected to all neurons in the previous layer.

Each layer consists of weights, biases, and an activation function.

Args:
    input_size (int): The size of the input to the layer.
    output_size (int): The size of the output from the layer.
    activation (str): The activation function to be used in the layer.

Attributes:
    weights (np.ndarray): Weights of the layer.
    biases (np.ndarray): Biases of the layer.
    activation (str): Activation function name.
    weight_gradients (np.ndarray): Gradients of the weights.
    bias_gradients (np.ndarray): Gradients of the biases.
    input_cache (np.ndarray): Cached input for backpropagation.
    output_cache (np.ndarray): Cached output for backpropagation.

Methods:
    zero_grad(): Resets the gradients of the weights and biases to zero.
    forward(X): Performs the forward pass of the layer.
    backward(dA, reg_lambda): Performs the backward pass of the layer.
    activate(Z): Applies the activation function.
    activation_derivative(Z): Applies the derivative of the activation function.

<a id="neural_networks.layers_jit.JITDenseLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(input_size, output_size, activation="relu")
```

Initializes the layer with weights, biases, and activation function.

Args:
    input_size: (int) - The number of input features to the layer.
    output_size: (int) - The number of output features from the layer.
    activation: (str), optional - The activation function to use (default is "relu").

Attributes:
    weights: (np.ndarray) - The weight matrix initialized using He initialization for ReLU or Leaky ReLU,
                or Xavier initialization for other activations.
    biases: (np.ndarray) - The bias vector initialized to zeros.
    activation: (str) - The activation function for the layer.
    weight_gradients: (np.ndarray) - Gradients of the weights, initialized to zeros.
    bias_gradients: (np.ndarray) - Gradients of the biases, initialized to zeros.
    input_cache: (np.ndarray) - Cached input values for backpropagation, initialized to zeros.
    output_cache: (np.ndarray) - Cached output values for backpropagation, initialized to zeros.
    input_size: (int) - The number of input features to the layer.
    output_size: (int) - The number of output features from the layer.

<a id="neural_networks.layers_jit.JITDenseLayer.zero_grad"></a>

#### zero\_grad

```python
def zero_grad()
```

Reset the gradients of the weights and biases to zero.

<a id="neural_networks.layers_jit.JITDenseLayer.forward"></a>

#### forward

```python
def forward(X)
```

Perform the forward pass of the layer.

<a id="neural_networks.layers_jit.JITDenseLayer.backward"></a>

#### backward

```python
def backward(dA, reg_lambda)
```

Perform the backward pass of the layer.

<a id="neural_networks.layers_jit.JITDenseLayer.activate"></a>

#### activate

```python
def activate(Z)
```

Apply activation function.

<a id="neural_networks.layers_jit.JITDenseLayer.activation_derivative"></a>

#### activation\_derivative

```python
def activation_derivative(Z)
```

Apply activation derivative.

<a id="neural_networks.layers_jit.flatten_spec"></a>

#### flatten\_spec

<a id="neural_networks.layers_jit.JITFlattenLayer"></a>

## JITFlattenLayer Objects

```python
@jitclass(flatten_spec)
class JITFlattenLayer()
```

A layer that flattens multi-dimensional input into a 2D array (batch_size, flattened_size).

Useful for transitioning from convolutional layers to dense layers.

<a id="neural_networks.layers_jit.JITFlattenLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__()
```

Initializes the layer with placeholder values for input and output dimensions.

Attributes:
    input_shape: (tuple) - Shape of the input data, initialized to (0, 0, 0).
               This will be set during the forward pass.
    output_size: (int) - Size of the output, initialized to 0.
             This will be set during the forward pass.
    input_size: (int) - Size of the input, initialized to 0.
            This will be set during the forward pass.
    input_cache: (any) - Cache for input data, to be set during the forward pass.

<a id="neural_networks.layers_jit.JITFlattenLayer.forward"></a>

#### forward

```python
def forward(X)
```

Flattens the input tensor.

Args:
    X (np.ndarray): Input data of shape (batch_size, channels, height, width)

Returns:
    np.ndarray: Flattened output of shape (batch_size, flattened_size)

<a id="neural_networks.layers_jit.JITFlattenLayer.backward"></a>

#### backward

```python
def backward(dA, reg_lambda=0)
```

Reshapes the gradient back to the original input shape.

Args:
    dA (np.ndarray): Gradient of the loss with respect to the layer's output,
                   shape (batch_size, flattened_size)
    reg_lambda (float): Regularization parameter (unused in FlattenLayer).

Returns:
    np.ndarray: Gradient with respect to the input, reshaped to original input shape.

<a id="neural_networks.layers_jit.conv_spec"></a>

#### conv\_spec

<a id="neural_networks.layers_jit.JITConvLayer"></a>

## JITConvLayer Objects

```python
@jitclass(conv_spec)
class JITConvLayer()
```

A convolutional layer implementation for neural networks using Numba JIT compilation.

<a id="neural_networks.layers_jit.JITConvLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(in_channels,
             out_channels,
             kernel_size,
             stride=1,
             padding=0,
             activation="relu")
```

Initializes the convolutional layer with weights, biases, and activation function.

Args:
    in_channels: (int) - Number of input channels.
    out_channels: (int) - Number of output channels.
    kernel_size: (int) - Size of the convolutional kernel (assumes square kernels).
    stride: (int), optional - Stride of the convolution (default is 1).
    padding: (int), optional - Padding added to the input (default is 0).
    activation: (str), optional - Activation function to use (default is "relu").

Attributes:
    weights: (np.ndarray) - Convolutional weight matrix initialized using He initialization.
    biases: (np.ndarray) - Bias vector initialized to zeros.
    activation: (str) - Activation function for the layer.
    weight_gradients: (np.ndarray) - Gradients of the weights, initialized to zeros.
    bias_gradients: (np.ndarray) - Gradients of the biases, initialized to zeros.
    input_cache: (np.ndarray) - Cached input values for backpropagation, initialized to zeros.
    X_cols: (np.ndarray) - Cached column-transformed input for backpropagation, initialized to zeros.
    X_padded: (np.ndarray) - Cached padded input for backpropagation, initialized to zeros.
    h_out: (int) - Height of the output feature map, initialized to 0.
    w_out: (int) - Width of the output feature map, initialized to 0.
    input_size: (int) - Number of input channels.
    output_size: (int) - Number of output channels.

<a id="neural_networks.layers_jit.JITConvLayer.zero_grad"></a>

#### zero\_grad

```python
def zero_grad()
```

Reset the gradients of the weights and biases to zero.

<a id="neural_networks.layers_jit.JITConvLayer._im2col"></a>

#### \_im2col

```python
def _im2col(x, h_out, w_out)
```

Convert image regions to columns for efficient convolution.

<a id="neural_networks.layers_jit.JITConvLayer._col2im"></a>

#### \_col2im

```python
def _col2im(dcol, x_shape)
```

Convert column back to image format for the backward pass.

<a id="neural_networks.layers_jit.JITConvLayer.forward"></a>

#### forward

```python
def forward(X)
```

Forward pass for convolutional layer.

Args:
    X: numpy array with shape (batch_size, in_channels, height, width)

Returns:
    Output feature maps after convolution and activation.

<a id="neural_networks.layers_jit.JITConvLayer.backward"></a>

#### backward

```python
def backward(d_out, reg_lambda=0)
```

Backward pass for convolutional layer.

Args:
    d_out (np.ndarray): Gradient of the loss with respect to the layer output
    reg_lambda (float, optional): Regularization parameter

Returns:
    dX: Gradient with respect to the input X

<a id="neural_networks.layers_jit.JITConvLayer.activate"></a>

#### activate

```python
def activate(Z)
```

Apply activation function.

<a id="neural_networks.layers_jit.JITConvLayer.activation_derivative"></a>

#### activation\_derivative

```python
def activation_derivative(Z)
```

Apply activation derivative.

<a id="neural_networks.layers_jit.JITRNNLayer"></a>

## JITRNNLayer Objects

```python
class JITRNNLayer()
```

A recurrent layer implementation for neural networks using Numba JIT compilation.

<a id="neural_networks.layers_jit.JITRNNLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(input_size, hidden_size, activation="tanh")
```

Will be implemented later.

<a id="neural_networks.loss"></a>

# neural\_networks.loss

<a id="neural_networks.loss._validate_shapes"></a>

#### \_validate\_shapes

```python
def _validate_shapes(logits, targets)
```

Validate that logits and targets have compatible shapes.

<a id="neural_networks.loss.CrossEntropyLoss"></a>

## CrossEntropyLoss Objects

```python
class CrossEntropyLoss()
```

Custom cross entropy loss implementation using numpy for multi-class classification.

Formula: -sum(y * log(p) + (1 - y) * log(1 - p)) / m
Methods:
    __call__(self, logits, targets): Calculate the cross entropy loss.

<a id="neural_networks.loss.CrossEntropyLoss.__call__"></a>

#### \_\_call\_\_

```python
def __call__(logits, targets)
```

Calculate the cross entropy loss.

Args:
    logits (np.ndarray): The logits (predicted values) of shape (num_samples, num_classes).
    targets (np.ndarray): The target labels of shape (num_samples,).

Returns:
    float: The cross entropy loss.

<a id="neural_networks.loss.BCEWithLogitsLoss"></a>

## BCEWithLogitsLoss Objects

```python
class BCEWithLogitsLoss()
```

Custom binary cross entropy loss with logits implementation using numpy.

Formula: -mean(y * log(p) + (1 - y) * log(1 - p))

Methods:
    __call__(self, logits, targets): Calculate the binary cross entropy loss.

<a id="neural_networks.loss.BCEWithLogitsLoss.__call__"></a>

#### \_\_call\_\_

```python
def __call__(logits, targets)
```

Calculate the binary cross entropy loss.

Args:
    logits (np.ndarray): The logits (predicted values) of shape (num_samples,).
    targets (np.ndarray): The target labels of shape (num_samples,).

Returns:
    float: The binary cross entropy loss.

<a id="neural_networks.loss.MeanSquaredErrorLoss"></a>

## MeanSquaredErrorLoss Objects

```python
class MeanSquaredErrorLoss()
```

Custom mean squared error loss implementation using numpy.

Formula: mean((y_true - y_pred) ** 2)

Methods:
    __call__(self, y_true, y_pred): Calculate the mean squared error loss.

<a id="neural_networks.loss.MeanSquaredErrorLoss.__call__"></a>

#### \_\_call\_\_

```python
def __call__(y_true, y_pred)
```

Calculate the mean squared error loss.

Args:
    y_true (np.ndarray): The true labels of shape (num_samples,).
    y_pred (np.ndarray): The predicted values of shape (num_samples,).

Returns:
    float: The mean squared error loss.

<a id="neural_networks.loss.MeanAbsoluteErrorLoss"></a>

## MeanAbsoluteErrorLoss Objects

```python
class MeanAbsoluteErrorLoss()
```

Custom mean absolute error loss implementation using numpy.

Formula: mean(abs(y_true - y_pred))

Methods:
    __call__(self, y_true, y_pred): Calculate the mean absolute error loss.

<a id="neural_networks.loss.MeanAbsoluteErrorLoss.__call__"></a>

#### \_\_call\_\_

```python
def __call__(y_true, y_pred)
```

Calculate the mean absolute error loss.

Args:
    y_true (np.ndarray): The true labels of shape (num_samples,).
    y_pred (np.ndarray): The predicted values of shape (num_samples,).

Returns:
    float: The mean absolute error loss.

<a id="neural_networks.loss.HuberLoss"></a>

## HuberLoss Objects

```python
class HuberLoss()
```

Custom Huber loss implementation using numpy.

Formula: mean(0.5 * (y_true - y_pred)**2) if abs(y_true - y_pred) <= delta else mean(delta * (abs(y_true - y_pred) - delta / 2))

Methods:
    __call__(self, y_true, y_pred, delta=1.0): Calculate the Huber loss.

<a id="neural_networks.loss.HuberLoss.__call__"></a>

#### \_\_call\_\_

```python
def __call__(y_true, y_pred, delta=1.0)
```

Calculate the Huber loss.

Args:
    y_true (np.ndarray): The true labels of shape (num_samples,).
    y_pred (np.ndarray): The predicted values of shape (num_samples,).
    delta (float): The threshold for the Huber loss.

Returns:
    float: The Huber loss.

<a id="neural_networks.loss_cupy"></a>

# neural\_networks.loss\_cupy

<a id="neural_networks.loss_cupy.CuPyCrossEntropyLoss"></a>

## CuPyCrossEntropyLoss Objects

```python
class CuPyCrossEntropyLoss()
```

Optimized cross entropy loss implementation using cupy for multi-class classification.

Formula: -sum(y * log(p)) / m
Methods:
    __call__(self, logits, targets): Calculate the cross entropy loss.

<a id="neural_networks.loss_cupy.CuPyCrossEntropyLoss.__call__"></a>

#### \_\_call\_\_

```python
def __call__(logits, targets)
```

Calculate the cross entropy loss.

Args:
    logits (cp.ndarray): The logits (predicted values) of shape (num_samples, num_classes).
    targets (cp.ndarray): The target labels of shape (num_samples, num_classes) or (num_samples,).

Returns:
    float: The cross entropy loss.

<a id="neural_networks.loss_cupy.CuPyBCEWithLogitsLoss"></a>

## CuPyBCEWithLogitsLoss Objects

```python
class CuPyBCEWithLogitsLoss()
```

Optimized binary cross entropy loss with logits implementation using cupy.

Formula: -mean(y * log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x)))

Methods:
    __call__(self, logits, targets): Calculate the binary cross entropy loss.

<a id="neural_networks.loss_cupy.CuPyBCEWithLogitsLoss.__call__"></a>

#### \_\_call\_\_

```python
def __call__(logits, targets)
```

Calculate the binary cross entropy loss.

Args:
    logits (cp.ndarray): The logits (predicted values) of shape (num_samples,).
    targets (cp.ndarray): The target labels of shape (num_samples,).

Returns:
    float: The binary cross entropy loss.

<a id="neural_networks.loss_jit"></a>

# neural\_networks.loss\_jit

<a id="neural_networks.loss_jit.CACHE"></a>

#### CACHE

<a id="neural_networks.loss_jit._validate_shapes"></a>

#### \_validate\_shapes

```python
def _validate_shapes(logits, targets)
```

Validate that logits and targets have compatible shapes.

<a id="neural_networks.loss_jit.JITCrossEntropyLoss"></a>

## JITCrossEntropyLoss Objects

```python
class JITCrossEntropyLoss()
```

Custom cross entropy loss implementation using numba for multi-class classification.

Formula: -sum(y * log(p) + (1 - y) * log(1 - p)) / m
Methods:
    calculate_loss(self, logits, targets): Calculate the cross entropy loss.

<a id="neural_networks.loss_jit.JITCrossEntropyLoss.__init__"></a>

#### \_\_init\_\_

```python
def __init__()
```

Initializes the instance variables for the class.

Args:
    logits: (np.ndarray) - A 2D array initialized to zeros with shape (1, 1),
               representing the predicted values or outputs of the model.
    targets: (np.ndarray) - A 2D array initialized to zeros with shape (1, 1),
                representing the ground truth or target values.

<a id="neural_networks.loss_jit.JITCrossEntropyLoss.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(logits, targets)
```

Calculate the cross entropy loss.

Args:
    logits (np.ndarray): The logits (predicted values) of shape (num_samples, num_classes).
    targets (np.ndarray): The target labels of shape (num_samples,).

Returns:
    float: The cross entropy loss.

<a id="neural_networks.loss_jit.calculate_cross_entropy_loss"></a>

#### calculate\_cross\_entropy\_loss

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_cross_entropy_loss(logits, targets)
```

Helper function to calculate the cross entropy loss.

<a id="neural_networks.loss_jit.JITBCEWithLogitsLoss"></a>

## JITBCEWithLogitsLoss Objects

```python
class JITBCEWithLogitsLoss()
```

Custom binary cross entropy loss with logits implementation using numba.

Formula: -mean(y * log(p) + (1 - y) * log(1 - p))

Methods:
    calculate_loss(self, logits, targets): Calculate the binary cross entropy loss.

<a id="neural_networks.loss_jit.JITBCEWithLogitsLoss.__init__"></a>

#### \_\_init\_\_

```python
def __init__()
```

Initializes the class with default values for logits and targets.

Attributes:
    logits (numpy.ndarray): A 2D array initialized to zeros with shape (1, 1),
                            representing the predicted values.
    targets (numpy.ndarray): A 2D array initialized to zeros with shape (1, 1),
                             representing the true target values.

<a id="neural_networks.loss_jit.JITBCEWithLogitsLoss.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(logits, targets)
```

Calculate the binary cross entropy loss.

Args:
    logits (np.ndarray): The logits (predicted values) of shape (num_samples,).
    targets (np.ndarray): The target labels of shape (num_samples,).

Returns:
    float: The binary cross entropy loss.

<a id="neural_networks.loss_jit.calculate_bce_with_logits_loss"></a>

#### calculate\_bce\_with\_logits\_loss

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_bce_with_logits_loss(logits, targets)
```

Helper function to calculate the binary cross entropy loss.

<a id="neural_networks.loss_jit.JITMeanSquaredErrorLoss"></a>

## JITMeanSquaredErrorLoss Objects

```python
class JITMeanSquaredErrorLoss()
```

Custom mean squared error loss implementation using numba.

<a id="neural_networks.loss_jit.JITMeanSquaredErrorLoss.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(y_pred, y_true)
```

Calculate the mean squared error loss.

<a id="neural_networks.loss_jit.JITMeanAbsoluteErrorLoss"></a>

## JITMeanAbsoluteErrorLoss Objects

```python
class JITMeanAbsoluteErrorLoss()
```

Custom mean absolute error loss implementation using numba.

<a id="neural_networks.loss_jit.JITMeanAbsoluteErrorLoss.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(y_pred, y_true)
```

Calculate the mean absolute error loss.

<a id="neural_networks.loss_jit.JITHuberLoss"></a>

## JITHuberLoss Objects

```python
class JITHuberLoss()
```

Custom Huber loss implementation using numba.

Attributes:
    delta (float): The threshold parameter for Huber loss. Default is 1.0.

<a id="neural_networks.loss_jit.JITHuberLoss.__init__"></a>

#### \_\_init\_\_

```python
def __init__(delta=1.0)
```

Initializes the JITHuberLoss instance.

Args:
    delta (float): The threshold at which the loss function transitions
                   from quadratic to linear. Default is 1.0.

<a id="neural_networks.loss_jit.JITHuberLoss.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(y_pred, y_true)
```

Calculate the Huber loss using the stored delta.

Args:
    y_pred (np.ndarray): Predicted values.
    y_true (np.ndarray): True target values.

Returns:
    float: The calculated Huber loss.

<a id="neural_networks.neuralNetworkBase"></a>

# neural\_networks.neuralNetworkBase

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase"></a>

## NeuralNetworkBase Objects

```python
class NeuralNetworkBase()
```

NeuralNetworkBase is an abstract base class for building neural networks.

It provides a framework for initializing layers, performing forward and backward propagation,
training, evaluating, and predicting with a neural network. Subclasses should implement
the abstract methods to define specific behavior.

Attributes:
    layer_sizes (list): Sizes of the layers in the network.
    dropout_rate (float): Dropout rate for regularization.
    reg_lambda (float): Regularization strength for L2 regularization.
    activations (list): Activation functions for each layer.
    layers (list): List of layer objects or configurations.
    weights (list): List of weight matrices for the layers.
    biases (list): List of bias vectors for the layers.
    layer_outputs (ndarray): Outputs of each layer during forward propagation.
    is_binary (bool): Whether the network is for binary classification.

Methods:
    __init__(layers, dropout_rate=0.0, reg_lambda=0.0, activations=None, loss_function=None, regressor=False):
        Initializes the neural network with the given layers and parameters.
    initialize_layers():
        Abstract method to initialize the weights and biases of the layers.
    forward(X, training=True):
        Abstract method to perform forward propagation through the network.
    backward(y):
        Abstract method to perform backward propagation through the network.
    train(X_train, y_train, X_val=None, y_val=None, optimizer=None, epochs=100,
            batch_size=32, early_stopping_threshold=10, lr_scheduler=None, p=True,
            use_tqdm=True, n_jobs=1, track_metrics=False, track_adv_metrics=False):
        Abstract method to train the neural network using the provided training data.
    evaluate(X, y):
        Abstract method to evaluate the neural network on the provided data.
    predict(X):
        Abstract method to make predictions using the trained neural network.
    calculate_loss(X, y):
        Abstract method to calculate the loss of the neural network.
    apply_dropout(X):
        Applies dropout to the activation values for regularization.
    compute_l2_reg(weights):
        Computes the L2 regularization term for the given weights.
    calculate_precision_recall_f1(X, y):
        Calculates precision, recall, and F1 score for the predictions.
    create_scheduler(scheduler_type, optimizer, **kwargs):
        Creates a learning rate scheduler based on the specified type.
    plot_metrics(save_dir=None):
        Plots the training and validation metrics, including loss, accuracy,
        learning rate, and optionally precision, recall, and F1 score.

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.__init__"></a>

#### \_\_init\_\_

```python
def __init__(layers,
             dropout_rate=0.0,
             reg_lambda=0.0,
             activations=None,
             loss_function=None,
             regressor=False)
```

Initializes the neural network with the specified layers, dropout rate, regularization, and activations.

Args:
    layers: (list) - A list of integers representing the sizes of each layer or a list of Layer objects.
    dropout_rate: (float), optional - The dropout rate for regularization (default is 0.0).
    reg_lambda: (float), optional - The regularization strength (default is 0.0).
    activations: (list of str), optional - A list of activation functions for each layer (default is None, which sets "relu" for hidden layers and "softmax" for the output layer).
    loss_function: (callable), optional - Custom loss function to use (default is None, which uses the default calculate_loss implementation).
    regressor: (bool), optional - If True, the network is treated as a regressor (default is False).

Raises:
    ValueError: If `layers` is not a list of integers or a list of Layer objects.

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.initialize_layers"></a>

#### initialize\_layers

```python
def initialize_layers()
```

Initializes the weights and biases of the layers.

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.forward"></a>

#### forward

```python
def forward(X, training=True)
```

Performs forward propagation through the network.

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.backward"></a>

#### backward

```python
def backward(y)
```

Performs backward propagation through the network.

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.train"></a>

#### train

```python
def train(X_train,
          y_train,
          X_val=None,
          y_val=None,
          optimizer=None,
          epochs=100,
          batch_size=32,
          early_stopping_threshold=10,
          lr_scheduler=None,
          p=True,
          use_tqdm=True,
          n_jobs=1,
          track_metrics=False,
          track_adv_metrics=False)
```

Trains the neural network using the provided training data.

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.evaluate"></a>

#### evaluate

```python
def evaluate(X, y)
```

Evaluates the neural network on the provided data.

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.predict"></a>

#### predict

```python
def predict(X)
```

Makes predictions using the trained neural network.

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(X, y)
```

Calculates the loss of the neural network.

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.apply_dropout"></a>

#### apply\_dropout

```python
def apply_dropout(X)
```

Applies dropout to the activation X.

Args:
    X: (ndarray) - Activation values.

Returns:
    ndarray: Activation values after applying dropout.

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.compute_l2_reg"></a>

#### compute\_l2\_reg

```python
def compute_l2_reg(weights)
```

Computes the L2 regularization term.

Args:
    weights: (list) - List of weight matrices.

Returns:
    float: L2 regularization term.

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.calculate_precision_recall_f1"></a>

#### calculate\_precision\_recall\_f1

```python
def calculate_precision_recall_f1(X, y)
```

Calculates precision, recall, and F1 score.

Args:
    X: (ndarray) - Input data
    y: (ndarray) - Target labels
Returns:
    precision: (float) - Precision score
    recall: (float) - Recall score
    f1: (float) - F1 score

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.create_scheduler"></a>

#### create\_scheduler

```python
def create_scheduler(scheduler_type, optimizer, **kwargs)
```

Creates a learning rate scheduler.

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.plot_metrics"></a>

#### plot\_metrics

```python
def plot_metrics(save_dir=None)
```

Plots the training and validation metrics.

<a id="neural_networks.neuralNetworkBaseBackend"></a>

# neural\_networks.neuralNetworkBaseBackend

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork"></a>

## BaseBackendNeuralNetwork Objects

```python
class BaseBackendNeuralNetwork(NeuralNetworkBase)
```

A class representing a backend implementation of a neural network with support for forward propagation, backward propagation, training, evaluation, and hyperparameter tuning.

This class extends the `NeuralNetworkBase` class and provides additional functionality
for managing layers, applying dropout, calculating loss, and optimizing weights and biases.

Attributes:
    layers (list): List of layer objects in the neural network.
    layer_outputs (list): Outputs of each layer during forward propagation.
    weights (list): Weights of each layer.
    biases (list): Biases of each layer.
    train_loss (list): Training loss values over epochs.
    train_accuracy (list): Training accuracy values over epochs.
    val_loss (list): Validation loss values over epochs.
    val_accuracy (list): Validation accuracy values over epochs.
    train_precision (list): Training precision values over epochs.
    train_recall (list): Training recall values over epochs.
    train_f1 (list): Training F1-score values over epochs.
    val_precision (list): Validation precision values over epochs.
    val_recall (list): Validation recall values over epochs.
    val_f1 (list): Validation F1-score values over epochs.
    learning_rates (list): Learning rates over epochs.

Methods:
    __init__(layers, dropout_rate=0.2, reg_lambda=0.01, activations=None, loss_function=None, regressor=False):
        Initializes the neural network with the specified layers, dropout rate,
        regularization parameter, activation functions, and optional loss function.
    initialize_new_layers():
        Initializes the layers of the neural network with random weights and biases.
    forward(X, training=True):
        Performs forward propagation through the neural network.
    backward(y):
        Performs backward propagation to calculate gradients for weight and bias updates.
    fit(X_train, y_train, X_val=None, y_val=None, optimizer=None, epochs=100, batch_size=32, ...):
        Fits the neural network to the training data.
    train(X_train, y_train, X_val=None, y_val=None, optimizer=None, epochs=100, batch_size=32, ...):
        Trains the neural network model with optional validation and early stopping.
    evaluate(X, y):
        Evaluates the model on the given data and returns accuracy and predictions.
    predict(X):
        Predicts the output for the given input data.
    calculate_loss(X, y):
        Calculates the loss with L2 regularization for the given input and target labels.
    _create_optimizer(optimizer_type, learning_rate, JIT=False):
        Helper method to create optimizer instances based on the specified type.
    tune_hyperparameters(X_train, y_train, X_val, y_val, param_grid, ...):
        Performs hyperparameter tuning using grid search.
    train_with_animation_capture(X_train, y_train, X_val=None, y_val=None, ...):
        Trains the neural network while capturing training metrics in real-time animation.

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.__init__"></a>

#### \_\_init\_\_

```python
def __init__(layers,
             dropout_rate=0.2,
             reg_lambda=0.01,
             activations=None,
             loss_function=None,
             regressor=False)
```

Initializes the Numba backend neural network.

Args:
    layers: (list) - List of layer sizes or Layer objects.
    dropout_rate: (float) - Dropout rate for regularization.
    reg_lambda: (float) - L2 regularization parameter.
    activations: (list) - List of activation functions for each layer.
    loss_function: (callable) optional - Custom loss function to use (default is None, which uses the default calculate_loss implementation).
    regressor: (bool) - Whether the model is a regressor (default is False).

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.initialize_new_layers"></a>

#### initialize\_new\_layers

```python
def initialize_new_layers()
```

Initializes the layers of the neural network.

Each layer is created with the specified number of neurons and activation function.

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.forward"></a>

#### forward

```python
def forward(X, training=True)
```

Performs forward propagation through the neural network.

Args:
    X: (ndarray): - Input data of shape (batch_size, input_size).
    training: (bool) - Whether the network is in training mode (applies dropout).

Returns:
    ndarray: Output predictions of shape (batch_size, output_size).

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.backward"></a>

#### backward

```python
def backward(y)
```

Performs backward propagation to calculate the gradients.

Args:
    y: (ndarray) - Target labels of shape (m, output_size).

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.fit"></a>

#### fit

```python
def fit(X_train,
        y_train,
        X_val=None,
        y_val=None,
        optimizer=None,
        epochs=100,
        batch_size=32,
        early_stopping_threshold=10,
        lr_scheduler=None,
        p=False,
        use_tqdm=False,
        n_jobs=1,
        track_metrics=False,
        track_adv_metrics=False,
        save_animation=False,
        save_path="training_animation.mp4",
        fps=1,
        dpi=100,
        frame_every=1)
```

Fits the neural network to the training data.

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.train"></a>

#### train

```python
def train(X_train,
          y_train,
          X_val=None,
          y_val=None,
          optimizer=None,
          epochs=100,
          batch_size=32,
          early_stopping_threshold=10,
          lr_scheduler=None,
          p=True,
          use_tqdm=True,
          n_jobs=1,
          track_metrics=False,
          track_adv_metrics=False,
          save_animation=False,
          save_path="training_animation.mp4",
          fps=1,
          dpi=100,
          frame_every=1)
```

Trains the neural network model.

Args:
    X_train: (ndarray) - Training data features.
    y_train: (ndarray) - Training data labels.
    X_val: (ndarray) - Validation data features, optional.
    y_val: (ndarray) - Validation data labels, optional.
    optimizer: (Optimizer) - Optimizer for updating parameters (default: Adam, lr=0.0001).
    epochs: (int) - Number of training epochs (default: 100).
    batch_size: (int) - Batch size for mini-batch gradient descent (default: 32).
    early_stopping_threshold: (int) - Patience for early stopping (default: 10).
    lr_scheduler: (Scheduler) - Learning rate scheduler (default: None).
    p: (bool) - Whether to print training progress (default: True).
    use_tqdm: (bool) - Whether to use tqdm for progress bar (default: True).
    n_jobs: (int) - Number of jobs for parallel processing (default: 1).
    track_metrics: (bool) - Whether to track training metrics (default: False).
    track_adv_metrics: (bool) - Whether to track advanced metrics (default: False).
    save_animation: (bool) - Whether to save the animation of metrics (default: False).
    save_path: (str) - Path to save the animation file. File extension must be .mp4 or .gif (default: 'training_animation.mp4').
    fps: (int) - Frames per second for the saved animation (default: 1).
    dpi: (int) - DPI for the saved animation (default: 100).
    frame_every: (int) - Capture frame every N epochs (to reduce file size) (default: 1).

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.evaluate"></a>

#### evaluate

```python
def evaluate(X, y)
```

Evaluates the model's performance on the given data.

Args:
    X: (np.ndarray) - Input feature data for evaluation.
    y: (np.ndarray) - True target labels corresponding to the input data.

Returns:
    metric: (float) - The evaluation metric (accuracy for classification, MSE for regression).
    predicted: (np.ndarray) - The predicted labels or values for the input data.

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.predict"></a>

#### predict

```python
def predict(X)
```

Generates predictions for the given input data.

Args:
    X: (np.ndarray) - Input feature data for which predictions are to be made.

Returns:
    outputs: (np.ndarray) - Predicted outputs. If the model is binary, returns the raw outputs.
                Otherwise, returns the class indices with the highest probability.

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(X, y)
```

Calculates the loss with L2 regularization.

Args:
    X: (np.ndarray) - Input feature data.
    y: (np.ndarray) - Target labels.

Returns:
    loss: (float) - The calculated loss value with L2 regularization.

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork._create_optimizer"></a>

#### \_create\_optimizer

```python
def _create_optimizer(optimizer_type, learning_rate, JIT=False)
```

Helper method to create optimizer instances.

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.tune_hyperparameters"></a>

#### tune\_hyperparameters

```python
def tune_hyperparameters(X_train,
                         y_train,
                         X_val,
                         y_val,
                         param_grid,
                         layer_configs=None,
                         optimizer_types=None,
                         lr_range=(0.0001, 0.01, 5),
                         epochs=30,
                         batch_size=32)
```

Performs hyperparameter tuning using grid search.

Args:
    X_train: (np.ndarray) - Training feature data.
    y_train: (np.ndarray) - Training target data.
    X_val: (np.ndarray) - Validation feature data.
    y_val: (np.ndarray) - Validation target data.
    param_grid: (dict) - Dictionary of parameters to try.
    layer_configs: (list), optional - List of layer configurations (default is None).
    optimizer_types: (list), optional - List of optimizer types (default is None).
    lr_range: (tuple) - Tuple of (min_lr, max_lr, num_steps) for learning rates.
    epochs: (int) - Maximum epochs for each trial.
    batch_size: (int) - Batch size for training.

Returns:
    best_params: (dict) - Best hyperparameters found.
    best_accuracy: (float) - Best validation accuracy.

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.train_with_animation_capture"></a>

#### train\_with\_animation\_capture

```python
def train_with_animation_capture(X_train,
                                 y_train,
                                 X_val=None,
                                 y_val=None,
                                 optimizer=None,
                                 epochs=100,
                                 batch_size=32,
                                 early_stopping_threshold=10,
                                 lr_scheduler=None,
                                 save_path="training_animation.mp4",
                                 fps=1,
                                 dpi=100,
                                 frame_every=1)
```

Trains the neural network model while capturing training metrics in real-time animation.

Args:
    X_train: (np.ndarray) - Training feature data.
    y_train: (np.ndarray) - Training target data.
    X_val: (np.ndarray), optional - Validation feature data (default is None).
    y_val: (np.ndarray), optional - Validation target data (default is None).
    optimizer: (Optimizer), optional - Optimizer for updating parameters (default is None).
    epochs: (int), optional - Number of training epochs (default is 100).
    batch_size: (int), optional - Batch size for mini-batch gradient descent (default is 32).
    early_stopping_threshold: (int), optional - Patience for early stopping (default is 10).
    lr_scheduler: (Scheduler), optional - Learning rate scheduler (default is None).
    save_path: (str), optional - Path to save the animation file (default is 'training_animation.mp4').
    fps: (int), optional - Frames per second for the saved animation (default is 1).
    dpi: (int), optional - DPI for the saved animation (default is 100).
    frame_every: (int), optional - Capture frame every N epochs (default is 1).

Returns:
    None

<a id="neural_networks.neuralNetworkCuPyBackend"></a>

# neural\_networks.neuralNetworkCuPyBackend

<a id="neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork"></a>

## CuPyBackendNeuralNetwork Objects

```python
class CuPyBackendNeuralNetwork(NeuralNetworkBase)
```

CuPyBackendNeuralNetwork is a neural network implementation that uses CuPy for GPU-accelerated computations.

It inherits from NeuralNetworkBase and provides functionality for forward and backward propagation,
training, evaluation, and optimization using CuPy arrays and operations.

Attributes:
    layers (list): List of layers in the neural network.
    compiled (bool): Indicates whether the network is compiled.
    trainable_layers (list): List of layers with trainable parameters.
    layer_outputs (list): Cache for forward pass outputs.
    is_binary (bool): Indicates if the network is for binary classification.
    weights (list): List of weights for trainable layers.
    biases (list): List of biases for trainable layers.
    dWs_cache (list): Cache for weight gradients.
    dbs_cache (list): Cache for bias gradients.
    stream_pool_size (int): Number of CUDA streams for asynchronous processing.
    stream_pool (list): Pool of CUDA streams for asynchronous operations.

Methods:
    __init__(layers, dropout_rate=0.2, reg_lambda=0.01, activations=None):
        Initializes the CuPyBackendNeuralNetwork with specified layers, dropout rate, regularization, and activations.
    initialize_new_layers():
        Initializes the layers of the neural network with specified sizes and activation functions.
    apply_dropout(X):
        Applies dropout regularization to the input data.
    forward(X, training=True):
        Performs forward propagation through the neural network.
    backward(y):
        Performs backward propagation to calculate gradients for weights and biases.
    _process_batches_async(X_shuffled, y_shuffled, batch_size, weights, biases, activations, dropout_rate, is_binary, reg_lambda, dWs_acc, dbs_acc):
        Processes batches asynchronously using CUDA streams for forward and backward propagation.
    is_not_instance_of_classes(obj, classes):
        Checks if an object is not an instance of any class in a given list of classes.
    train(X_train, y_train, X_val=None, y_val=None, optimizer=None, epochs=100, batch_size=32, early_stopping_threshold=10, lr_scheduler=None, p=True, use_tqdm=True, n_jobs=1, track_metrics=False, track_adv_metrics=False, save_animation=False, save_path="training_animation.mp4", fps=1, dpi=100, frame_every=1):
        Trains the neural network model with specified parameters and options.
    evaluate(X, y):
        Evaluates the model performance on the given input data and labels.
    _evaluate_cupy(y_hat, y_true, is_binary):
        Evaluates model performance using CuPy arrays for predictions and true labels.
    predict(X):
        Predicts the output for the given input data.
    calculate_loss(X, y):
        Calculates the loss with L2 regularization for the given input data and labels.
    _create_optimizer(optimizer_type, learning_rate, JIT=False):
        Helper method to create optimizer instances based on the specified type and learning rate.

<a id="neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.__init__"></a>

#### \_\_init\_\_

```python
def __init__(layers, dropout_rate=0.2, reg_lambda=0.01, activations=None)
```

Initializes the CuPy backend neural network.

Args:
    layers: (list) - List of layer sizes or Layer objects.
    dropout_rate: (float) - Dropout rate for regularization (default is 0.2).
    reg_lambda: (float) - L2 regularization parameter (default is 0.01).
    activations: (list), optional - List of activation functions for each layer (default is None).

Returns:
    None

<a id="neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.initialize_new_layers"></a>

#### initialize\_new\_layers

```python
def initialize_new_layers()
```

Initializes the layers of the neural network.

Each layer is created with the specified number of neurons and activation function.

<a id="neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.apply_dropout"></a>

#### apply\_dropout

```python
def apply_dropout(X)
```

Applies dropout regularization to the input data.

<a id="neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.forward"></a>

#### forward

```python
def forward(X, training=True)
```

Performs forward propagation through the neural network.

Args:
    X (ndarray): Input data of shape (batch_size, input_size).
    training (bool): Whether the network is in training mode (applies dropout).

Returns:
    ndarray: Output predictions of shape (batch_size, output_size).

<a id="neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.backward"></a>

#### backward

```python
def backward(y)
```

Performs backward propagation to calculate the gradients.

Args:
    y (ndarray): Target labels of shape (m, output_size).

<a id="neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork._process_batches_async"></a>

#### \_process\_batches\_async

```python
def _process_batches_async(X_shuffled, y_shuffled, batch_size, weights, biases,
                           activations, dropout_rate, is_binary, reg_lambda,
                           dWs_acc, dbs_acc)
```

<a id="neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.is_not_instance_of_classes"></a>

#### is\_not\_instance\_of\_classes

```python
@staticmethod
def is_not_instance_of_classes(obj, classes)
```

Checks if an object is not an instance of any class in a list of classes.

Args:
    obj: The object to check.
    classes: A list of classes.

Returns:
    bool: True if the object is not an instance of any class in the list of classes, False otherwise.

<a id="neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.train"></a>

#### train

```python
def train(X_train,
          y_train,
          X_val=None,
          y_val=None,
          optimizer=None,
          epochs=100,
          batch_size=32,
          early_stopping_threshold=10,
          lr_scheduler=None,
          p=True,
          use_tqdm=True,
          n_jobs=1,
          track_metrics=False,
          track_adv_metrics=False,
          save_animation=False,
          save_path="training_animation.mp4",
          fps=1,
          dpi=100,
          frame_every=1)
```

Trains the neural network model.

Args:
    X_train: (ndarray) - Training data features.
    y_train: (ndarray) - Training data labels.
    X_val: (ndarray) - Validation data features, optional.
    y_val: (ndarray) - Validation data labels, optional.
    optimizer: (Optimizer) - Optimizer for updating parameters (default: JITAdam, lr=0.0001).
    epochs: (int) - Number of training epochs (default: 100).
    batch_size: (int) - Batch size for mini-batch gradient descent (default: 32).
    early_stopping_threshold: (int) - Patience for early stopping (default: 10).
    lr_scheduler: (Scheduler) - Learning rate scheduler (default: None).
    p: (bool) - Whether to print training progress (default: True).
    use_tqdm: (bool) - Whether to use tqdm for progress bar (default: True).
    n_jobs: (int) - Number of jobs for parallel processing (default: 1).
    track_metrics: (bool) - Whether to track training metrics (default: False).
    track_adv_metrics: (bool) - Whether to track advanced metrics (default: False).
    save_animation: (bool) - Whether to save the animation of metrics (default: False).
    save_path: (str) - Path to save the animation file. File extension must be .mp4 or .gif (default: 'training_animation.mp4').
    fps: (int) - Frames per second for the saved animation (default: 1).
    dpi: (int) - DPI for the saved animation (default: 100).
    frame_every: (int) - Capture frame every N epochs (to reduce file size) (default: 1).

<a id="neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.evaluate"></a>

#### evaluate

```python
def evaluate(X, y)
```

Evaluates the model performance on the given data.

Args:
    X: (np.ndarray or cp.ndarray) - Input feature data.
    y: (np.ndarray or cp.ndarray) - Target labels.

Returns:
    accuracy: (float) - The accuracy of the model.
    predicted: (np.ndarray) - Predicted labels as a NumPy array.

<a id="neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork._evaluate_cupy"></a>

#### \_evaluate\_cupy

```python
@staticmethod
def _evaluate_cupy(y_hat, y_true, is_binary)
```

CuPy-based function to evaluate model performance.

Args:
    y_hat (cp.ndarray): Model predictions (CuPy array).
    y_true (cp.ndarray): True labels (CuPy array).
    is_binary (bool): Whether the model is binary or multi-class.

Returns:
    tuple: Accuracy (CuPy scalar) and predicted labels (CuPy array).

<a id="neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.predict"></a>

#### predict

```python
def predict(X)
```

Predicts the output for the given input data.

Args:
    X (ndarray): Input data.

Returns:
    ndarray: Predicted outputs.

<a id="neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(X, y)
```

Calculates the loss with L2 regularization.

Args:
    X (ndarray): Input data.
    y (ndarray): Target labels.

Returns:
    float: The calculated loss value.

<a id="neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork._create_optimizer"></a>

#### \_create\_optimizer

```python
def _create_optimizer(optimizer_type, learning_rate, JIT=False)
```

Helper method to create optimizer instances.

<a id="neural_networks.neuralNetworkNumbaBackend"></a>

# neural\_networks.neuralNetworkNumbaBackend

<a id="neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork"></a>

## NumbaBackendNeuralNetwork Objects

```python
class NumbaBackendNeuralNetwork(NeuralNetworkBase)
```

A neural network implementation using Numba for Just-In-Time (JIT) compilation to optimize performance.

This class supports forward and backward propagation, training, evaluation, and hyperparameter tuning
with various optimizers and activation functions.

Attributes:
    compiled (bool): Indicates whether Numba functions are compiled.
    trainable_layers (list): Layers with trainable parameters (weights and biases).
    progress_bar (bool): Whether to display a progress bar during training.

Methods:
    __init__(layers, dropout_rate, reg_lambda, activations, compile_numba, progress_bar):
        Initializes the neural network with the specified parameters.
    store_init_layers():
        Stores the initial layers and their parameters for restoration after initialization.
    restore_layers():
        Restores the layers and their parameters after initialization.
    initialize_new_layers():
        Initializes the layers of the neural network with specified sizes and activation functions.
    forward(X, training):
        Performs forward propagation through the neural network.
    backward(y):
        Performs backward propagation to calculate gradients.
    is_not_instance_of_classes(obj, classes):
        Checks if an object is not an instance of any class in a list of classes.
    train(X_train, y_train, X_val, y_val, optimizer, epochs, batch_size, early_stopping_threshold,
          lr_scheduler, p, use_tqdm, n_jobs, track_metrics, track_adv_metrics, save_animation,
          save_path, fps, dpi, frame_every):
        Trains the neural network model with the specified parameters.
    evaluate(X, y):
        Evaluates the neural network on the given data and returns accuracy and predictions.
    predict(X):
        Predicts the output for the given input data.
    calculate_loss(X, y):
        Calculates the loss with L2 regularization.
    _create_optimizer(optimizer_type, learning_rate, JIT):
        Helper method to create optimizer instances.
    tune_hyperparameters(X_train, y_train, X_val, y_val, param_grid, layer_configs, optimizer_types,
                         lr_range, epochs, batch_size):
        Performs hyperparameter tuning using grid search.
    compile_numba_functions(progress_bar):
        Compiles all Numba JIT functions to improve performance.

<a id="neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.__init__"></a>

#### \_\_init\_\_

```python
def __init__(layers,
             dropout_rate=0.2,
             reg_lambda=0.01,
             activations=None,
             loss_function=None,
             regressor=False,
             compile_numba=True,
             progress_bar=True)
```

Initializes the Numba backend neural network.

Args:
    layers: (list) - List of layer sizes or Layer objects.
    dropout_rate: (float) - Dropout rate for regularization.
    reg_lambda: (float) - L2 regularization parameter.
    activations: (list) - List of activation functions for each layer.
    loss_function: (callable) optional - Custom loss function (default: selects based on task).
    regressor: (bool) - Whether the model is a regressor (default is False).
    compile_numba: (bool) - Whether to compile Numba functions.
    progress_bar: (bool) - Whether to display a progress bar.

<a id="neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.store_init_layers"></a>

#### store\_init\_layers

```python
def store_init_layers()
```

Stores the layers to restore after initialization.

<a id="neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.restore_layers"></a>

#### restore\_layers

```python
def restore_layers()
```

Restores the layers after initialization.

<a id="neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.initialize_new_layers"></a>

#### initialize\_new\_layers

```python
def initialize_new_layers()
```

Initializes the layers of the neural network.

Each layer is created with the specified number of neurons and activation function.

<a id="neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.forward"></a>

#### forward

```python
def forward(X, training=True)
```

Performs forward propagation through the neural network.

Args:
    X (ndarray): Input data of shape (batch_size, input_size).
    training (bool): Whether the network is in training mode (applies dropout).

Returns:
    ndarray: Output predictions of shape (batch_size, output_size).

<a id="neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.backward"></a>

#### backward

```python
def backward(y)
```

Performs backward propagation to calculate the gradients.

Args:
    y (ndarray): Target labels of shape (m, output_size).

<a id="neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.is_not_instance_of_classes"></a>

#### is\_not\_instance\_of\_classes

```python
@staticmethod
def is_not_instance_of_classes(obj, classes)
```

Checks if an object is not an instance of any class in a list of classes.

Args:
    obj: The object to check.
    classes: A list of classes.

Returns:
    bool: True if the object is not an instance of any class in the list of classes, False otherwise.

<a id="neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.train"></a>

#### train

```python
def train(X_train,
          y_train,
          X_val=None,
          y_val=None,
          optimizer=None,
          epochs=100,
          batch_size=32,
          early_stopping_threshold=10,
          lr_scheduler=None,
          p=True,
          use_tqdm=True,
          n_jobs=1,
          track_metrics=False,
          track_adv_metrics=False,
          save_animation=False,
          save_path="training_animation.mp4",
          fps=1,
          dpi=100,
          frame_every=1)
```

Trains the neural network model.

Args:
    X_train: (ndarray) - Training data features.
    y_train: (ndarray) - Training data labels.
    X_val: (ndarray) - Validation data features, optional.
    y_val: (ndarray) - Validation data labels, optional.
    optimizer: (Optimizer) - Optimizer for updating parameters (default: JITAdam, lr=0.0001).
    epochs: (int) - Number of training epochs (default: 100).
    batch_size: (int) - Batch size for mini-batch gradient descent (default: 32).
    early_stopping_threshold: (int) - Patience for early stopping (default: 10).
    lr_scheduler: (Scheduler) - Learning rate scheduler (default: None).
    p: (bool) - Whether to print training progress (default: True).
    use_tqdm: (bool) - Whether to use tqdm for progress bar (default: True).
    n_jobs: (int) - Number of jobs for parallel processing (default: 1).
    track_metrics: (bool) - Whether to track training metrics (default: False).
    track_adv_metrics: (bool) - Whether to track advanced metrics (default: False).
    save_animation: (bool) - Whether to save the animation of metrics (default: False).
    save_path: (str) - Path to save the animation file. File extension must be .mp4 or .gif (default: 'training_animation.mp4').
    fps: (int) - Frames per second for the saved animation (default: 1).
    dpi: (int) - DPI for the saved animation (default: 100).
    frame_every: (int) - Capture frame every N epochs (to reduce file size) (default: 1).

<a id="neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.evaluate"></a>

#### evaluate

```python
def evaluate(X, y)
```

Evaluates the neural network on the given data.

Args:
    X (ndarray): Input data.
    y (ndarray): Target labels.

Returns:
    tuple: Accuracy and predicted labels.

<a id="neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.predict"></a>

#### predict

```python
def predict(X)
```

Predicts the output for the given input data.

Args:
    X (ndarray): Input data.

Returns:
    ndarray: Predicted outputs.

<a id="neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(X, y)
```

Calculates the loss with L2 regularization.

Args:
    X (ndarray): Input data.
    y (ndarray): Target labels.

Returns:
    float: The calculated loss value.

<a id="neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork._create_optimizer"></a>

#### \_create\_optimizer

```python
def _create_optimizer(optimizer_type, learning_rate, JIT=False)
```

Helper method to create optimizer instances.

<a id="neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.tune_hyperparameters"></a>

#### tune\_hyperparameters

```python
def tune_hyperparameters(X_train,
                         y_train,
                         X_val,
                         y_val,
                         param_grid,
                         layer_configs=None,
                         optimizer_types=None,
                         lr_range=(0.0001, 0.01, 5),
                         epochs=30,
                         batch_size=32)
```

Performs hyperparameter tuning using grid search.

Args:
    X_train: (np.ndarray) - Training feature data.
    y_train: (np.ndarray) - Training target data.
    X_val: (np.ndarray) - Validation feature data.
    y_val: (np.ndarray) - Validation target data.
    param_grid: (dict) - Dictionary of parameters to try.
    layer_configs: (list), optional - List of layer configurations (default is None).
    optimizer_types: (list), optional - List of optimizer types (default is None).
    lr_range: (tuple), optional - (min_lr, max_lr, num_steps) for learning rates (default is (0.0001, 0.01, 5)).
    epochs: (int), optional - Max epochs for each trial (default is 30).
    batch_size: (int), optional - Batch size for training (default is 32).

Returns:
    best_params: (dict) - Best hyperparameters found.
    best_accuracy: (float) - Best validation accuracy.

<a id="neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork._get_jit_loss_calculator"></a>

#### \_get\_jit\_loss\_calculator

```python
def _get_jit_loss_calculator()
```

Helper to get the corresponding @njit loss calculation function.

<a id="neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.compile_numba_functions"></a>

#### compile\_numba\_functions

```python
def compile_numba_functions(progress_bar=True)
```

Compiles all Numba JIT functions to improve performance.

Args:
    progress_bar (bool): Whether to display a progress bar.

<a id="neural_networks.numba_utils"></a>

# neural\_networks.numba\_utils

<a id="neural_networks.numba_utils.CACHE"></a>

#### CACHE

<a id="neural_networks.numba_utils.calculate_loss_from_outputs_binary"></a>

#### calculate\_loss\_from\_outputs\_binary

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_loss_from_outputs_binary(outputs, y, weights, reg_lambda)
```

Calculate binary classification loss with L2 regularization.

<a id="neural_networks.numba_utils.calculate_loss_from_outputs_multi"></a>

#### calculate\_loss\_from\_outputs\_multi

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_loss_from_outputs_multi(outputs, y, weights, reg_lambda)
```

Calculate multi-class classification loss with L2 regularization.

<a id="neural_networks.numba_utils.calculate_cross_entropy_loss"></a>

#### calculate\_cross\_entropy\_loss

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_cross_entropy_loss(logits, targets)
```

Calculate cross-entropy loss for multi-class classification.

<a id="neural_networks.numba_utils.calculate_bce_with_logits_loss"></a>

#### calculate\_bce\_with\_logits\_loss

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_bce_with_logits_loss(logits, targets)
```

Calculate binary cross-entropy loss with logits.

<a id="neural_networks.numba_utils._compute_l2_reg"></a>

#### \_compute\_l2\_reg

```python
@njit(fastmath=True, nogil=True, parallel=True, cache=CACHE)
def _compute_l2_reg(weights)
```

Compute L2 regularization for a list of weight matrices.

<a id="neural_networks.numba_utils.evaluate_batch"></a>

#### evaluate\_batch

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def evaluate_batch(y_hat, y_true, is_binary)
```

Evaluate accuracy for a batch of predictions.

<a id="neural_networks.numba_utils.calculate_mse_loss"></a>

#### calculate\_mse\_loss

```python
@njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
def calculate_mse_loss(y_pred, y_true)
```

Helper function to calculate the mean squared error loss. Handles 1D and 2D inputs.

<a id="neural_networks.numba_utils.calculate_mae_loss"></a>

#### calculate\_mae\_loss

```python
@njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
def calculate_mae_loss(y_pred, y_true)
```

Helper function to calculate the mean absolute error loss. Handles 1D and 2D inputs.

<a id="neural_networks.numba_utils.calculate_huber_loss"></a>

#### calculate\_huber\_loss

```python
@njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
def calculate_huber_loss(y_pred, y_true, delta=1.0)
```

Helper function to calculate the Huber loss. Handles 1D and 2D inputs.

<a id="neural_networks.numba_utils.relu"></a>

#### relu

```python
@njit(fastmath=True, cache=CACHE)
def relu(z)
```

Apply ReLU activation function.

<a id="neural_networks.numba_utils.relu_derivative"></a>

#### relu\_derivative

```python
@njit(fastmath=True, cache=CACHE)
def relu_derivative(z)
```

Compute the derivative of the ReLU activation function.

<a id="neural_networks.numba_utils.leaky_relu"></a>

#### leaky\_relu

```python
@njit(fastmath=True, cache=CACHE)
def leaky_relu(z, alpha=0.01)
```

Apply Leaky ReLU activation function.

<a id="neural_networks.numba_utils.leaky_relu_derivative"></a>

#### leaky\_relu\_derivative

```python
@njit(fastmath=True, cache=CACHE)
def leaky_relu_derivative(z, alpha=0.01)
```

Compute the derivative of the Leaky ReLU activation function.

<a id="neural_networks.numba_utils.tanh"></a>

#### tanh

```python
@njit(fastmath=True, cache=CACHE)
def tanh(z)
```

Apply tanh activation function.

<a id="neural_networks.numba_utils.tanh_derivative"></a>

#### tanh\_derivative

```python
@njit(fastmath=True, cache=CACHE)
def tanh_derivative(z)
```

Compute the derivative of the tanh activation function.

<a id="neural_networks.numba_utils.sigmoid"></a>

#### sigmoid

```python
@njit(fastmath=True, cache=CACHE)
def sigmoid(z)
```

Apply sigmoid activation function.

<a id="neural_networks.numba_utils.sigmoid_derivative"></a>

#### sigmoid\_derivative

```python
@njit(fastmath=True, cache=CACHE)
def sigmoid_derivative(z)
```

Compute the derivative of the sigmoid activation function.

<a id="neural_networks.numba_utils.softmax"></a>

#### softmax

```python
@njit(parallel=True, fastmath=True, cache=CACHE)
def softmax(z)
```

Apply softmax activation function.

<a id="neural_networks.numba_utils.sum_reduce"></a>

#### sum\_reduce

```python
@njit(fastmath=True, cache=CACHE)
def sum_reduce(arr)
```

Sum elements along the last axis and reduce the array.

<a id="neural_networks.numba_utils.sum_axis0"></a>

#### sum\_axis0

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def sum_axis0(arr)
```

Sum elements along axis 0.

<a id="neural_networks.numba_utils.apply_dropout_jit"></a>

#### apply\_dropout\_jit

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def apply_dropout_jit(X, dropout_rate)
```

Apply dropout to activation values.

<a id="neural_networks.numba_utils.compute_l2_reg"></a>

#### compute\_l2\_reg

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def compute_l2_reg(weights)
```

Compute L2 regularization for weights.

<a id="neural_networks.numba_utils.one_hot_encode"></a>

#### one\_hot\_encode

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def one_hot_encode(y, num_classes)
```

One-hot encode a vector of class labels.

<a id="neural_networks.numba_utils.process_batches_binary"></a>

#### process\_batches\_binary

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def process_batches_binary(X_shuffled, y_shuffled, batch_size, layers,
                           dropout_rate, dropout_layer_indices, reg_lambda,
                           dWs_acc, dbs_acc)
```

Process batches for binary classification.

<a id="neural_networks.numba_utils.process_batches_multi"></a>

#### process\_batches\_multi

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def process_batches_multi(X_shuffled, y_shuffled, batch_size, layers,
                          dropout_rate, dropout_layer_indices, reg_lambda,
                          dWs_acc, dbs_acc)
```

Process batches for multi-class classification.

<a id="neural_networks.numba_utils.process_batches_regression_jit"></a>

#### process\_batches\_regression\_jit

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def process_batches_regression_jit(X_shuffled, y_shuffled, batch_size, layers,
                                   dropout_rate, dropout_layer_indices,
                                   reg_lambda, dWs_acc, dbs_acc,
                                   loss_calculator_func)
```

Process batches for regression tasks using Numba.

<a id="neural_networks.numba_utils.evaluate_jit"></a>

#### evaluate\_jit

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def evaluate_jit(y_hat, y_true, is_binary)
```

Evaluate model performance and return accuracy and predictions.

<a id="neural_networks.numba_utils.evaluate_regression_jit"></a>

#### evaluate\_regression\_jit

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def evaluate_regression_jit(y_pred, y_true, loss_function)
```

Evaluate model performance for regression tasks using Numba.

Args:
    y_pred (ndarray): Model predictions.
    y_true (ndarray): True target values.
    loss_function (object): The JIT loss function instance (e.g., JITMeanSquaredErrorLoss).

Returns:
    tuple: Metric value (e.g., MSE) and the predictions.

<a id="neural_networks.optimizers"></a>

# neural\_networks.optimizers

<a id="neural_networks.optimizers.AdamOptimizer"></a>

## AdamOptimizer Objects

```python
class AdamOptimizer()
```

Adam optimizer class for training neural networks.

Formula: w = w - alpha * m_hat / (sqrt(v_hat) + epsilon) - lambda * w
Derived from: https://arxiv.org/abs/1412.6980

Args:
    learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
    beta1 (float, optional): The exponential decay rate for the first moment estimates. Defaults to 0.9.
    beta2 (float, optional): The exponential decay rate for the second moment estimates. Defaults to 0.999.
    epsilon (float, optional): A small value to prevent division by zero. Defaults to 1e-8.
    reg_lambda (float, optional): The regularization parameter. Defaults to 0.01.

<a id="neural_networks.optimizers.AdamOptimizer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(learning_rate=0.001,
             beta1=0.9,
             beta2=0.999,
             epsilon=1e-8,
             reg_lambda=0.01)
```

Initializes the optimizer with the given hyperparameters.

Args:
    learning_rate (float, optional): The learning rate (alpha) for the optimizer. Defaults to 0.001.
    beta1 (float, optional): Exponential decay rate for the first moment estimates. Defaults to 0.9.
    beta2 (float, optional): Exponential decay rate for the second moment estimates. Defaults to 0.999.
    epsilon (float, optional): A small value to prevent division by zero. Defaults to 1e-8.
    reg_lambda (float, optional): Regularization parameter; higher values indicate stronger regularization. Defaults to 0.01.

Attributes:
    learning_rate (float): The learning rate for the optimizer.
    beta1 (float): Exponential decay rate for the first moment estimates.
    beta2 (float): Exponential decay rate for the second moment estimates.
    epsilon (float): A small value to prevent division by zero.
    reg_lambda (float): Regularization parameter for controlling overfitting.
    m (list): List to store first moment estimates for each parameter.
    v (list): List to store second moment estimates for each parameter.
    t (int): Time step counter for the optimizer.

<a id="neural_networks.optimizers.AdamOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the first and second moment estimates for each layer's weights.

Args:
    layers (list): List of layers in the neural network.

Returns:
    None

<a id="neural_networks.optimizers.AdamOptimizer.update"></a>

#### update

```python
def update(layer, dW, db, index)
```

Updates the weights and biases of a layer using the Adam optimization algorithm.

Args:
    layer (Layer): The layer to update.
    dW (ndarray): The gradient of the weights.
    db (ndarray): The gradient of the biases.
    index (int): The index of the layer.
Returns: None

<a id="neural_networks.optimizers.SGDOptimizer"></a>

## SGDOptimizer Objects

```python
class SGDOptimizer()
```

Stochastic Gradient Descent (SGD) optimizer class for training neural networks.

Formula: w = w - learning_rate * dW, b = b - learning_rate * db

Args:
    learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
    momentum (float, optional): The momentum factor. Defaults to 0.0.
    reg_lambda (float, optional): The regularization parameter. Defaults to 0.0.

<a id="neural_networks.optimizers.SGDOptimizer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(learning_rate=0.001, momentum=0.0, reg_lambda=0.0)
```

Initializes the optimizer with specified parameters.

Args:
    learning_rate (float, optional): The step size for updating weights. Defaults to 0.001.
    momentum (float, optional): The momentum factor to accelerate gradient descent. Defaults to 0.0.
    reg_lambda (float, optional): The regularization parameter to prevent overfitting. Defaults to 0.0.

<a id="neural_networks.optimizers.SGDOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the velocity for each layer's weights.

Args:
    layers (list): List of layers in the neural network.

Returns:
    None

<a id="neural_networks.optimizers.SGDOptimizer.update"></a>

#### update

```python
def update(layer, dW, db, index)
```

Updates the weights and biases of a layer using the SGD optimization algorithm.

Args:
    layer (Layer): The layer to update.
    dW (ndarray): The gradient of the weights.
    db (ndarray): The gradient of the biases.
    index (int): The index of the layer.

Returns:
    None

<a id="neural_networks.optimizers.AdadeltaOptimizer"></a>

## AdadeltaOptimizer Objects

```python
class AdadeltaOptimizer()
```

Adadelta optimizer class for training neural networks.

Formula:
    E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * g^2
    Delta_x = - (sqrt(E[delta_x^2]_{t-1} + epsilon) / sqrt(E[g^2]_t + epsilon)) * g
    E[delta_x^2]_t = rho * E[delta_x^2]_{t-1} + (1 - rho) * Delta_x^2
Derived from: https://arxiv.org/abs/1212.5701

Args:
    learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1.0.
    rho (float, optional): The decay rate. Defaults to 0.95.
    epsilon (float, optional): A small value to prevent division by zero. Defaults to 1e-6.
    reg_lambda (float, optional): The regularization parameter. Defaults to 0.0.

<a id="neural_networks.optimizers.AdadeltaOptimizer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(learning_rate=1.0, rho=0.95, epsilon=1e-6, reg_lambda=0.0)
```

Initializes the optimizer with the specified hyperparameters.

Args:
    learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1.0.
    rho (float, optional): The decay rate for the running averages. Defaults to 0.95.
    epsilon (float, optional): A small value to prevent division by zero. Defaults to 1e-6.
    reg_lambda (float, optional): The regularization parameter for weight decay. Defaults to 0.0.

<a id="neural_networks.optimizers.AdadeltaOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the running averages for each layer's weights.

Args:
    layers (list): List of layers in the neural network.

Returns:
    None

<a id="neural_networks.optimizers.AdadeltaOptimizer.update"></a>

#### update

```python
def update(layer, dW, db, index)
```

Updates the weights and biases of a layer using the Adadelta optimization algorithm.

Args:
    layer (Layer): The layer to update.
    dW (ndarray): The gradient of the weights.
    db (ndarray): The gradient of the biases.
    index (int): The index of the layer.

Returns:
    None

<a id="neural_networks.optimizers_cupy"></a>

# neural\_networks.optimizers\_cupy

<a id="neural_networks.optimizers_cupy.CuPyAdamOptimizer"></a>

## CuPyAdamOptimizer Objects

```python
class CuPyAdamOptimizer()
```

Adam optimizer class for training neural networks.

Formula: w = w - alpha * m_hat / (sqrt(v_hat) + epsilon) - lambda * w
Derived from: https://arxiv.org/abs/1412.6980

Args:
    learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
    beta1 (float, optional): The exponential decay rate for the first moment estimates. Defaults to 0.9.
    beta2 (float, optional): The exponential decay rate for the second moment estimates. Defaults to 0.999.
    epsilon (float, optional): A small value to prevent division by zero. Defaults to 1e-8.
    reg_lambda (float, optional): The regularization parameter. Defaults to 0.01.

<a id="neural_networks.optimizers_cupy.CuPyAdamOptimizer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(learning_rate=0.001,
             beta1=0.9,
             beta2=0.999,
             epsilon=1e-8,
             reg_lambda=0.01)
```

Initializes the optimizer with the specified hyperparameters.

Args:
    learning_rate: (float), optional - The step size for updating weights (default is 0.001).
    beta1: (float), optional - Exponential decay rate for the first moment estimates (default is 0.9).
    beta2: (float), optional - Exponential decay rate for the second moment estimates (default is 0.999).
    epsilon: (float), optional - A small constant to prevent division by zero (default is 1e-8).
    reg_lambda: (float), optional - Regularization parameter for weight decay (default is 0.01).

<a id="neural_networks.optimizers_cupy.CuPyAdamOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the optimizer's internal state for the given layers.

Args:
    layers: (list) - A list of layers, each containing weights and biases.

<a id="neural_networks.optimizers_cupy.CuPyAdamOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

Updates the weights and biases of the layers using Adam optimization.

Args:
    layers: (list) - A list of layers to update.
    dWs: (list) - Gradients of the weights for each layer.
    dbs: (list) - Gradients of the biases for each layer.

<a id="neural_networks.optimizers_cupy.CuPySGDOptimizer"></a>

## CuPySGDOptimizer Objects

```python
class CuPySGDOptimizer()
```

Stochastic Gradient Descent (SGD) optimizer class for training neural networks.

Formula: v = momentum * v - learning_rate * dW, w = w + v - learning_rate * reg_lambda * w

Args:
    learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
    momentum (float, optional): The momentum factor. Defaults to 0.0.
    reg_lambda (float, optional): The regularization parameter. Defaults to 0.0.

<a id="neural_networks.optimizers_cupy.CuPySGDOptimizer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(learning_rate=0.001, momentum=0.0, reg_lambda=0.0)
```

Initializes the optimizer with specified hyperparameters.

Args:
    learning_rate: (float), optional - The step size for updating weights (default is 0.001).
    momentum: (float), optional - The momentum factor for accelerating gradient descent (default is 0.0).
    reg_lambda: (float), optional - The regularization strength to prevent overfitting (default is 0.0).

Attributes:
    velocity: (None or np.ndarray) - The velocity term used for momentum-based updates (initialized as None).

<a id="neural_networks.optimizers_cupy.CuPySGDOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the optimizer's velocity for the given layers.

Args:
    layers: (list) - A list of layers, each containing weights and biases.

<a id="neural_networks.optimizers_cupy.CuPySGDOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

Updates the weights and biases of the layers using SGD optimization.

Args:
    layers: (list) - A list of layers to update.
    dWs: (list) - Gradients of the weights for each layer.
    dbs: (list) - Gradients of the biases for each layer.

<a id="neural_networks.optimizers_cupy.CuPyAdadeltaOptimizer"></a>

## CuPyAdadeltaOptimizer Objects

```python
class CuPyAdadeltaOptimizer()
```

Adadelta optimizer class for training neural networks.

Formula:
    E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * g^2
    Delta_x = - (sqrt(E[delta_x^2]_{t-1} + epsilon) / sqrt(E[g^2]_t + epsilon)) * g
    E[delta_x^2]_t = rho * E[delta_x^2]_{t-1} + (1 - rho) * Delta_x^2
Derived from: https://arxiv.org/abs/1212.5701

Args:
    learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1.0.
    rho (float, optional): The decay rate. Defaults to 0.95.
    epsilon (float, optional): A small value to prevent division by zero. Defaults to 1e-6.
    reg_lambda (float, optional): The regularization parameter. Defaults to 0.0.

<a id="neural_networks.optimizers_cupy.CuPyAdadeltaOptimizer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(learning_rate=1.0, rho=0.95, epsilon=1e-6, reg_lambda=0.0)
```

Initializes the optimizer with the specified hyperparameters.

Args:
    learning_rate: (float), optional - The learning rate for the optimizer (default is 1.0).
    rho: (float), optional - The decay rate for the moving average of squared gradients (default is 0.95).
    epsilon: (float), optional - A small constant to prevent division by zero (default is 1e-6).
    reg_lambda: (float), optional - The regularization parameter for weight decay (default is 0.0).

Attributes:
    E_g2: (None or np.ndarray) - The moving average of squared gradients, initialized as None.
    E_delta_x2: (None or np.ndarray) - The moving average of squared parameter updates, initialized as None.

<a id="neural_networks.optimizers_cupy.CuPyAdadeltaOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the optimizer's internal state for the given layers.

Args:
    layers: (list) - A list of layers, each containing weights and biases.

<a id="neural_networks.optimizers_cupy.CuPyAdadeltaOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

Updates the weights and biases of the layers using Adadelta optimization.

Args:
    layers: (list) - A list of layers to update.
    dWs: (list) - Gradients of the weights for each layer.
    dbs: (list) - Gradients of the biases for each layer.

<a id="neural_networks.optimizers_jit"></a>

# neural\_networks.optimizers\_jit

<a id="neural_networks.optimizers_jit.CACHE"></a>

#### CACHE

<a id="neural_networks.optimizers_jit.spec_adam"></a>

#### spec\_adam

<a id="neural_networks.optimizers_jit.JITAdamOptimizer"></a>

## JITAdamOptimizer Objects

```python
@jitclass(spec_adam)
class JITAdamOptimizer()
```

Adam optimizer class for training neural networks.

Formula: w = w - alpha * m_hat / (sqrt(v_hat) + epsilon) - lambda * w
Derived from: https://arxiv.org/abs/1412.6980
Args:
    learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
    beta1 (float, optional): The exponential decay rate for the first moment estimates. Defaults to 0.9.
    beta2 (float, optional): The exponential decay rate for the second moment estimates. Defaults to 0.999.
    epsilon (float, optional): A small value to prevent division by zero. Defaults to 1e-8.
    reg_lambda (float, optional): The regularization parameter. Defaults to 0.01.

<a id="neural_networks.optimizers_jit.JITAdamOptimizer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(learning_rate=0.001,
             beta1=0.9,
             beta2=0.999,
             epsilon=1e-8,
             reg_lambda=0.01)
```

Initializes the optimizer with the specified hyperparameters.

Args:
    learning_rate: (float), optional - The learning rate for the optimizer (default is 0.001).
    beta1: (float), optional - Exponential decay rate for the first moment estimates (default is 0.9).
    beta2: (float), optional - Exponential decay rate for the second moment estimates (default is 0.999).
    epsilon: (float), optional - A small value to prevent division by zero (default is 1e-8).
    reg_lambda: (float), optional - Regularization parameter; larger values imply stronger regularization (default is 0.01).

<a id="neural_networks.optimizers_jit.JITAdamOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the first and second moment estimates for each layer's weights.

Args:
    layers: (list) - List of layers in the neural network.

Returns:
    None

<a id="neural_networks.optimizers_jit.JITAdamOptimizer.update"></a>

#### update

```python
def update(layer, dW, db, index)
```

Updates the weights and biases of a layer using the Adam optimization algorithm.

Args:
    layer: (Layer) - The layer to update.
    dW: (np.ndarray) - The gradient of the weights.
    db: (np.ndarray) - The gradient of the biases.
    index: (int) - The index of the layer.

Returns:
    None

<a id="neural_networks.optimizers_jit.JITAdamOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

Updates all layers' weights and biases using the Adam optimization algorithm.

Args:
    layers: (list) - List of layers in the neural network.
    dWs: (list of np.ndarray) - Gradients of the weights for each layer.
    dbs: (list of np.ndarray) - Gradients of the biases for each layer.

Returns:
    None

<a id="neural_networks.optimizers_jit.adam_update_layers"></a>

#### adam\_update\_layers

```python
@njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
def adam_update_layers(m, v, t, layers, dWs, dbs, learning_rate, beta1, beta2,
                       epsilon, reg_lambda)
```

Performs parallelized Adam updates for all layers.

Args:
    m: (np.ndarray) - First moment estimates.
    v: (np.ndarray) - Second moment estimates.
    t: (int) - Current time step.
    layers: (list) - List of layers in the neural network.
    dWs: (list of np.ndarray) - Gradients of the weights for each layer.
    dbs: (list of np.ndarray) - Gradients of the biases for each layer.
    learning_rate: (float) - Learning rate for the optimizer.
    beta1: (float) - Exponential decay rate for the first moment estimates.
    beta2: (float) - Exponential decay rate for the second moment estimates.
    epsilon: (float) - Small value to prevent division by zero.
    reg_lambda: (float) - Regularization parameter.

Returns:
    None

<a id="neural_networks.optimizers_jit.spec_sgd"></a>

#### spec\_sgd

<a id="neural_networks.optimizers_jit.JITSGDOptimizer"></a>

## JITSGDOptimizer Objects

```python
@jitclass(spec_sgd)
class JITSGDOptimizer()
```

Stochastic Gradient Descent (SGD) optimizer class for training neural networks.

Formula: w = w - learning_rate * dW, b = b - learning_rate * db
Args:
    learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
    momentum (float, optional): The momentum factor. Defaults to 0.0.
    reg_lambda (float, optional): The regularization parameter. Defaults to 0.0.

<a id="neural_networks.optimizers_jit.JITSGDOptimizer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(learning_rate=0.001, momentum=0.0, reg_lambda=0.0)
```

Initializes the optimizer with specified hyperparameters.

Args:
    learning_rate: (float), optional - The learning rate for the optimizer (default is 0.001).
    momentum: (float), optional - The momentum factor for the optimizer (default is 0.0).
    reg_lambda: (float), optional - The regularization parameter (default is 0.0).

Attributes:
    learning_rate: (float) - The learning rate for the optimizer.
    momentum: (float) - The momentum factor for the optimizer.
    reg_lambda: (float) - The regularization parameter.
    velocity: (np.ndarray) - The velocity used for momentum updates, initialized to zeros.

<a id="neural_networks.optimizers_jit.JITSGDOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the velocity for each layer's weights.

Args:
    layers: (list) - List of layers in the neural network.

Returns:
    None

<a id="neural_networks.optimizers_jit.JITSGDOptimizer.update"></a>

#### update

```python
def update(layer, dW, db, index)
```

Updates the weights and biases of a layer using the SGD optimization algorithm.

Args:
    layer: (Layer) - The layer to update.
    dW: (np.ndarray) - The gradient of the weights.
    db: (np.ndarray) - The gradient of the biases.
    index: (int) - The index of the layer.

Returns:
   None

<a id="neural_networks.optimizers_jit.JITSGDOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

Updates all layers' weights and biases using the SGD optimization algorithm.

Args:
    layers: (list) - List of layers in the neural network.
    dWs: (list of np.ndarray) - Gradients of the weights for each layer.
    dbs: (list of np.ndarray) - Gradients of the biases for each layer.

Returns:
    None

<a id="neural_networks.optimizers_jit.sgd_update_layers"></a>

#### sgd\_update\_layers

```python
@njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
def sgd_update_layers(velocity, layers, dWs, dbs, learning_rate, momentum,
                      reg_lambda)
```

Performs parallelized SGD updates for all layers.

Args:
    velocity: (np.ndarray) - Velocity for momentum.
    layers: (list) - List of layers in the neural network.
    dWs: (list of np.ndarray) - Gradients of the weights for each layer.
    dbs: (list of np.ndarray) - Gradients of the biases for each layer.
    learning_rate: (float) - Learning rate for the optimizer.
    momentum: (float) - Momentum factor.
    reg_lambda: (float) - Regularization parameter.

Returns:
    None

<a id="neural_networks.optimizers_jit.spec_adadelta"></a>

#### spec\_adadelta

<a id="neural_networks.optimizers_jit.JITAdadeltaOptimizer"></a>

## JITAdadeltaOptimizer Objects

```python
@jitclass(spec_adadelta)
class JITAdadeltaOptimizer()
```

Adadelta optimizer class for training neural networks.

Formula:
    E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * g^2
    Delta_x = - (sqrt(E[delta_x^2]_{t-1} + epsilon) / sqrt(E[g^2]_t + epsilon)) * g
    E[delta_x^2]_t = rho * E[delta_x^2]_{t-1} + (1 - rho) * Delta_x^2
Derived from: https://arxiv.org/abs/1212.5701
Args:
    learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1.0.
    rho (float, optional): The decay rate. Defaults to 0.95.
    epsilon (float, optional): A small value to prevent division by zero. Defaults to 1e-6.
    reg_lambda (float, optional): The regularization parameter. Defaults to 0.0.

<a id="neural_networks.optimizers_jit.JITAdadeltaOptimizer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(learning_rate=1.0, rho=0.95, epsilon=1e-6, reg_lambda=0.0)
```

Initializes the optimizer with specified hyperparameters.

Args:
    learning_rate: (float), optional - The learning rate for the optimizer (default is 1.0).
    rho: (float), optional - The decay rate for the running averages (default is 0.95).
    epsilon: (float), optional - A small value to prevent division by zero (default is 1e-6).
    reg_lambda: (float), optional - The regularization parameter (default is 0.0).

Attributes:
    E_g2: (np.ndarray) - Running average of squared gradients.
    E_delta_x2: (np.ndarray) - Running average of squared parameter updates.

<a id="neural_networks.optimizers_jit.JITAdadeltaOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the running averages for each layer's weights.

Args:
    layers: (list) - List of layers in the neural network.

Returns:
    None

<a id="neural_networks.optimizers_jit.JITAdadeltaOptimizer.update"></a>

#### update

```python
def update(layer, dW, db, index)
```

Updates the weights and biases of a layer using the Adadelta optimization algorithm.

Args:
    layer: (Layer) - The layer to update.
    dW: (np.ndarray) - The gradient of the weights.
    db: (np.ndarray) - The gradient of the biases.
    index: (int) - The index of the layer.

Returns:
    None

<a id="neural_networks.optimizers_jit.JITAdadeltaOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

Updates all layers' weights and biases using the Adadelta optimization algorithm.

Args:
    layers: (list) - List of layers in the neural network.
    dWs: (list of np.ndarray) - Gradients of the weights for each layer.
    dbs: (list of np.ndarray) - Gradients of the biases for each layer.

Returns:
    None

<a id="neural_networks.optimizers_jit.adadelta_update_layers"></a>

#### adadelta\_update\_layers

```python
@njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
def adadelta_update_layers(E_g2, E_delta_x2, layers, dWs, dbs, learning_rate,
                           rho, epsilon, reg_lambda)
```

Performs parallelized Adadelta updates for all layers.

Args:
    E_g2: (np.ndarray) - Running average of squared gradients.
    E_delta_x2: (np.ndarray) - Running average of squared parameter updates.
    layers: (list) - List of layers in the neural network.
    dWs: (list of np.ndarray) - Gradients of the weights for each layer.
    dbs: (list of np.ndarray) - Gradients of the biases for each layer.
    learning_rate: (float) - Learning rate for the optimizer.
    rho: (float) - Decay rate.
    epsilon: (float) - Small value to prevent division by zero.
    reg_lambda: (float) - Regularization parameter.

Returns:
    None

<a id="neural_networks.schedulers"></a>

# neural\_networks.schedulers

<a id="neural_networks.schedulers.lr_scheduler_step"></a>

## lr\_scheduler\_step Objects

```python
class lr_scheduler_step()
```

Learning rate scheduler class for training neural networks.

Reduces the learning rate by a factor of lr_decay every lr_decay_epoch epochs.

Args:
    optimizer (Optimizer): The optimizer to adjust the learning rate for.
    lr_decay (float, optional): The factor to reduce the learning rate by. Defaults to 0.1.
    lr_decay_epoch (int, optional): The number of epochs to wait before decaying the learning rate. Defaults to 10

<a id="neural_networks.schedulers.lr_scheduler_step.__init__"></a>

#### \_\_init\_\_

```python
def __init__(optimizer, lr_decay=0.1, lr_decay_epoch=10)
```

Initializes the scheduler with the given optimizer and learning rate decay parameters.

Args:
    optimizer (Optimizer): The optimizer whose learning rate will be scheduled.
    lr_decay (float, optional): The factor by which the learning rate will be multiplied at each decay step. Default is 0.1.
    lr_decay_epoch (int, optional): The number of epochs after which the learning rate will be decayed. Default is 10.

<a id="neural_networks.schedulers.lr_scheduler_step.__repr__"></a>

#### \_\_repr\_\_

```python
def __repr__()
```

Returns a string representation of the scheduler.

<a id="neural_networks.schedulers.lr_scheduler_step.step"></a>

#### step

```python
def step(epoch)
```

Adjusts the learning rate based on the current epoch. Decays the learning rate by lr_decay every lr_decay_epoch epochs.

Args:
    epoch (int): The current epoch number.

Returns:
    None

<a id="neural_networks.schedulers.lr_scheduler_step.reduce"></a>

#### reduce

```python
def reduce()
```

Reduces the learning rate by the decay factor.

<a id="neural_networks.schedulers.lr_scheduler_exp"></a>

## lr\_scheduler\_exp Objects

```python
class lr_scheduler_exp()
```

Learning rate scheduler class for training neural networks.

Reduces the learning rate exponentially by lr_decay every lr_decay_epoch epochs.

<a id="neural_networks.schedulers.lr_scheduler_exp.__init__"></a>

#### \_\_init\_\_

```python
def __init__(optimizer, lr_decay=0.1, lr_decay_epoch=10)
```

Initializes the scheduler with the given optimizer and learning rate decay parameters.

Args:
    optimizer (Optimizer): The optimizer whose learning rate will be scheduled.
    lr_decay (float, optional): The factor by which the learning rate will be multiplied at each decay step. Default is 0.1.
    lr_decay_epoch (int, optional): The number of epochs after which the learning rate will be decayed. Default is 10.

<a id="neural_networks.schedulers.lr_scheduler_exp.__repr__"></a>

#### \_\_repr\_\_

```python
def __repr__()
```

Returns a string representation of the scheduler.

<a id="neural_networks.schedulers.lr_scheduler_exp.step"></a>

#### step

```python
def step(epoch)
```

Adjusts the learning rate based on the current epoch. Decays the learning rate by lr_decay every lr_decay_epoch epochs.

Args:
    epoch (int): The current epoch number.
Returns: None

<a id="neural_networks.schedulers.lr_scheduler_exp.reduce"></a>

#### reduce

```python
def reduce()
```

Reduces the learning rate exponentially.

<a id="neural_networks.schedulers.lr_scheduler_plateau"></a>

## lr\_scheduler\_plateau Objects

```python
class lr_scheduler_plateau()
```

A custom learning rate scheduler that adjusts the learning rate based on the plateau of the loss function.

Args:
    lr_scheduler (object): The learning rate scheduler object.
    patience (int): The number of epochs to wait for improvement before reducing the learning rate. Default is 5.
    threshold (float): The minimum improvement threshold required to update the best loss. Default is 0.01.

Methods:
    step(loss): Updates the learning rate based on the loss value.

<a id="neural_networks.schedulers.lr_scheduler_plateau.__init__"></a>

#### \_\_init\_\_

```python
def __init__(lr_scheduler, patience=5, threshold=0.01)
```

Initializes the scheduler with the given learning rate scheduler, patience, and threshold.

Args:
    lr_scheduler (torch.optim.lr_scheduler): The learning rate scheduler to be used.
    patience (int, optional): Number of epochs to wait for improvement before taking action. Defaults to 5.
    threshold (float, optional): Minimum change in the monitored value to qualify as an improvement. Defaults to 0.01.

<a id="neural_networks.schedulers.lr_scheduler_plateau.__repr__"></a>

#### \_\_repr\_\_

```python
def __repr__()
```

Returns a string representation of the scheduler.

<a id="neural_networks.schedulers.lr_scheduler_plateau.step"></a>

#### step

```python
def step(epoch, loss)
```

Updates the learning rate based on the loss value.

Args:
    epoch (int): The current epoch number.
    loss (float): The current loss value.

<a id="neural_networks_cupy_dev"></a>

# neural\_networks\_cupy\_dev

<a id="neural_networks_cupy_dev.__all__"></a>

#### \_\_all\_\_

<a id="neural_networks_cupy_dev.cupy_utils"></a>

# neural\_networks\_cupy\_dev.cupy\_utils

<a id="neural_networks_cupy_dev.cupy_utils.fused_dropout"></a>

#### fused\_dropout

```python
@fuse()
def fused_dropout(x, dropout_rate, random_vals)
```

<a id="neural_networks_cupy_dev.cupy_utils.apply_dropout"></a>

#### apply\_dropout

```python
def apply_dropout(X, dropout_rate)
```

<a id="neural_networks_cupy_dev.cupy_utils.fused_relu"></a>

#### fused\_relu

```python
@fuse()
def fused_relu(x)
```

<a id="neural_networks_cupy_dev.cupy_utils.fused_sigmoid"></a>

#### fused\_sigmoid

```python
@fuse()
def fused_sigmoid(x)
```

<a id="neural_networks_cupy_dev.cupy_utils.fused_leaky_relu"></a>

#### fused\_leaky\_relu

```python
@fuse()
def fused_leaky_relu(x, alpha=0.01)
```

<a id="neural_networks_cupy_dev.cupy_utils.forward_cupy"></a>

#### forward\_cupy

```python
def forward_cupy(X, weights, biases, activations, dropout_rate, training,
                 is_binary)
```

<a id="neural_networks_cupy_dev.cupy_utils.backward_cupy"></a>

#### backward\_cupy

```python
def backward_cupy(layer_outputs, y, weights, activations, reg_lambda,
                  is_binary, dWs, dbs)
```

<a id="neural_networks_cupy_dev.cupy_utils.logsumexp"></a>

#### logsumexp

```python
def logsumexp(a, axis=None, keepdims=False)
```

<a id="neural_networks_cupy_dev.cupy_utils.calculate_cross_entropy_loss"></a>

#### calculate\_cross\_entropy\_loss

```python
def calculate_cross_entropy_loss(logits, targets)
```

<a id="neural_networks_cupy_dev.cupy_utils.calculate_bce_with_logits_loss"></a>

#### calculate\_bce\_with\_logits\_loss

```python
def calculate_bce_with_logits_loss(logits, targets)
```

<a id="neural_networks_cupy_dev.cupy_utils.calculate_loss_from_outputs_binary"></a>

#### calculate\_loss\_from\_outputs\_binary

```python
def calculate_loss_from_outputs_binary(outputs, y, weights, reg_lambda)
```

<a id="neural_networks_cupy_dev.cupy_utils.calculate_loss_from_outputs_multi"></a>

#### calculate\_loss\_from\_outputs\_multi

```python
def calculate_loss_from_outputs_multi(outputs, y, weights, reg_lambda)
```

<a id="neural_networks_cupy_dev.cupy_utils.evaluate_batch"></a>

#### evaluate\_batch

```python
def evaluate_batch(y_hat, y_true, is_binary)
```

<a id="neural_networks_cupy_dev.loss"></a>

# neural\_networks\_cupy\_dev.loss

<a id="neural_networks_cupy_dev.loss.CrossEntropyLoss"></a>

## CrossEntropyLoss Objects

```python
class CrossEntropyLoss()
```

Optimized cross entropy loss implementation using cupy for multi-class classification.
Formula: -sum(y * log(p)) / m
Methods:
    __call__(self, logits, targets): Calculate the cross entropy loss.

<a id="neural_networks_cupy_dev.loss.CrossEntropyLoss.__call__"></a>

#### \_\_call\_\_

```python
def __call__(logits, targets)
```

Calculate the cross entropy loss.
Args:
    logits (cp.ndarray): The logits (predicted values) of shape (num_samples, num_classes).
    targets (cp.ndarray): The target labels of shape (num_samples, num_classes) or (num_samples,).
Returns:
    float: The cross entropy loss.

<a id="neural_networks_cupy_dev.loss.BCEWithLogitsLoss"></a>

## BCEWithLogitsLoss Objects

```python
class BCEWithLogitsLoss()
```

Optimized binary cross entropy loss with logits implementation using cupy.
Formula: -mean(y * log(sigmoid(x)) + (1 - y) * log(1 - sigmoid(x)))
Methods:
    __call__(self, logits, targets): Calculate the binary cross entropy loss.

<a id="neural_networks_cupy_dev.loss.BCEWithLogitsLoss.__call__"></a>

#### \_\_call\_\_

```python
def __call__(logits, targets)
```

Calculate the binary cross entropy loss.
Args:
    logits (cp.ndarray): The logits (predicted values) of shape (num_samples,).
    targets (cp.ndarray): The target labels of shape (num_samples,).
Returns:
    float: The binary cross entropy loss.

<a id="neural_networks_cupy_dev.neuralNetwork"></a>

# neural\_networks\_cupy\_dev.neuralNetwork

<a id="neural_networks_cupy_dev.neuralNetwork.NeuralNetwork"></a>

## NeuralNetwork Objects

```python
class NeuralNetwork()
```

Neural network class for training and evaluating a custom neural network model.
Args:
    - layer_sizes (list): A list of integers representing the sizes of each layer in the neural network.
    - dropout_rate (float): The dropout rate to be applied during training. Default is 0.2.
    - reg_lambda (float): The regularization lambda value. Default is 0.01.
    - activations (list): A list of activation functions for each layer. Default is ['relu', 'relu', ... 'softmax'].

<a id="neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.__init__"></a>

#### \_\_init\_\_

```python
def __init__(layer_sizes, dropout_rate=0.2, reg_lambda=0.01, activations=None)
```

<a id="neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.apply_dropout"></a>

#### apply\_dropout

```python
def apply_dropout(X)
```

<a id="neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.forward"></a>

#### forward

```python
def forward(X, training=True)
```

<a id="neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.backward"></a>

#### backward

```python
def backward(y)
```

<a id="neural_networks_cupy_dev.neuralNetwork.NeuralNetwork._process_batches_async"></a>

#### \_process\_batches\_async

```python
def _process_batches_async(X_shuffled, y_shuffled, batch_size, weights, biases,
                           activations, dropout_rate, is_binary, reg_lambda,
                           dWs_acc, dbs_acc)
```

<a id="neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.train"></a>

#### train

```python
def train(X_train,
          y_train,
          X_val=None,
          y_val=None,
          optimizer=None,
          epochs=100,
          batch_size=32,
          early_stopping_threshold=10,
          lr_scheduler=None,
          p=True,
          use_tqdm=True,
          n_jobs=1)
```

Trains the neural network model.
Args:
    - X_train (ndarray): Training data features.
    - y_train (ndarray): Training data labels.
    - X_val (ndarray): Validation data features, optional.
    - y_val (ndarray): Validation data labels, optional.
    - optimizer (Optimizer): Optimizer for updating parameters (default: Adam, lr=0.0001).
    - epochs (int): Number of training epochs (default: 100).
    - batch_size (int): Batch size for mini-batch gradient descent (default: 32).
    - early_stopping_patience (int): Patience for early stopping (default: 10).
    - lr_scheduler (Scheduler): Learning rate scheduler (default: None).
    - verbose (bool): Whether to print training progress (default: True).
    - use_tqdm (bool): Whether to use tqdm for progress bar (default: True).
    - n_jobs (int): Number of jobs for parallel processing (default: 1).

<a id="neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(X, y, class_weights=None)
```

<a id="neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.evaluate"></a>

#### evaluate

```python
def evaluate(X, y)
```

Evaluates the model performance.
Args:
    - X (ndarray): Input data (NumPy or CuPy array)
    - y (ndarray): Target labels (NumPy or CuPy array)
Returns:
    - accuracy (float): Model accuracy
    - predicted (ndarray): Predicted labels (NumPy array)

<a id="neural_networks_cupy_dev.neuralNetwork.NeuralNetwork._evaluate_cupy"></a>

#### \_evaluate\_cupy

```python
@staticmethod
def _evaluate_cupy(y_hat, y_true, is_binary)
```

CuPy-based function to evaluate model performance.
Args:
    y_hat (cp.ndarray): Model predictions (CuPy array).
    y_true (cp.ndarray): True labels (CuPy array).
    is_binary (bool): Whether the model is binary or multi-class.
Returns:
    tuple: Accuracy (CuPy scalar) and predicted labels (CuPy array).

<a id="neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.predict"></a>

#### predict

```python
def predict(X)
```

Generate predictions for input data.
Args:
    - X (ndarray): Input data
Returns:
    - predictions: Model predictions (class probabilities or labels)

<a id="neural_networks_cupy_dev.neuralNetwork.NeuralNetwork._create_optimizer"></a>

#### \_create\_optimizer

```python
def _create_optimizer(optimizer_type, learning_rate)
```

Helper method to create optimizer instances.

<a id="neural_networks_cupy_dev.neuralNetwork.NeuralNetwork.create_scheduler"></a>

#### create\_scheduler

```python
def create_scheduler(scheduler_type, optimizer, **kwargs)
```

Creates a learning rate scheduler.

<a id="neural_networks_cupy_dev.neuralNetwork.Layer"></a>

## Layer Objects

```python
class Layer()
```

Initializes a Layer object.
Args:
    input_size (int): The size of the input to the layer.
    output_size (int): The size of the output from the layer.
    activation (str): The activation function to be used in the layer.

<a id="neural_networks_cupy_dev.neuralNetwork.Layer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(input_size, output_size, activation="relu")
```

<a id="neural_networks_cupy_dev.neuralNetwork.Layer.zero_grad"></a>

#### zero\_grad

```python
def zero_grad()
```

Reset the gradients of the weights and biases to zero.

<a id="neural_networks_cupy_dev.neuralNetwork.Layer.activate"></a>

#### activate

```python
def activate(Z)
```

Apply activation function.

<a id="neural_networks_cupy_dev.neuralNetwork.Layer.activation_derivative"></a>

#### activation\_derivative

```python
def activation_derivative(Z)
```

Apply activation derivative.

<a id="neural_networks_cupy_dev.neuralNetwork.Activation"></a>

## Activation Objects

```python
class Activation()
```

<a id="neural_networks_cupy_dev.neuralNetwork.Activation.relu"></a>

#### relu

```python
@staticmethod
def relu(z)
```

ReLU (Rectified Linear Unit) activation function: f(z) = max(0, z)
Returns the input directly if it's positive, otherwise returns 0.

<a id="neural_networks_cupy_dev.neuralNetwork.Activation.relu_derivative"></a>

#### relu\_derivative

```python
@staticmethod
def relu_derivative(z)
```

Derivative of the ReLU function: f'(z) = 1 if z > 0, else 0
Returns 1 for positive input, and 0 for negative input.

<a id="neural_networks_cupy_dev.neuralNetwork.Activation.leaky_relu"></a>

#### leaky\_relu

```python
@staticmethod
def leaky_relu(z, alpha=0.01)
```

Leaky ReLU activation function: f(z) = z if z > 0, else alpha * z
Allows a small, non-zero gradient when the input is negative to address the dying ReLU problem.

<a id="neural_networks_cupy_dev.neuralNetwork.Activation.leaky_relu_derivative"></a>

#### leaky\_relu\_derivative

```python
@staticmethod
def leaky_relu_derivative(z, alpha=0.01)
```

Derivative of the Leaky ReLU function: f'(z) = 1 if z > 0, else alpha
Returns 1 for positive input, and alpha for negative input.

<a id="neural_networks_cupy_dev.neuralNetwork.Activation.tanh"></a>

#### tanh

```python
@staticmethod
def tanh(z)
```

Hyperbolic tangent (tanh) activation function: f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
Maps input to the range [-1, 1], typically used for normalized input.

<a id="neural_networks_cupy_dev.neuralNetwork.Activation.tanh_derivative"></a>

#### tanh\_derivative

```python
@staticmethod
def tanh_derivative(z)
```

Derivative of the tanh function: f'(z) = 1 - tanh(z)^2
Used for backpropagation through the tanh activation.

<a id="neural_networks_cupy_dev.neuralNetwork.Activation.sigmoid"></a>

#### sigmoid

```python
@staticmethod
def sigmoid(z)
```

Sigmoid activation function: f(z) = 1 / (1 + exp(-z))
Maps input to the range [0, 1], commonly used for binary classification.

<a id="neural_networks_cupy_dev.neuralNetwork.Activation.sigmoid_derivative"></a>

#### sigmoid\_derivative

```python
@staticmethod
def sigmoid_derivative(z)
```

Derivative of the sigmoid function: f'(z) = sigmoid(z) * (1 - sigmoid(z))
Used for backpropagation through the sigmoid activation.

<a id="neural_networks_cupy_dev.neuralNetwork.Activation.softmax"></a>

#### softmax

```python
@staticmethod
def softmax(z)
```

Softmax activation function: f(z)_i = exp(z_i) / sum(exp(z_j)) for all j
Maps input into a probability distribution over multiple classes. Used for multiclass classification.

<a id="neural_networks_cupy_dev.optimizers"></a>

# neural\_networks\_cupy\_dev.optimizers

<a id="neural_networks_cupy_dev.optimizers.AdamOptimizer"></a>

## AdamOptimizer Objects

```python
class AdamOptimizer()
```

Adam optimizer class for training neural networks.
Formula: w = w - alpha * m_hat / (sqrt(v_hat) + epsilon) - lambda * w
Derived from: https://arxiv.org/abs/1412.6980
Args:
    learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
    beta1 (float, optional): The exponential decay rate for the first moment estimates. Defaults to 0.9.
    beta2 (float, optional): The exponential decay rate for the second moment estimates. Defaults to 0.999.
    epsilon (float, optional): A small value to prevent division by zero. Defaults to 1e-8.
    reg_lambda (float, optional): The regularization parameter. Defaults to 0.01.

<a id="neural_networks_cupy_dev.optimizers.AdamOptimizer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(learning_rate=0.001,
             beta1=0.9,
             beta2=0.999,
             epsilon=1e-8,
             reg_lambda=0.01)
```

<a id="neural_networks_cupy_dev.optimizers.AdamOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

<a id="neural_networks_cupy_dev.optimizers.AdamOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

<a id="neural_networks_cupy_dev.optimizers.SGDOptimizer"></a>

## SGDOptimizer Objects

```python
class SGDOptimizer()
```

Stochastic Gradient Descent (SGD) optimizer class for training neural networks.
Formula: v = momentum * v - learning_rate * dW, w = w + v - learning_rate * reg_lambda * w
Args:
    learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
    momentum (float, optional): The momentum factor. Defaults to 0.0.
    reg_lambda (float, optional): The regularization parameter. Defaults to 0.0.

<a id="neural_networks_cupy_dev.optimizers.SGDOptimizer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(learning_rate=0.001, momentum=0.0, reg_lambda=0.0)
```

<a id="neural_networks_cupy_dev.optimizers.SGDOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

<a id="neural_networks_cupy_dev.optimizers.SGDOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

<a id="neural_networks_cupy_dev.optimizers.AdadeltaOptimizer"></a>

## AdadeltaOptimizer Objects

```python
class AdadeltaOptimizer()
```

Adadelta optimizer class for training neural networks.
Formula:
    E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * g^2
    Delta_x = - (sqrt(E[delta_x^2]_{t-1} + epsilon) / sqrt(E[g^2]_t + epsilon)) * g
    E[delta_x^2]_t = rho * E[delta_x^2]_{t-1} + (1 - rho) * Delta_x^2
Derived from: https://arxiv.org/abs/1212.5701
Args:
    learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1.0.
    rho (float, optional): The decay rate. Defaults to 0.95.
    epsilon (float, optional): A small value to prevent division by zero. Defaults to 1e-6.
    reg_lambda (float, optional): The regularization parameter. Defaults to 0.0.

<a id="neural_networks_cupy_dev.optimizers.AdadeltaOptimizer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(learning_rate=1.0, rho=0.95, epsilon=1e-6, reg_lambda=0.0)
```

<a id="neural_networks_cupy_dev.optimizers.AdadeltaOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

<a id="neural_networks_cupy_dev.optimizers.AdadeltaOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

<a id="neural_networks_cupy_dev.schedulers"></a>

# neural\_networks\_cupy\_dev.schedulers

<a id="neural_networks_cupy_dev.schedulers.lr_scheduler_step"></a>

## lr\_scheduler\_step Objects

```python
class lr_scheduler_step()
```

Learning rate scheduler class for training neural networks.
Reduces the learning rate by a factor of lr_decay every lr_decay_epoch epochs.
Args:
    optimizer (Optimizer): The optimizer to adjust the learning rate for.
    lr_decay (float, optional): The factor to reduce the learning rate by. Defaults to 0.1.
    lr_decay_epoch (int, optional): The number of epochs to wait before decaying the learning rate. Defaults to 10

<a id="neural_networks_cupy_dev.schedulers.lr_scheduler_step.__init__"></a>

#### \_\_init\_\_

```python
def __init__(optimizer, lr_decay=0.1, lr_decay_epoch=10)
```

<a id="neural_networks_cupy_dev.schedulers.lr_scheduler_step.__repr__"></a>

#### \_\_repr\_\_

```python
def __repr__()
```

<a id="neural_networks_cupy_dev.schedulers.lr_scheduler_step.step"></a>

#### step

```python
def step(epoch)
```

Adjusts the learning rate based on the current epoch. Decays the learning rate by lr_decay every lr_decay_epoch epochs.
Args:
    epoch (int): The current epoch number.
Returns: None

<a id="neural_networks_cupy_dev.schedulers.lr_scheduler_step.reduce"></a>

#### reduce

```python
def reduce()
```

<a id="neural_networks_cupy_dev.schedulers.lr_scheduler_exp"></a>

## lr\_scheduler\_exp Objects

```python
class lr_scheduler_exp()
```

Learning rate scheduler class for training neural networks.
Reduces the learning rate exponentially by lr_decay every lr_decay_epoch epochs.

<a id="neural_networks_cupy_dev.schedulers.lr_scheduler_exp.__init__"></a>

#### \_\_init\_\_

```python
def __init__(optimizer, lr_decay=0.1, lr_decay_epoch=10)
```

<a id="neural_networks_cupy_dev.schedulers.lr_scheduler_exp.__repr__"></a>

#### \_\_repr\_\_

```python
def __repr__()
```

<a id="neural_networks_cupy_dev.schedulers.lr_scheduler_exp.step"></a>

#### step

```python
def step(epoch)
```

Adjusts the learning rate based on the current epoch. Decays the learning rate by lr_decay every lr_decay_epoch epochs.
Args:
    epoch (int): The current epoch number.
Returns: None

<a id="neural_networks_cupy_dev.schedulers.lr_scheduler_exp.reduce"></a>

#### reduce

```python
def reduce()
```

<a id="neural_networks_cupy_dev.schedulers.lr_scheduler_plateau"></a>

## lr\_scheduler\_plateau Objects

```python
class lr_scheduler_plateau()
```

A custom learning rate scheduler that adjusts the learning rate based on the plateau of the loss function.
Args:
    lr_scheduler (object): The learning rate scheduler object.
    patience (int): The number of epochs to wait for improvement before reducing the learning rate. Default is 5.
    threshold (float): The minimum improvement threshold required to update the best loss. Default is 0.01.
Methods:
    step(loss): Updates the learning rate based on the loss value.

<a id="neural_networks_cupy_dev.schedulers.lr_scheduler_plateau.__init__"></a>

#### \_\_init\_\_

```python
def __init__(lr_scheduler, patience=5, threshold=0.01)
```

<a id="neural_networks_cupy_dev.schedulers.lr_scheduler_plateau.__repr__"></a>

#### \_\_repr\_\_

```python
def __repr__()
```

<a id="neural_networks_cupy_dev.schedulers.lr_scheduler_plateau.step"></a>

#### step

```python
def step(epoch, loss)
```

Updates the learning rate based on the loss value.
Args:
    loss (float): The current loss value.

<a id="neural_networks_numba_dev"></a>

# neural\_networks\_numba\_dev

<a id="neural_networks_numba_dev.__all__"></a>

#### \_\_all\_\_

<a id="neural_networks_numba_dev.layers_jit_unified"></a>

# neural\_networks\_numba\_dev.layers\_jit\_unified

<a id="neural_networks_numba_dev.layers_jit_unified.layer_spec"></a>

#### layer\_spec

<a id="neural_networks_numba_dev.layers_jit_unified.JITLayer"></a>

## JITLayer Objects

```python
@jitclass(layer_spec)
class JITLayer()
```

<a id="neural_networks_numba_dev.layers_jit_unified.JITLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(layer_type,
             input_size,
             output_size,
             activation="relu",
             kernel_size=0,
             stride=1,
             padding=0)
```

<a id="neural_networks_numba_dev.layers_jit_unified.JITLayer.zero_grad"></a>

#### zero\_grad

```python
def zero_grad()
```

Zero out gradients for the layer.

<a id="neural_networks_numba_dev.layers_jit_unified.JITLayer.forward"></a>

#### forward

```python
def forward(X)
```

<a id="neural_networks_numba_dev.layers_jit_unified.JITLayer.backward"></a>

#### backward

```python
def backward(dA, reg_lambda)
```

<a id="neural_networks_numba_dev.layers_jit_unified.JITLayer._forward_dense"></a>

#### \_forward\_dense

```python
def _forward_dense(X)
```

Forward pass for dense layer.

<a id="neural_networks_numba_dev.layers_jit_unified.JITLayer._backward_dense"></a>

#### \_backward\_dense

```python
def _backward_dense(dA, reg_lambda)
```

Backward pass for dense layer.

<a id="neural_networks_numba_dev.layers_jit_unified.JITLayer._forward_conv"></a>

#### \_forward\_conv

```python
def _forward_conv(X)
```

Forward pass for convolutional layer.

Args:
    X: numpy array with shape (batch_size, in_channels, height, width)
        in_channels = input channels
        height = input height
        width = input width
Returns:
    Output feature maps after convolution and activation.

<a id="neural_networks_numba_dev.layers_jit_unified.JITLayer._backward_conv"></a>

#### \_backward\_conv

```python
def _backward_conv(d_out, reg_lambda=0.0)
```

Backward pass for convolutional layer.

Args:
    d_out (np.ndarray): Gradient of the loss with respect to the layer output
    reg_lambda (float, optional): Regularization parameter

Returns:
    dX: Gradient with respect to the input X

<a id="neural_networks_numba_dev.layers_jit_unified.JITLayer._forward_flatten"></a>

#### \_forward\_flatten

```python
def _forward_flatten(X)
```

Flattens the input tensor.

Args:
    X (np.ndarray): Input data of shape (batch_size, channels, height, width)

Returns:
    np.ndarray: Flattened output of shape (batch_size, flattened_size)

<a id="neural_networks_numba_dev.layers_jit_unified.JITLayer._backward_flatten"></a>

#### \_backward\_flatten

```python
def _backward_flatten(dA)
```

Reshapes the gradient back to the original input shape.

Args:
    dA (np.ndarray): Gradient of the loss with respect to the layer's output,
                   shape (batch_size, flattened_size)
    reg_lambda (float): Regularization parameter (unused in FlattenLayer).

Returns:
    np.ndarray: Gradient with respect to the input, reshaped to original input shape.

<a id="neural_networks_numba_dev.layers_jit_unified.JITLayer.activate"></a>

#### activate

```python
def activate(Z)
```

Apply activation function.

<a id="neural_networks_numba_dev.layers_jit_unified.JITLayer.activation_derivative"></a>

#### activation\_derivative

```python
def activation_derivative(Z)
```

Apply activation derivative.

<a id="neural_networks_numba_dev.layers_jit_unified.JITLayer._im2col"></a>

#### \_im2col

```python
def _im2col(x, h_out, w_out)
```

Convert image regions to columns for efficient convolution.
Fixed to avoid reshape contiguity issues.

<a id="neural_networks_numba_dev.layers_jit_unified.JITLayer._col2im"></a>

#### \_col2im

```python
def _col2im(dcol, x_shape)
```

Convert column back to image format for the backward pass.
Fixed to avoid reshape contiguity issues.

<a id="neural_networks_numba_dev.layer_jit_utils"></a>

# neural\_networks\_numba\_dev.layer\_jit\_utils

<a id="neural_networks_numba_dev.layer_jit_utils.CACHE"></a>

#### CACHE

<a id="neural_networks_numba_dev.layer_jit_utils.forward_dense"></a>

#### forward\_dense

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def forward_dense(X, weights, biases, activation_func)
```

Forward pass for dense layer.

<a id="neural_networks_numba_dev.layer_jit_utils.backward_dense"></a>

#### backward\_dense

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def backward_dense(dA, reg_lambda, weights, input_cache, output_cache,
                   activation_func)
```

Backward pass for dense layer.

<a id="neural_networks_numba_dev.layer_jit_utils.activate"></a>

#### activate

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def activate(Z, activation=None)
```

Apply activation function.

<a id="neural_networks_numba_dev.layer_jit_utils.activation_derivative"></a>

#### activation\_derivative

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def activation_derivative(Z, activation=None)
```

Apply activation derivative.

<a id="neural_networks_numba_dev.loss_jit"></a>

# neural\_networks\_numba\_dev.loss\_jit

<a id="neural_networks_numba_dev.loss_jit.CACHE"></a>

#### CACHE

<a id="neural_networks_numba_dev.loss_jit.JITCrossEntropyLoss"></a>

## JITCrossEntropyLoss Objects

```python
class JITCrossEntropyLoss()
```

<a id="neural_networks_numba_dev.loss_jit.JITCrossEntropyLoss.__init__"></a>

#### \_\_init\_\_

```python
def __init__()
```

<a id="neural_networks_numba_dev.loss_jit.JITCrossEntropyLoss.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(logits, targets)
```

<a id="neural_networks_numba_dev.loss_jit.calculate_cross_entropy_loss"></a>

#### calculate\_cross\_entropy\_loss

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_cross_entropy_loss(logits, targets)
```

<a id="neural_networks_numba_dev.loss_jit.JITBCEWithLogitsLoss"></a>

## JITBCEWithLogitsLoss Objects

```python
class JITBCEWithLogitsLoss()
```

<a id="neural_networks_numba_dev.loss_jit.JITBCEWithLogitsLoss.__init__"></a>

#### \_\_init\_\_

```python
def __init__()
```

<a id="neural_networks_numba_dev.loss_jit.JITBCEWithLogitsLoss.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(logits, targets)
```

<a id="neural_networks_numba_dev.loss_jit.calculate_bce_with_logits_loss"></a>

#### calculate\_bce\_with\_logits\_loss

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_bce_with_logits_loss(logits, targets)
```

<a id="neural_networks_numba_dev.neuralNetworkBase"></a>

# neural\_networks\_numba\_dev.neuralNetworkBase

<a id="neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase"></a>

## NeuralNetworkBase Objects

```python
class NeuralNetworkBase()
```

<a id="neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.__init__"></a>

#### \_\_init\_\_

```python
def __init__(layers, dropout_rate=0.0, reg_lambda=0.0, activations=None)
```

<a id="neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.initialize_layers"></a>

#### initialize\_layers

```python
def initialize_layers()
```

<a id="neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.forward"></a>

#### forward

```python
def forward(X, training=True)
```

<a id="neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.backward"></a>

#### backward

```python
def backward(y)
```

<a id="neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.train"></a>

#### train

```python
def train(X_train,
          y_train,
          X_val=None,
          y_val=None,
          optimizer=None,
          epochs=100,
          batch_size=32,
          early_stopping_threshold=10,
          lr_scheduler=None,
          p=True,
          use_tqdm=True,
          n_jobs=1,
          track_metrics=False,
          track_adv_metrics=False)
```

<a id="neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.evaluate"></a>

#### evaluate

```python
def evaluate(X, y)
```

<a id="neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.predict"></a>

#### predict

```python
def predict(X)
```

<a id="neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(X, y)
```

<a id="neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.apply_dropout"></a>

#### apply\_dropout

```python
def apply_dropout(X)
```

Applies dropout to the activation X.
Args:
    X (ndarray): Activation values.
Returns:
    ndarray: Activation values after applying dropout.

<a id="neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.compute_l2_reg"></a>

#### compute\_l2\_reg

```python
def compute_l2_reg(weights)
```

Computes the L2 regularization term.
Args:
    weights (list): List of weight matrices.
Returns:
    float: L2 regularization term.

<a id="neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.calculate_precision_recall_f1"></a>

#### calculate\_precision\_recall\_f1

```python
def calculate_precision_recall_f1(X, y)
```

Calculates precision, recall, and F1 score.
Args:
    - X (ndarray): Input data
    - y (ndarray): Target labels
Returns:
    - precision (float): Precision score
    - recall (float): Recall score
    - f1 (float): F1 score

<a id="neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.create_scheduler"></a>

#### create\_scheduler

```python
def create_scheduler(scheduler_type, optimizer, **kwargs)
```

Creates a learning rate scheduler.

<a id="neural_networks_numba_dev.neuralNetworkBase.NeuralNetworkBase.plot_metrics"></a>

#### plot\_metrics

```python
def plot_metrics(save_dir=None)
```

Plots the training and validation metrics.

<a id="neural_networks_numba_dev.neuralNetworkNumbaBackend"></a>

# neural\_networks\_numba\_dev.neuralNetworkNumbaBackend

<a id="neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork"></a>

## NumbaBackendNeuralNetwork Objects

```python
class NumbaBackendNeuralNetwork(NeuralNetworkBase)
```

<a id="neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.__init__"></a>

#### \_\_init\_\_

```python
def __init__(layers,
             dropout_rate=0.2,
             reg_lambda=0.01,
             activations=None,
             compile_numba=True,
             progress_bar=True)
```

Initializes the Numba backend neural network.
Args:
    layers (list): List of layer sizes or Layer objects.
    dropout_rate (float): Dropout rate for regularization.
    reg_lambda (float): L2 regularization parameter.
    activations (list): List of activation functions for each layer.
    compile_numba (bool): Whether to compile Numba functions.
    progress_bar (bool): Whether to display a progress bar.

<a id="neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.store_init_layers"></a>

#### store\_init\_layers

```python
def store_init_layers()
```

Stores the layers to restore after initialization.

<a id="neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.restore_layers"></a>

#### restore\_layers

```python
def restore_layers()
```

Restores the layers after initialization.

<a id="neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.initialize_new_layers"></a>

#### initialize\_new\_layers

```python
def initialize_new_layers()
```

Initializes the layers of the neural network.
Each layer is created with the specified number of neurons and activation function.

<a id="neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.forward"></a>

#### forward

```python
def forward(X, training=True)
```

Performs forward propagation through the neural network.
Args:
    X (ndarray): Input data of shape (batch_size, input_size).
    training (bool): Whether the network is in training mode (applies dropout).
Returns:
    ndarray: Output predictions of shape (batch_size, output_size).

<a id="neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.backward"></a>

#### backward

```python
def backward(y)
```

Performs backward propagation to calculate the gradients.
Args:
    y (ndarray): Target labels of shape (m, output_size).

<a id="neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.is_not_instance_of_classes"></a>

#### is\_not\_instance\_of\_classes

```python
@staticmethod
def is_not_instance_of_classes(obj, classes)
```

Checks if an object is not an instance of any class in a list of classes.
Args:
    obj: The object to check.
    classes: A list of classes.
Returns:
    bool: True if the object is not an instance of any class in the list of classes, False otherwise.

<a id="neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.train"></a>

#### train

```python
def train(X_train,
          y_train,
          X_val=None,
          y_val=None,
          optimizer=None,
          epochs=100,
          batch_size=32,
          early_stopping_threshold=10,
          lr_scheduler=None,
          p=True,
          use_tqdm=True,
          n_jobs=1,
          track_metrics=False,
          track_adv_metrics=False)
```

Trains the neural network model.
Args:
    X_train (ndarray): Training data features.
    y_train (ndarray): Training data labels.
    X_val (ndarray): Validation data features, optional.
    y_val (ndarray): Validation data labels, optional.
    optimizer (Optimizer): Optimizer for updating parameters (default: JITAdam, lr=0.0001).
    epochs (int): Number of training epochs (default: 100).
    batch_size (int): Batch size for mini-batch gradient descent (default: 32).
    early_stopping_threshold (int): Patience for early stopping (default: 10).
    lr_scheduler (Scheduler): Learning rate scheduler (default: None).
    p (bool): Whether to print training progress (default: True).
    use_tqdm (bool): Whether to use tqdm for progress bar (default: True).
    n_jobs (int): Number of jobs for parallel processing (default: 1).
    track_metrics (bool): Whether to track training metrics (default: False).
    track_adv_metrics (bool): Whether to track advanced metrics (default: False).

<a id="neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.evaluate"></a>

#### evaluate

```python
def evaluate(X, y)
```

Evaluates the neural network on the given data.
Args:
    X (ndarray): Input data.
    y (ndarray): Target labels.
Returns:
    tuple: Accuracy and predicted labels.

<a id="neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.predict"></a>

#### predict

```python
def predict(X)
```

Predicts the output for the given input data.
Args:
    X (ndarray): Input data.
Returns:
    ndarray: Predicted outputs.

<a id="neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(X, y)
```

Calculates the loss with L2 regularization.
Args:
    X (ndarray): Input data.
    y (ndarray): Target labels.
Returns:
    float: The calculated loss value.

<a id="neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork._create_optimizer"></a>

#### \_create\_optimizer

```python
def _create_optimizer(optimizer_type, learning_rate, JIT=False)
```

Helper method to create optimizer instances.

<a id="neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.tune_hyperparameters"></a>

#### tune\_hyperparameters

```python
def tune_hyperparameters(X_train,
                         y_train,
                         X_val,
                         y_val,
                         param_grid,
                         layer_configs=None,
                         optimizer_types=None,
                         lr_range=(0.0001, 0.01, 5),
                         epochs=30,
                         batch_size=32)
```

Performs hyperparameter tuning using grid search.
Args:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data
    - param_grid: Dict of parameters to try
    - layer_configs: List of layer configurations
    - optimizer_types: List of optimizer types
    - lr_range: (min_lr, max_lr, num_steps) for learning rates
    - epochs: Max epochs for each trial
    - batch_size: Batch size for training
Returns:
    - best_params: Best hyperparameters found
    - best_accuracy: Best validation accuracy

<a id="neural_networks_numba_dev.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.compile_numba_functions"></a>

#### compile\_numba\_functions

```python
def compile_numba_functions(progress_bar=True)
```

Compiles all Numba JIT functions to improve performance.
Args:
    progress_bar (bool): Whether to display a progress bar.

<a id="neural_networks_numba_dev.numba_utils"></a>

# neural\_networks\_numba\_dev.numba\_utils

<a id="neural_networks_numba_dev.numba_utils.CACHE"></a>

#### CACHE

<a id="neural_networks_numba_dev.numba_utils.calculate_loss_from_outputs_binary"></a>

#### calculate\_loss\_from\_outputs\_binary

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_loss_from_outputs_binary(outputs, y, weights, reg_lambda)
```

<a id="neural_networks_numba_dev.numba_utils.calculate_loss_from_outputs_multi"></a>

#### calculate\_loss\_from\_outputs\_multi

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_loss_from_outputs_multi(outputs, y, weights, reg_lambda)
```

<a id="neural_networks_numba_dev.numba_utils.calculate_cross_entropy_loss"></a>

#### calculate\_cross\_entropy\_loss

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_cross_entropy_loss(logits, targets)
```

<a id="neural_networks_numba_dev.numba_utils.calculate_bce_with_logits_loss"></a>

#### calculate\_bce\_with\_logits\_loss

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_bce_with_logits_loss(logits, targets)
```

<a id="neural_networks_numba_dev.numba_utils._compute_l2_reg"></a>

#### \_compute\_l2\_reg

```python
@njit(fastmath=True, nogil=True, parallel=True, cache=CACHE)
def _compute_l2_reg(weights)
```

<a id="neural_networks_numba_dev.numba_utils.evaluate_batch"></a>

#### evaluate\_batch

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def evaluate_batch(y_hat, y_true, is_binary)
```

<a id="neural_networks_numba_dev.numba_utils.relu"></a>

#### relu

```python
@njit(fastmath=True, cache=CACHE)
def relu(z)
```

<a id="neural_networks_numba_dev.numba_utils.relu_derivative"></a>

#### relu\_derivative

```python
@njit(fastmath=True, cache=CACHE)
def relu_derivative(z)
```

<a id="neural_networks_numba_dev.numba_utils.leaky_relu"></a>

#### leaky\_relu

```python
@njit(fastmath=True, cache=CACHE)
def leaky_relu(z, alpha=0.01)
```

<a id="neural_networks_numba_dev.numba_utils.leaky_relu_derivative"></a>

#### leaky\_relu\_derivative

```python
@njit(fastmath=True, cache=CACHE)
def leaky_relu_derivative(z, alpha=0.01)
```

<a id="neural_networks_numba_dev.numba_utils.tanh"></a>

#### tanh

```python
@njit(fastmath=True, cache=CACHE)
def tanh(z)
```

<a id="neural_networks_numba_dev.numba_utils.tanh_derivative"></a>

#### tanh\_derivative

```python
@njit(fastmath=True, cache=CACHE)
def tanh_derivative(z)
```

<a id="neural_networks_numba_dev.numba_utils.sigmoid"></a>

#### sigmoid

```python
@njit(fastmath=True, cache=CACHE)
def sigmoid(z)
```

<a id="neural_networks_numba_dev.numba_utils.sigmoid_derivative"></a>

#### sigmoid\_derivative

```python
@njit(fastmath=True, cache=CACHE)
def sigmoid_derivative(z)
```

<a id="neural_networks_numba_dev.numba_utils.softmax"></a>

#### softmax

```python
@njit(parallel=True, fastmath=True, cache=CACHE)
def softmax(z)
```

<a id="neural_networks_numba_dev.numba_utils.sum_reduce"></a>

#### sum\_reduce

```python
@njit(fastmath=True, cache=CACHE)
def sum_reduce(arr)
```

<a id="neural_networks_numba_dev.numba_utils.sum_axis0"></a>

#### sum\_axis0

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def sum_axis0(arr)
```

<a id="neural_networks_numba_dev.numba_utils.apply_dropout_jit"></a>

#### apply\_dropout\_jit

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def apply_dropout_jit(X, dropout_rate)
```

Numba JIT-compiled function to apply dropout.
Args:
    X (ndarray): Activation values.
    dropout_rate (float): Dropout rate.
Returns:
    ndarray: Activation values after applying dropout.

<a id="neural_networks_numba_dev.numba_utils.compute_l2_reg"></a>

#### compute\_l2\_reg

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def compute_l2_reg(weights)
```

<a id="neural_networks_numba_dev.numba_utils.one_hot_encode"></a>

#### one\_hot\_encode

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def one_hot_encode(y, num_classes)
```

<a id="neural_networks_numba_dev.numba_utils.process_batches_binary"></a>

#### process\_batches\_binary

```python
def process_batches_binary(X_shuffled, y_shuffled, batch_size, layers,
                           dropout_rate, dropout_layer_indices, reg_lambda,
                           dWs_acc, dbs_acc)
```

<a id="neural_networks_numba_dev.numba_utils.process_batches_multi"></a>

#### process\_batches\_multi

```python
def process_batches_multi(X_shuffled, y_shuffled, batch_size, layers,
                          dropout_rate, dropout_layer_indices, reg_lambda,
                          dWs_acc, dbs_acc)
```

<a id="neural_networks_numba_dev.numba_utils.evaluate_jit"></a>

#### evaluate\_jit

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def evaluate_jit(y_hat, y_true, is_binary)
```

Numba JIT-compiled function to evaluate model performance.
Args:
    y_hat (ndarray): Model predictions.
    is_binary (bool): Whether the model is binary or multi-class.
Returns:
    tuple: Accuracy and predicted labels.

<a id="neural_networks_numba_dev.optimizers_jit"></a>

# neural\_networks\_numba\_dev.optimizers\_jit

<a id="neural_networks_numba_dev.optimizers_jit.CACHE"></a>

#### CACHE

<a id="neural_networks_numba_dev.optimizers_jit.spec_adam"></a>

#### spec\_adam

<a id="neural_networks_numba_dev.optimizers_jit.JITAdamOptimizer"></a>

## JITAdamOptimizer Objects

```python
@jitclass(spec_adam)
class JITAdamOptimizer()
```

Adam optimizer class for training neural networks.
Formula: w = w - alpha * m_hat / (sqrt(v_hat) + epsilon) - lambda * w
Derived from: https://arxiv.org/abs/1412.6980
Args:
    learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
    beta1 (float, optional): The exponential decay rate for the first moment estimates. Defaults to 0.9.
    beta2 (float, optional): The exponential decay rate for the second moment estimates. Defaults to 0.999.
    epsilon (float, optional): A small value to prevent division by zero. Defaults to 1e-8.
    reg_lambda (float, optional): The regularization parameter. Defaults to 0.01.

<a id="neural_networks_numba_dev.optimizers_jit.JITAdamOptimizer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(learning_rate=0.001,
             beta1=0.9,
             beta2=0.999,
             epsilon=1e-8,
             reg_lambda=0.01)
```

<a id="neural_networks_numba_dev.optimizers_jit.JITAdamOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the first and second moment estimates for each layer's weights.
Args:
    layers (list): List of layers in the neural network.
    layer_type (str): Type of layers ('dense' or 'conv').
Returns: None

<a id="neural_networks_numba_dev.optimizers_jit.JITAdamOptimizer.update"></a>

#### update

```python
def update(layer, dW, db, index, layer_type)
```

<a id="neural_networks_numba_dev.optimizers_jit.JITAdamOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

<a id="neural_networks_numba_dev.optimizers_jit.dense_adam_update_layers"></a>

#### dense\_adam\_update\_layers

```python
@njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
def dense_adam_update_layers(m, v, t, indices, weights, biases, dWs, dbs,
                             learning_rate, beta1, beta2, epsilon, reg_lambda)
```

<a id="neural_networks_numba_dev.optimizers_jit.conv_adam_update_layers"></a>

#### conv\_adam\_update\_layers

```python
@njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
def conv_adam_update_layers(m, v, t, indices, weights, biases, dWs, dbs,
                            learning_rate, beta1, beta2, epsilon, reg_lambda)
```

<a id="neural_networks_numba_dev.optimizers_jit.spec_sgd"></a>

#### spec\_sgd

<a id="neural_networks_numba_dev.optimizers_jit.JITSGDOptimizer"></a>

## JITSGDOptimizer Objects

```python
@jitclass(spec_sgd)
class JITSGDOptimizer()
```

Stochastic Gradient Descent (SGD) optimizer class for training neural networks.
Formula: w = w - learning_rate * dW, b = b - learning_rate * db
Args:
    learning_rate (float, optional): The learning rate for the optimizer. Defaults to 0.001.
    momentum (float, optional): The momentum factor. Defaults to 0.0.
    reg_lambda (float, optional): The regularization parameter. Defaults to 0.0.

<a id="neural_networks_numba_dev.optimizers_jit.JITSGDOptimizer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(learning_rate=0.001, momentum=0.0, reg_lambda=0.0)
```

<a id="neural_networks_numba_dev.optimizers_jit.JITSGDOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the velocity for each layer's weights.
Args: layers (list): List of layers in the neural network.
Returns: None

<a id="neural_networks_numba_dev.optimizers_jit.JITSGDOptimizer.update"></a>

#### update

```python
def update(layer, dW, db, index, layer_type)
```

Updates the weights and biases of a layer using the SGD optimization algorithm.
Args:
    layer (Layer): The layer to update.
    dW (ndarray): The gradient of the weights.
    db (ndarray): The gradient of the biases.
    index (int): The index of the layer.
    layer_type (str): Type of layers ('dense' or 'conv').
Returns: None

<a id="neural_networks_numba_dev.optimizers_jit.JITSGDOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

<a id="neural_networks_numba_dev.optimizers_jit.dense_sgd_update_layers"></a>

#### dense\_sgd\_update\_layers

```python
@njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
def dense_sgd_update_layers(velocity, indices, weights, biases, dWs, dbs,
                            learning_rate, momentum, reg_lambda)
```

<a id="neural_networks_numba_dev.optimizers_jit.conv_sgd_update_layers"></a>

#### conv\_sgd\_update\_layers

```python
@njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
def conv_sgd_update_layers(velocity, indices, weights, biases, dWs, dbs,
                           learning_rate, momentum, reg_lambda)
```

<a id="neural_networks_numba_dev.optimizers_jit.spec_adadelta"></a>

#### spec\_adadelta

<a id="neural_networks_numba_dev.optimizers_jit.JITAdadeltaOptimizer"></a>

## JITAdadeltaOptimizer Objects

```python
@jitclass(spec_adadelta)
class JITAdadeltaOptimizer()
```

Adadelta optimizer class for training neural networks.
Formula:
    E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * g^2
    Delta_x = - (sqrt(E[delta_x^2]_{t-1} + epsilon) / sqrt(E[g^2]_t + epsilon)) * g
    E[delta_x^2]_t = rho * E[delta_x^2]_{t-1} + (1 - rho) * Delta_x^2
Derived from: https://arxiv.org/abs/1212.5701
Args:
    learning_rate (float, optional): The learning rate for the optimizer. Defaults to 1.0.
    rho (float, optional): The decay rate. Defaults to 0.95.
    epsilon (float, optional): A small value to prevent division by zero. Defaults to 1e-6.
    reg_lambda (float, optional): The regularization parameter. Defaults to 0.0.

<a id="neural_networks_numba_dev.optimizers_jit.JITAdadeltaOptimizer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(learning_rate=1.0, rho=0.95, epsilon=1e-6, reg_lambda=0.0)
```

<a id="neural_networks_numba_dev.optimizers_jit.JITAdadeltaOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the running averages for each layer's weights.
Args: layers (list): List of layers in the neural network.
Returns: None

<a id="neural_networks_numba_dev.optimizers_jit.JITAdadeltaOptimizer.update"></a>

#### update

```python
def update(layer, dW, db, index, layer_type)
```

Updates the weights and biases of a layer using the Adadelta optimization algorithm.
Args:
    layer (Layer): The layer to update.
    dW (ndarray): The gradient of the weights.
    db (ndarray): The gradient of the biases.
    index (int): The index of the layer.
Returns: None

<a id="neural_networks_numba_dev.optimizers_jit.JITAdadeltaOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

<a id="neural_networks_numba_dev.optimizers_jit.dense_adadelta_update_layers"></a>

#### dense\_adadelta\_update\_layers

```python
@njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
def dense_adadelta_update_layers(E_g2, E_delta_x2, indices, weights, biases,
                                 dWs, dbs, learning_rate, rho, epsilon,
                                 reg_lambda)
```

<a id="neural_networks_numba_dev.optimizers_jit.conv_adadelta_update_layers"></a>

#### conv\_adadelta\_update\_layers

```python
@njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
def conv_adadelta_update_layers(E_g2, E_delta_x2, indices, weights, biases,
                                dWs, dbs, learning_rate, rho, epsilon,
                                reg_lambda)
```

<a id="neural_networks_numba_dev.schedulers"></a>

# neural\_networks\_numba\_dev.schedulers

<a id="neural_networks_numba_dev.schedulers.lr_scheduler_step"></a>

## lr\_scheduler\_step Objects

```python
class lr_scheduler_step()
```

Learning rate scheduler class for training neural networks.
Reduces the learning rate by a factor of lr_decay every lr_decay_epoch epochs.
Args:
    optimizer (Optimizer): The optimizer to adjust the learning rate for.
    lr_decay (float, optional): The factor to reduce the learning rate by. Defaults to 0.1.
    lr_decay_epoch (int, optional): The number of epochs to wait before decaying the learning rate. Defaults to 10

<a id="neural_networks_numba_dev.schedulers.lr_scheduler_step.__init__"></a>

#### \_\_init\_\_

```python
def __init__(optimizer, lr_decay=0.1, lr_decay_epoch=10)
```

<a id="neural_networks_numba_dev.schedulers.lr_scheduler_step.__repr__"></a>

#### \_\_repr\_\_

```python
def __repr__()
```

<a id="neural_networks_numba_dev.schedulers.lr_scheduler_step.step"></a>

#### step

```python
def step(epoch)
```

Adjusts the learning rate based on the current epoch. Decays the learning rate by lr_decay every lr_decay_epoch epochs.
Args:
    epoch (int): The current epoch number.
Returns: None

<a id="neural_networks_numba_dev.schedulers.lr_scheduler_step.reduce"></a>

#### reduce

```python
def reduce()
```

<a id="neural_networks_numba_dev.schedulers.lr_scheduler_exp"></a>

## lr\_scheduler\_exp Objects

```python
class lr_scheduler_exp()
```

Learning rate scheduler class for training neural networks.
Reduces the learning rate exponentially by lr_decay every lr_decay_epoch epochs.

<a id="neural_networks_numba_dev.schedulers.lr_scheduler_exp.__init__"></a>

#### \_\_init\_\_

```python
def __init__(optimizer, lr_decay=0.1, lr_decay_epoch=10)
```

<a id="neural_networks_numba_dev.schedulers.lr_scheduler_exp.__repr__"></a>

#### \_\_repr\_\_

```python
def __repr__()
```

<a id="neural_networks_numba_dev.schedulers.lr_scheduler_exp.step"></a>

#### step

```python
def step(epoch)
```

Adjusts the learning rate based on the current epoch. Decays the learning rate by lr_decay every lr_decay_epoch epochs.
Args:
    epoch (int): The current epoch number.
Returns: None

<a id="neural_networks_numba_dev.schedulers.lr_scheduler_exp.reduce"></a>

#### reduce

```python
def reduce()
```

<a id="neural_networks_numba_dev.schedulers.lr_scheduler_plateau"></a>

## lr\_scheduler\_plateau Objects

```python
class lr_scheduler_plateau()
```

A custom learning rate scheduler that adjusts the learning rate based on the plateau of the loss function.
Args:
    lr_scheduler (object): The learning rate scheduler object.
    patience (int): The number of epochs to wait for improvement before reducing the learning rate. Default is 5.
    threshold (float): The minimum improvement threshold required to update the best loss. Default is 0.01.
Methods:
    step(loss): Updates the learning rate based on the loss value.

<a id="neural_networks_numba_dev.schedulers.lr_scheduler_plateau.__init__"></a>

#### \_\_init\_\_

```python
def __init__(lr_scheduler, patience=5, threshold=0.01)
```

<a id="neural_networks_numba_dev.schedulers.lr_scheduler_plateau.__repr__"></a>

#### \_\_repr\_\_

```python
def __repr__()
```

<a id="neural_networks_numba_dev.schedulers.lr_scheduler_plateau.step"></a>

#### step

```python
def step(epoch, loss)
```

Updates the learning rate based on the loss value.
Args:
    loss (float): The current loss value.

<a id="svm"></a>

# svm

<a id="svm.__all__"></a>

#### \_\_all\_\_

<a id="svm.baseSVM"></a>

# svm.baseSVM

<a id="svm.baseSVM.BaseSVM"></a>

## BaseSVM Objects

```python
class BaseSVM()
```

BaseSVM: A base class for Support Vector Machines (SVM) with kernel support.

This class provides the foundation for implementing SVM models with various kernels
and supports both classification and regression tasks.

Attributes:
    C (float): Regularization parameter. Default is 1.0.
    tol (float): Tolerance for stopping criteria. Default is 1e-4.
    max_iter (int): Maximum number of iterations for optimization. Default is 1000.
    learning_rate (float): Step size for optimization. Default is 0.01.
    kernel (str): Kernel type ('linear', 'poly', 'rbf', or 'sigmoid'). Default is 'linear'.
    degree (int): Degree for polynomial kernel. Default is 3.
    gamma (str or float): Kernel coefficient ('scale', 'auto', or float). Default is 'scale'.
    coef0 (float): Independent term in poly and sigmoid kernels. Default is 0.0.
    regression (bool): Whether to use regression (SVR) or classification (SVC). Default is False.
    w (ndarray): Weight vector for linear kernel.
    b (float): Bias term.
    support_vectors_ (ndarray): Support vectors identified during training.
    support_vector_labels_ (ndarray): Labels of the support vectors.
    support_vector_alphas_ (ndarray): Lagrange multipliers for the support vectors.

Methods:
    __init__(self, C=1.0, tol=1e-4, max_iter=1000, learning_rate=0.01, kernel='linear', degree=3, gamma='scale', coef0=0.0, regression=False):
        Initializes the BaseSVM instance with specified hyperparameters.
    fit(self, X, y=None):
        Fits the SVM model to the training data.
    _fit(self, X, y):
        Abstract method to be implemented by subclasses for training.
    _compute_kernel(self, X1, X2):
        Computes the kernel function between two input matrices.
    decision_function(self, X):
        Computes the decision function for input samples.
    predict(self, X):
        Predicts class labels for input samples.
    score(self, X, y):
        Computes the mean accuracy of the model on the given test data.
    get_params(self, deep=True):
        Retrieves the hyperparameters of the model.
    set_params(self, **parameters):
        Sets the hyperparameters of the model.
    __sklearn_is_fitted__(self):
        Checks if the model has been fitted (for sklearn compatibility).

<a id="svm.baseSVM.BaseSVM.__init__"></a>

#### \_\_init\_\_

```python
def __init__(C=1.0,
             tol=1e-4,
             max_iter=1000,
             learning_rate=0.01,
             kernel="linear",
             degree=3,
             gamma="scale",
             coef0=0.0,
             regression=False)
```

Initializes the BaseSVM instance with specified hyperparameters.

Args:
    C: (float) - Regularization parameter. Default is 1.0.
    tol: (float) - Tolerance for stopping criteria. Default is 1e-4.
    max_iter: (int) - Maximum number of iterations for optimization. Default is 1000.
    learning_rate: (float) - Step size for optimization. Default is 0.01.
    kernel: (str) - Kernel type ('linear', 'poly', 'rbf', or 'sigmoid'). Default is 'linear'.
    degree: (int) - Degree for polynomial kernel. Default is 3.
    gamma: (str or float) - Kernel coefficient ('scale', 'auto', or float). Default is 'scale'.
    coef0: (float) - Independent term in poly and sigmoid kernels. Default is 0.0.
    regression: (bool) - Whether to use regression (SVR) or classification (SVC). Default is False.

<a id="svm.baseSVM.BaseSVM.fit"></a>

#### fit

```python
def fit(X, y=None)
```

Fits the SVM model to the training data.

Args:
    X: (array-like of shape (n_samples, n_features)) - Training vectors.
    y: (array-like of shape (n_samples,)) - Target values. Default is None.

Returns:
    self: (BaseSVM) - The fitted instance.

<a id="svm.baseSVM.BaseSVM._fit"></a>

#### \_fit

```python
def _fit(X, y)
```

Abstract method to be implemented by subclasses for training.

Args:
    X: (array-like of shape (n_samples, n_features)) - Training vectors.
    y: (array-like of shape (n_samples,)) - Target values.

Raises:
    NotImplementedError: If the method is not overridden by subclasses.

<a id="svm.baseSVM.BaseSVM._compute_kernel"></a>

#### \_compute\_kernel

```python
def _compute_kernel(X1, X2)
```

Computes the kernel function between two input matrices.

Args:
    X1: (array-like of shape (n_samples1, n_features)) - First input matrix.
    X2: (array-like of shape (n_samples2, n_features)) - Second input matrix.

Returns:
    kernel_matrix: (ndarray of shape (n_samples1, n_samples2)) - Computed kernel matrix.

<a id="svm.baseSVM.BaseSVM.decision_function"></a>

#### decision\_function

```python
def decision_function(X)
```

Computes the decision function for input samples.

Args:
    X: (array-like of shape (n_samples, n_features)) - Input samples.

Returns:
    decision_values: (ndarray of shape (n_samples,)) - Decision function values.

<a id="svm.baseSVM.BaseSVM.predict"></a>

#### predict

```python
def predict(X)
```

Predicts class labels for input samples.

Args:
    X: (array-like of shape (n_samples, n_features)) - Input samples.

Returns:
    predicted_labels: (ndarray of shape (n_samples,)) - Predicted class labels.

<a id="svm.baseSVM.BaseSVM.score"></a>

#### score

```python
def score(X, y)
```

Computes the mean accuracy of the model on the given test data.

Args:
    X: (array-like of shape (n_samples, n_features)) - Test samples.
    y: (array-like of shape (n_samples,)) - True class labels.

Returns:
    score: (float) - Mean accuracy of predictions.

<a id="svm.baseSVM.BaseSVM.get_params"></a>

#### get\_params

```python
def get_params(deep=True)
```

Retrieves the hyperparameters of the model.

Args:
    deep: (bool) - If True, returns parameters of subobjects as well. Default is True.

Returns:
    params: (dict) - Dictionary of hyperparameter names and values.

<a id="svm.baseSVM.BaseSVM.set_params"></a>

#### set\_params

```python
def set_params(**parameters)
```

Sets the hyperparameters of the model.

Args:
    **parameters: (dict) - Hyperparameter names and values.

Returns:
    self: (BaseSVM) - The updated estimator instance.

<a id="svm.baseSVM.BaseSVM.__sklearn_is_fitted__"></a>

#### \_\_sklearn\_is\_fitted\_\_

```python
def __sklearn_is_fitted__()
```

Checks if the model has been fitted (for sklearn compatibility).

Returns:
    fitted: (bool) - True if the model has been fitted, otherwise False.

<a id="svm.generalizedSVM"></a>

# svm.generalizedSVM

<a id="svm.generalizedSVM.GeneralizedSVR"></a>

## GeneralizedSVR Objects

```python
class GeneralizedSVR(BaseSVM)
```

GeneralizedSVR: A Support Vector Regression (SVR) model with support for multiple kernels.

This class implements an SVR model using gradient descent for optimization. It supports
linear and non-linear kernels, including polynomial and RBF kernels.

Attributes:
    C (float): Regularization parameter. Default is 1.0.
    tol (float): Tolerance for stopping criteria. Default is 1e-4.
    max_iter (int): Maximum number of iterations for gradient descent. Default is 1000.
    learning_rate (float): Learning rate for gradient descent. Default is 0.01.
    epsilon (float): Epsilon parameter for epsilon-insensitive loss. Default is 0.1.
    kernel (str): Kernel type ('linear', 'poly', 'rbf'). Default is 'linear'.
    degree (int): Degree of the polynomial kernel function ('poly'). Ignored by other kernels. Default is 3.
    gamma (str or float): Kernel coefficient for 'rbf' and 'poly'. Default is 'scale'.
    coef0 (float): Independent term in kernel function ('poly'). Default is 0.0.

Methods:
    __init__(self, C=1.0, tol=1e-4, max_iter=1000, learning_rate=0.01, epsilon=0.1, kernel="linear", degree=3, gamma="scale", coef0=0.0):
        Initialize the GeneralizedSVR model with specified hyperparameters.
    _fit(self, X, y):
        Fit the GeneralizedSVR model to the training data using gradient descent.
    predict(self, X):
        Predict continuous target values for input samples.
    decision_function(self, X):
        Compute raw decision function values for input samples.
    score(self, X, y):
        Compute the coefficient of determination (R▓ score) for the model's predictions.

Raises:
    ValueError: If numerical instability is detected during training.

<a id="svm.generalizedSVM.GeneralizedSVR.__init__"></a>

#### \_\_init\_\_

```python
def __init__(C=1.0,
             tol=1e-4,
             max_iter=1000,
             learning_rate=0.01,
             epsilon=0.1,
             kernel="linear",
             degree=3,
             gamma="scale",
             coef0=0.0)
```

Initializes the GeneralizedSVR model with specified hyperparameters.

Args:
    C: (float) - Regularization parameter. Default is 1.0.
    tol: (float) - Tolerance for stopping criteria. Default is 1e-4.
    max_iter: (int) - Maximum number of iterations for gradient descent. Default is 1000.
    learning_rate: (float) - Learning rate for gradient descent. Default is 0.01.
    epsilon: (float) - Epsilon parameter for epsilon-insensitive loss. Default is 0.1.
    kernel: (str) - Kernel type ('linear', 'poly', 'rbf'). Default is 'linear'.
    degree: (int) - Degree of the polynomial kernel function ('poly'). Ignored by other kernels. Default is 3.
    gamma: (str or float) - Kernel coefficient for 'rbf' and 'poly'. Default is 'scale'.
    coef0: (float) - Independent term in kernel function ('poly'). Default is 0.0.

<a id="svm.generalizedSVM.GeneralizedSVR._fit"></a>

#### \_fit

```python
def _fit(X, y)
```

Fit the GeneralizedSVR model to the training data using gradient descent.

<a id="svm.generalizedSVM.GeneralizedSVR.predict"></a>

#### predict

```python
def predict(X)
```

Predict continuous target values for input samples.

<a id="svm.generalizedSVM.GeneralizedSVR.decision_function"></a>

#### decision\_function

```python
def decision_function(X)
```

Compute raw decision function values for input samples.

<a id="svm.generalizedSVM.GeneralizedSVR.score"></a>

#### score

```python
def score(X, y)
```

Compute the coefficient of determination (R▓ score) for the model's predictions.

<a id="svm.generalizedSVM.GeneralizedSVC"></a>

## GeneralizedSVC Objects

```python
class GeneralizedSVC(BaseSVM)
```

GeneralizedSVC: A Support Vector Classifier (SVC) model with support for multiple kernels.

This class implements an SVC model using gradient descent for optimization. It supports
linear and non-linear kernels, including polynomial and RBF kernels.

Attributes:
    C (float): Regularization parameter. Default is 1.0.
    tol (float): Tolerance for stopping criteria. Default is 1e-4.
    max_iter (int): Maximum number of iterations for gradient descent. Default is 1000.
    learning_rate (float): Learning rate for gradient descent. Default is 0.01.
    kernel (str): Kernel type ('linear', 'poly', 'rbf'). Default is 'linear'.
    degree (int): Degree of the polynomial kernel function ('poly'). Ignored by other kernels. Default is 3.
    gamma (str or float): Kernel coefficient for 'rbf' and 'poly'. Default is 'scale'.
    coef0 (float): Independent term in kernel function ('poly'). Default is 0.0.

Methods:
    __init__(self, C=1.0, tol=1e-4, max_iter=1000, learning_rate=0.01, kernel="linear", degree=3, gamma="scale", coef0=0.0):
        Initialize the GeneralizedSVC model with specified hyperparameters.
    _fit(self, X, y):
        Fit the GeneralizedSVC model to the training data using gradient descent.
    _predict_binary(self, X):
        Predict binary class labels for input samples.
    _predict_multiclass(self, X):
        Predict multi-class labels using one-vs-rest strategy.
    decision_function(self, X):
        Compute raw decision function values for input samples.
    _score_binary(self, X, y):
        Compute the accuracy score for binary classification.
    _score_multiclass(self, X, y):
        Compute the accuracy score for multi-class classification.

Raises:
    ValueError: If numerical instability is detected during training.

<a id="svm.generalizedSVM.GeneralizedSVC.__init__"></a>

#### \_\_init\_\_

```python
def __init__(C=1.0,
             tol=1e-4,
             max_iter=1000,
             learning_rate=0.01,
             kernel="linear",
             degree=3,
             gamma="scale",
             coef0=0.0)
```

Initializes the GeneralizedSVC model with specified hyperparameters.

Args:
    C: (float) - Regularization parameter. Default is 1.0.
    tol: (float) - Tolerance for stopping criteria. Default is 1e-4.
    max_iter: (int) - Maximum number of iterations for gradient descent. Default is 1000.
    learning_rate: (float) - Learning rate for gradient descent. Default is 0.01.
    kernel: (str) - Kernel type ('linear', 'poly', 'rbf'). Default is 'linear'.
    degree: (int) - Degree of the polynomial kernel function ('poly'). Ignored by other kernels. Default is 3.
    gamma: (str or float) - Kernel coefficient for 'rbf' and 'poly'. Default is 'scale'.
    coef0: (float) - Independent term in kernel function ('poly'). Default is 0.0.

<a id="svm.generalizedSVM.GeneralizedSVC._fit"></a>

#### \_fit

```python
def _fit(X, y)
```

Fit the GeneralizedSVC model to the training data using gradient descent.

<a id="svm.generalizedSVM.GeneralizedSVC._predict_binary"></a>

#### \_predict\_binary

```python
def _predict_binary(X)
```

Predict binary class labels for input samples.

<a id="svm.generalizedSVM.GeneralizedSVC._predict_multiclass"></a>

#### \_predict\_multiclass

```python
def _predict_multiclass(X)
```

Predict multi-class labels using one-vs-rest strategy.

<a id="svm.generalizedSVM.GeneralizedSVC._score_binary"></a>

#### \_score\_binary

```python
def _score_binary(X, y)
```

Compute the accuracy score for binary classification.

<a id="svm.generalizedSVM.GeneralizedSVC._score_multiclass"></a>

#### \_score\_multiclass

```python
def _score_multiclass(X, y)
```

Compute the accuracy score for multi-class classification.

<a id="svm.generalizedSVM.GeneralizedSVC.decision_function"></a>

#### decision\_function

```python
def decision_function(X)
```

Compute raw decision function values for input samples.

<a id="svm.linerarSVM"></a>

# svm.linerarSVM

<a id="svm.linerarSVM.LinearSVC"></a>

## LinearSVC Objects

```python
class LinearSVC(BaseSVM)
```

LinearSVC is a linear Support Vector Classifier (SVC) implementation that uses gradient descent for optimization.

It supports binary and multi-class classification using a one-vs-rest strategy.

Attributes:
    C (float): Regularization parameter. Default is 1.0.
    tol (float): Tolerance for stopping criteria. Default is 1e-4.
    max_iter (int): Maximum number of iterations for gradient descent. Default is 1000.
    learning_rate (float): Learning rate for gradient descent. Default is 0.01.
    numba (bool): Whether to use Numba-accelerated computations. Default is False.
    w (ndarray): Weight vector for the linear model.
    b (float): Bias term for the linear model.
    numba_available (bool): Indicates if Numba is available for use.

Methods:
    __init__(self, C=1.0, tol=1e-4, max_iter=1000, learning_rate=0.01, numba=False):
        Initializes the LinearSVC instance with hyperparameters and checks for Numba availability.
    _fit(self, X, y):
        Fits the LinearSVC model to the training data using gradient descent.
    _predict_binary(self, X):
        Predicts class labels {-1, 1} for binary classification.
    _predict_multiclass(self, X):
        Predicts class labels for multi-class classification using one-vs-rest strategy.
    decision_function(self, X):
        Computes raw decision function values before thresholding.
    _score_binary(self, X, y):
        Computes the mean accuracy of predictions for binary classification.
    _score_multiclass(self, X, y):
        Computes the mean accuracy of predictions for multi-class classification.

<a id="svm.linerarSVM.LinearSVC.__init__"></a>

#### \_\_init\_\_

```python
def __init__(C=1.0, tol=1e-4, max_iter=1000, learning_rate=0.01, numba=False)
```

Initializes the LinearSVC instance with hyperparameters and checks for Numba availability.

Args:
    C: (float) - Regularization parameter. Default is 1.0.
    tol: (float) - Tolerance for stopping criteria. Default is 1e-4.
    max_iter: (int) - Maximum number of iterations for gradient descent. Default is 1000.
    learning_rate: (float) - Learning rate for gradient descent. Default is 0.01.
    numba: (bool) - Whether to use Numba-accelerated computations. Default is False.

<a id="svm.linerarSVM.LinearSVC._fit"></a>

#### \_fit

```python
def _fit(X, y)
```

Implement the fitting procedure for LinearSVC using gradient descent.

Args:
    X: (array-like of shape (n_samples, n_features)) - Training vectors.
    y: (array-like of shape (n_samples,)) - Target labels in {-1, 1}.

Returns:
    self: (LinearSVC) - The fitted instance.

Algorithm:
    Initialize Parameters: Initialize the weight vector w and bias b.
    Set Hyperparameters: Define the learning rate and the number of iterations.
    Gradient Descent Loop: Iterate over the dataset to update the weights and bias using gradient descent.
    Compute Hinge Loss: Calculate the hinge loss and its gradient.
    Update Parameters: Update the weights and bias using the gradients.
    Stopping Criteria: Check for convergence based on the tolerance level

<a id="svm.linerarSVM.LinearSVC._predict_binary"></a>

#### \_predict\_binary

```python
def _predict_binary(X)
```

Predict class labels for binary classification.

Args:
    X (array-like of shape (n_samples, n_features)): Input samples.

Returns:
    y_pred (array of shape (n_samples,)): Predicted class labels {-1, 1}.

<a id="svm.linerarSVM.LinearSVC._predict_multiclass"></a>

#### \_predict\_multiclass

```python
def _predict_multiclass(X)
```

Predict class labels for multi-class classification using one-vs-rest strategy.

Args:
    X (array-like of shape (n_samples, n_features)): Input samples.

Returns:
    predicted_labels (array of shape (n_samples,)): Predicted class labels.

<a id="svm.linerarSVM.LinearSVC.decision_function"></a>

#### decision\_function

```python
def decision_function(X)
```

Compute raw decision function values before thresholding.

Args:
    X (array-like of shape (n_samples, n_features)): Input samples.

Returns:
    scores (array of shape (n_samples,)): Decision function values.

<a id="svm.linerarSVM.LinearSVC._score_binary"></a>

#### \_score\_binary

```python
def _score_binary(X, y)
```

Compute the mean accuracy of predictions for binary classification.

Args:
    X (array-like of shape (n_samples, n_features)): Test samples.
    y (array-like of shape (n_samples,)): True labels.

Returns:
    score (float): Mean accuracy of predictions.

<a id="svm.linerarSVM.LinearSVC._score_multiclass"></a>

#### \_score\_multiclass

```python
def _score_multiclass(X, y)
```

Compute the mean accuracy of predictions for multi-class classification.

Args:
    X (array-like of shape (n_samples, n_features)): Test samples.
    y (array-like of shape (n_samples,)): True labels.

Returns:
    score (float): Mean accuracy of predictions.

<a id="svm.linerarSVM.LinearSVR"></a>

## LinearSVR Objects

```python
class LinearSVR(BaseSVM)
```

LinearSVR: A linear Support Vector Regression (SVR) model using epsilon-insensitive loss.

This class implements a linear SVR model with support for mini-batch gradient descent
and optional acceleration using Numba. It is designed for regression tasks and uses
epsilon-insensitive loss to handle errors within a specified margin.

Attributes:
    C (float): Regularization parameter. Default is 1.0.
    tol (float): Tolerance for stopping criteria. Default is 1e-4.
    max_iter (int): Maximum number of iterations for gradient descent. Default is 1000.
    learning_rate (float): Learning rate for gradient descent. Default is 0.01.
    epsilon (float): Epsilon parameter for epsilon-insensitive loss. Default is 0.1.
    numba (bool): Whether to use Numba for acceleration. Default is False.
    w (ndarray): Weight vector of the model.
    b (float): Bias term of the model.
    numba_available (bool): Indicates if Numba is available for acceleration.
    X_train (ndarray): Training data used for fitting.
    y_train (ndarray): Target values used for fitting.

Methods:
    __init__(self, C=1.0, tol=1e-4, max_iter=1000, learning_rate=0.01, epsilon=0.1, numba=False):
        Initialize the LinearSVR model with specified hyperparameters.
    _fit(self, X, y):
        Fit the LinearSVR model to the training data using mini-batch gradient descent.
    predict(self, X):
        Predict continuous target values for input samples.
    decision_function(self, X):
        Compute raw decision function values for input samples.
    score(self, X, y):
        Compute the coefficient of determination (R▓ score) for the model's predictions.

Raises:
    ValueError: If a non-linear kernel is specified, as LinearSVR only supports linear kernels.

<a id="svm.linerarSVM.LinearSVR.__init__"></a>

#### \_\_init\_\_

```python
def __init__(C=1.0,
             tol=1e-4,
             max_iter=1000,
             learning_rate=0.01,
             epsilon=0.1,
             numba=False)
```

Initializes the LinearSVR instance with hyperparameters and checks for Numba availability.

Args:
    C: (float) - Regularization parameter. Default is 1.0.
    tol: (float) - Tolerance for stopping criteria. Default is 1e-4.
    max_iter: (int) - Maximum number of iterations for gradient descent. Default is 1000.
    learning_rate: (float) - Learning rate for gradient descent. Default is 0.01.
    epsilon: (float) - Epsilon parameter for epsilon-insensitive loss. Default is 0.1.
    numba: (bool) - Whether to use Numba-accelerated computations. Default is False.

Returns:
    None

<a id="svm.linerarSVM.LinearSVR._fit"></a>

#### \_fit

```python
def _fit(X, y)
```

Implement the fitting procedure for LinearSVR using the epsilon-insensitive loss.

Args:
    X: (array-like of shape (n_samples, n_features)) - Training vectors.
    y: (array-like of shape (n_samples,)) - Target values.

Returns:
    self: (LinearSVR) - The fitted instance.

Algorithm:
    Initialize Parameters: Initialize the weight vector w and bias b.
    Set Hyperparameters: Define the learning rate and the number of iterations.
    Gradient Descent Loop: Iterate over the dataset to update the weights and bias using gradient descent.
    Compute Epsilon-Insensitive Loss: Calculate the epsilon-insensitive loss and its gradient.
    Update Parameters: Update the weights and bias using the gradients.
    Stopping Criteria: Check for convergence based on the tolerance level

<a id="svm.linerarSVM.LinearSVR.predict"></a>

#### predict

```python
def predict(X)
```

Predict continuous target values for input samples.

Args:
    X: (array-like of shape (n_samples, n_features)) - Input samples.

Returns:
    y_pred: (array of shape (n_samples,)) - Predicted values.

<a id="svm.linerarSVM.LinearSVR.decision_function"></a>

#### decision\_function

```python
def decision_function(X)
```

Compute raw decision function values.

Args:
    X: (array-like of shape (n_samples, n_features)) - Input samples.

Returns:
    scores: (array of shape (n_samples,)) - Predicted values.

<a id="svm.linerarSVM.LinearSVR.score"></a>

#### score

```python
def score(X, y)
```

Compute the coefficient of determination (R▓ score).

Args:
    X: (array-like of shape (n_samples, n_features)) - Test samples.
    y: (array-like of shape (n_samples,)) - True target values.

Returns:
    score: (float) - R▓ score of predictions.

<a id="svm.oneClassSVM"></a>

# svm.oneClassSVM

<a id="svm.oneClassSVM.OneClassSVM"></a>

## OneClassSVM Objects

```python
class OneClassSVM(BaseSVM)
```

OneClassSVM is a custom implementation of a One-Class Support Vector Machine (SVM) for anomaly detection using gradient descent.

It inherits from the BaseSVM class and supports various kernel functions.

Attributes:
    support_vectors_ (array-like of shape (n_support_vectors, n_features)):
        The support vectors identified during training.
    support_vector_alphas_ (array-like of shape (n_support_vectors,)):
        The Lagrange multipliers (alpha) corresponding to the support vectors.
    b (float):
        The bias term (rho) computed during training.

Methods:
    __init__(C=1.0, tol=1e-4, max_iter=1000, learning_rate=0.01, kernel="linear",
             degree=3, gamma="scale", coef0=0.0):
        Initialize the OneClassSVM with hyperparameters.
    _fit(X, y=None):
        Fit the OneClassSVM model using gradient descent for anomaly detection.
    decision_function(X):
        Compute the decision function values for the input samples.
    predict(X):
        Predict whether the input samples are inliers (1) or outliers (-1).
    score(X, y):
        Compute the mean accuracy of predictions compared to true labels.
    __sklearn_is_fitted__():
        Check if the model has been fitted. For compatibility with sklearn.

<a id="svm.oneClassSVM.OneClassSVM.__init__"></a>

#### \_\_init\_\_

```python
def __init__(C=1.0,
             tol=1e-4,
             max_iter=1000,
             learning_rate=0.01,
             kernel="linear",
             degree=3,
             gamma="scale",
             coef0=0.0)
```

Initialize the OneClassSVM with hyperparameters.

Args:
    C: (float) - Regularization parameter (default is 1.0).
    tol: (float) - Tolerance for stopping criteria (default is 1e-4).
    max_iter: (int) - Maximum number of iterations (default is 1000).
    learning_rate: (float) - Learning rate for gradient descent (default is 0.01).
    kernel: (str) - Kernel type ("linear", "poly", "rbf", "sigmoid") (default is "linear").
    degree: (int) - Degree for polynomial kernel (default is 3).
    gamma: (str or float) - Kernel coefficient ("scale", "auto", or float) (default is "scale").
    coef0: (float) - Independent term in kernel function (default is 0.0).

<a id="svm.oneClassSVM.OneClassSVM._fit"></a>

#### \_fit

```python
def _fit(X, y=None)
```

Fit the OneClassSVM model using gradient descent for anomaly detection.

Args:
    X: (array-like of shape (n_samples, n_features)) - Training vectors.
    y: (array-like of shape (n_samples,)) - Target values (ignored).

Returns:
    self: (OneClassSVM) - The fitted instance.

Algorithm:
    - Initialize weights w and bias b.
    - Use gradient descent to minimize the One-Class SVM objective:
      (1/2) ||w|^2 + b + C * sum(max(0, -(w^T x_i + b))).
    - Update w and b based on subgradients.
    - Stop when gradients are below tolerance or max iterations reached.

<a id="svm.oneClassSVM.OneClassSVM.decision_function"></a>

#### decision\_function

```python
def decision_function(X)
```

Compute the decision function values for the input samples.

<a id="svm.oneClassSVM.OneClassSVM.predict"></a>

#### predict

```python
def predict(X)
```

Predict whether the input samples are inliers (1) or outliers (-1).

<a id="svm.oneClassSVM.OneClassSVM.score"></a>

#### score

```python
def score(X, y)
```

Compute the mean accuracy of predictions.

Args:
    X: (array-like of shape (n_samples, n_features)) - Test samples.
    y: (array-like of shape (n_samples,)) - True labels (+1 for inliers, -1 for outliers).

Returns:
    score: (float) - Mean accuracy of predictions.

<a id="svm.oneClassSVM.OneClassSVM.__sklearn_is_fitted__"></a>

#### \_\_sklearn\_is\_fitted\_\_

```python
def __sklearn_is_fitted__()
```

Check if the model has been fitted. For compatibility with sklearn.

Returns:
    fitted: (bool) - True if the model has been fitted, otherwise False.

<a id="svm._LinearSVM_jit_utils"></a>

# svm.\_LinearSVM\_jit\_utils

<a id="svm._LinearSVM_jit_utils._linearSVC_minibatches"></a>

#### \_linearSVC\_minibatches

```python
@njit(fastmath=True)
def _linearSVC_minibatches(X, y, w, b, C, beta, learning_rate, batch_size)
```

Process all mini-batches for LinearSVC using gradient descent with momentum.

Args:
    X (ndarray): Training data of shape (n_samples, n_features).
    y (ndarray): Target labels of shape (n_samples,).
    w (ndarray): Weight vector of shape (n_features,).
    b (float): Bias term.
    C (float): Regularization parameter.
    beta (float): Momentum factor.
    learning_rate (float): Learning rate for gradient descent.
    batch_size (int): Size of each mini-batch.

Returns:
    w (ndarray): Updated weight vector.
    b (float): Updated bias term.
    dw (ndarray): Gradient of the weight vector.
    db (float): Gradient of the bias term.

<a id="svm._LinearSVM_jit_utils._linearSVR_minibatches"></a>

#### \_linearSVR\_minibatches

```python
@njit(fastmath=True)
def _linearSVR_minibatches(X, y, w, b, C, beta, learning_rate, batch_size,
                           epsilon)
```

Process all mini-batches for LinearSVR using gradient descent with momentum.

Args:
    X (ndarray): Training data of shape (n_samples, n_features).
    y (ndarray): Target values of shape (n_samples,).
    w (ndarray): Weight vector of shape (n_features,).
    b (float): Bias term.
    C (float): Regularization parameter.
    beta (float): Momentum factor.
    learning_rate (float): Learning rate for gradient descent.
    batch_size (int): Size of each mini-batch.
    epsilon (float): Epsilon-insensitive loss parameter.

Returns:
    w (ndarray): Updated weight vector.
    b (float): Updated bias term.
    dw (ndarray): Gradient of the weight vector.
    db (float): Gradient of the bias term.

<a id="time_series"></a>

# time\_series

<a id="time_series.__all__"></a>

#### \_\_all\_\_

<a id="time_series.arima"></a>

# time\_series.arima

<a id="time_series.arima.ARIMA"></a>

## ARIMA Objects

```python
class ARIMA()
```

ARIMA model for time series forecasting.

ARIMA is a class of models that explains a given time series based on its own past values,
its own past forecast errors, and a number of lagged forecast errors.
It is a combination of Auto-Regressive (AR), Moving Average (MA) models, and differencing (I) to make the series stationary.

The model is defined by three parameters: p, d, and q, which represent the order of the AR,
the degree of differencing, and the order of the MA components, respectively.

Attributes:
    order (tuple): The order of the ARIMA model (p, d, q).
    p (int): The order of the Auto-Regressive (AR) component.
    d (int): The degree of differencing.
    q (int): The order of the Moving Average (MA) component.
    model (array-like): The original time series data.
    fitted_model (dict): The fitted ARIMA model containing AR and MA components.
    _differenced_series (array-like): The differenced series used for fitting ARMA.
    _residuals (array-like): The residuals after fitting the AR component.

<a id="time_series.arima.ARIMA.__init__"></a>

#### \_\_init\_\_

```python
def __init__(order)
```

Initialize the ARIMA model.

ARIMA(p, d, q) model where:
    - p: Order of the Auto-Regressive (AR) component.
    - d: Degree of differencing (number of times the series is differenced).
    - q: Order of the Moving Average (MA) component.

Args:
    order (tuple): The order of the ARIMA model (p, d, q).

Selecting the right values:
    - p: Use the Partial Autocorrelation Function (PACF) plot to determine the lag where the PACF cuts off.
    - d: Use the Augmented Dickey-Fuller (ADF) test to check stationarity. Increase `d` until the series becomes stationary.
    - q: Use the Autocorrelation Function (ACF) plot to determine the lag where the ACF cuts off.

<a id="time_series.arima.ARIMA.__name__"></a>

#### \_\_name\_\_

```python
def __name__()
```

<a id="time_series.arima.ARIMA.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

<a id="time_series.arima.ARIMA.fit"></a>

#### fit

```python
def fit(time_series)
```

Fit the ARIMA model to the given time series data.

Args:
    time_series (array-like): The time series data to fit the model to.

<a id="time_series.arima.ARIMA.forecast"></a>

#### forecast

```python
def forecast(steps)
```

Forecast future values using the fitted ARIMA model.

Args:
    steps (int): The number of steps to forecast.

Returns:
    array-like: The forecasted values.

<a id="time_series.arima.ARIMA._compute_residuals"></a>

#### \_compute\_residuals

```python
def _compute_residuals(differenced_series, ar_coefficients)
```

Compute residuals from the AR model.

<a id="time_series.arima.ARIMA._compute_ar_part"></a>

#### \_compute\_ar\_part

```python
def _compute_ar_part(ar_coefficients, forecasted_values, p)
```

Compute the AR contribution to the forecast.

<a id="time_series.arima.ARIMA._compute_ma_part"></a>

#### \_compute\_ma\_part

```python
def _compute_ma_part(ma_coefficients, residuals, q)
```

Compute the MA contribution to the forecast.

<a id="time_series.arima.ARIMA._difference_series"></a>

#### \_difference\_series

```python
def _difference_series(time_series, d)
```

Perform differencing on the time series to make it stationary.

Args:
    time_series (array-like): The original time series data.
    d (int): The degree of differencing.

Returns:
    array-like: The differenced time series.

<a id="time_series.arima.ARIMA._fit_ar_model"></a>

#### \_fit\_ar\_model

```python
def _fit_ar_model(time_series, p)
```

Fit the Auto-Regressive (AR) component of the model.

Args:
    time_series (array-like): The stationary time series data.
    p (int): The order of the AR component.

Returns:
    array-like: The AR coefficients.

<a id="time_series.arima.ARIMA._fit_ma_model"></a>

#### \_fit\_ma\_model

```python
def _fit_ma_model(residuals, q)
```

Fit the Moving Average (MA) component of the model.

Args:
    residuals (array-like): The residuals from the AR model.
    q (int): The order of the MA component.

Returns:
    array-like: The MA coefficients.

<a id="time_series.arima.ARIMA._combine_ar_ma"></a>

#### \_combine\_ar\_ma

```python
def _combine_ar_ma(ar_coefficients, ma_coefficients)
```

Combine AR and MA components into a single model.

Args:
    ar_coefficients (array-like): The AR coefficients.
    ma_coefficients (array-like): The MA coefficients.

Returns:
    dict: The combined ARIMA model.

<a id="time_series.arima.ARIMA._forecast_arima"></a>

#### \_forecast\_arima

```python
def _forecast_arima(fitted_model, steps)
```

Forecast future values using the fitted ARIMA model.

Args:
    fitted_model (dict): The fitted ARIMA model containing AR and MA components.
    steps (int): The number of steps to forecast.

Returns:
    array-like: The forecasted values.

<a id="time_series.arima.ARIMA._inverse_difference"></a>

#### \_inverse\_difference

```python
def _inverse_difference(original_series, differenced_series, d)
```

Reconstruct the original series from the differenced series.

Args:
    original_series (array-like): The original time series data.
    differenced_series (array-like): The differenced time series.
    d (int): The degree of differencing.

Returns:
    array-like: The reconstructed time series.

<a id="time_series.arima.ARIMA.suggest_order"></a>

#### suggest\_order

```python
@staticmethod
def suggest_order(time_series, max_p=5, max_d=2, max_q=5)
```

Suggest the optimal ARIMA order (p, d, q) for the given time series.

Args:
    time_series (array-like): The time series data.
    max_p (int): Maximum order for AR component.
    max_d (int): Maximum degree of differencing.
    max_q (int): Maximum order for MA component.

Returns:
    tuple: The optimal order (p, d, q).

<a id="time_series.arima.ARIMA.find_best_order"></a>

#### find\_best\_order

```python
@staticmethod
def find_best_order(train_series,
                    test_series,
                    max_p=5,
                    max_d=2,
                    max_q=5,
                    subset_size=1.0)
```

Find the best ARIMA order using grid search.

Args:
    train_series (array-like): The training time series data.
    test_series (array-like): The testing time series data.
    max_p (int): Maximum order for AR component.
    max_d (int): Maximum degree of differencing.
    max_q (int): Maximum order for MA component.
    subset_size (float): Proportion of the training set to use for fitting.

Returns:
    tuple: The best order (p, d, q).

<a id="time_series.arima.SARIMA"></a>

## SARIMA Objects

```python
class SARIMA(ARIMA)
```

SARIMA model for time series forecasting.

SARIMA extends ARIMA by including seasonal components.

Attributes:
    order (tuple): The non-seasonal order of the ARIMA model (p, d, q).
    seasonal_order (tuple): The seasonal order of the SARIMA model (P, D, Q, m).
    p (int): The order of the Auto-Regressive (AR) component.
    d (int): The degree of differencing.
    q (int): The order of the Moving Average (MA) component.
    P (int): The order of the seasonal Auto-Regressive (SAR) component.
    D (int): The degree of seasonal differencing.
    Q (int): The order of the seasonal Moving Average (SMA) component.
    m (int): The number of time steps in a seasonal period.

<a id="time_series.arima.SARIMA.__init__"></a>

#### \_\_init\_\_

```python
def __init__(order=(0, 0, 0), seasonal_order=(0, 0, 0, 1))
```

Initialize the SARIMA model.

Args:
    order (tuple): Non-seasonal ARIMA order (p, d, q).
    seasonal_order (tuple): Seasonal order (P, D, Q, m).

<a id="time_series.arima.SARIMA.__name__"></a>

#### \_\_name\_\_

```python
def __name__()
```

<a id="time_series.arima.SARIMA.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

<a id="time_series.arima.SARIMA.fit"></a>

#### fit

```python
def fit(time_series)
```

Fit the SARIMA model to the given time series data.

First fits the ARIMA model on the seasonally-differenced series.
Then, forecasts the seasonally-differenced series and inverts the seasonal differencing.

Args:
    time_series (array-like): The time series data to fit the model to.

<a id="time_series.arima.SARIMA.forecast"></a>

#### forecast

```python
def forecast(steps)
```

Forecast future values using the fitted SARIMA model.

Args:
    steps (int): The number of steps to forecast.

Returns:
    array-like: The forecasted values.

<a id="time_series.arima.SARIMA._seasonal_difference"></a>

#### \_seasonal\_difference

```python
def _seasonal_difference(series, D, m)
```

Apply D rounds of lag-m differencing.

Args:
    series (array-like): The time series data.
    D (int): The degree of seasonal differencing.
    m (int): The seasonal period.

Returns:
    array-like: The seasonally differenced time series.

<a id="time_series.arima.SARIMA._inverse_seasonal_difference"></a>

#### \_inverse\_seasonal\_difference

```python
def _inverse_seasonal_difference(diff_forecast)
```

Reconstruct original scale from seasonally differenced forecasts.

Args:
    diff_forecast (array-like): The seasonally differenced forecasts.

Returns:
    array-like: The original time series.

<a id="time_series.arima.SARIMA.suggest_order"></a>

#### suggest\_order

```python
@staticmethod
def suggest_order(time_series,
                  max_p=3,
                  max_d=2,
                  max_q=3,
                  max_P=2,
                  max_D=1,
                  max_Q=2,
                  max_m=100)
```

Suggest the optimal SARIMA order for the given time series.

Args:
    time_series (array-like): The time series data.
    max_p (int): Maximum order for AR component.
    max_d (int): Maximum degree of differencing.
    max_q (int): Maximum order for MA component.
    max_P (int): Maximum order for seasonal AR component.
    max_D (int): Maximum degree of seasonal differencing.
    max_Q (int): Maximum order for seasonal MA component.
    max_m (int): Maximum seasonal period to consider.

Returns:
    tuple: The optimal orders (p, d, q, P, D, Q, m).

<a id="time_series.arima.SARIMA.find_best_order"></a>

#### find\_best\_order

```python
@staticmethod
def find_best_order(train_series,
                    test_series,
                    max_p=2,
                    max_d=1,
                    max_q=2,
                    max_P=1,
                    max_D=1,
                    max_Q=1,
                    max_m=100)
```

Find the best SARIMA order using grid search.

Args:
    train_series (array-like): The training time series data.
    test_series (array-like): The testing time series data.
    max_p, max_d, max_q: Maximum values for non-seasonal components.
    max_P, max_D, max_Q, max_m: Maximum values for seasonal components.

Returns:
    tuple: The best orders as ((p,d,q), (P,D,Q,m)).

<a id="time_series.arima.SARIMAX"></a>

## SARIMAX Objects

```python
class SARIMAX(SARIMA)
```

SARIMAX model with exogenous regressors.

SARIMAX takes the same time_series input as SARIMA, but also allows for exogenous regressors.
These are additional variables that can help explain the time series.

Two-step approach:
  1. OLS regression of y on exog to get beta + residuals
    - beta = (X'X)^-1 X'y
    - resid = y - X @ beta
  2. SARIMA fit on the residuals of the OLS regression

Forecast = SARIMA_forecast(resid) + exog_future @ beta

Attributes:
    beta (np.ndarray): The beta coefficients.
    k_exog (int): The number of exogenous variables.

<a id="time_series.arima.SARIMAX.__init__"></a>

#### \_\_init\_\_

```python
def __init__(order=(0, 0, 0), seasonal_order=(0, 0, 0, 1))
```

Initialize the SARIMAX model.

Args:
    order (tuple): Non-seasonal ARIMA order (p, d, q).
    seasonal_order (tuple): Seasonal order (P, D, Q, m).

<a id="time_series.arima.SARIMAX.__name__"></a>

#### \_\_name\_\_

```python
def __name__()
```

<a id="time_series.arima.SARIMAX.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

<a id="time_series.arima.SARIMAX.fit"></a>

#### fit

```python
def fit(time_series, exog, bound_lower=None, bound_upper=None)
```

Fit the SARIMAX model to the given time series and exogenous regressors.

Args:
    time_series (array-like): The time series data.
    exog (array-like): The exogenous regressors.
    bound_lower (float): Lower bound for beta coefficients.
    bound_upper (float): Upper bound for beta coefficients.

Returns:
    self: The fitted SARIMAX model.

<a id="time_series.arima.SARIMAX.forecast"></a>

#### forecast

```python
def forecast(steps, exog_future)
```

Forecast future values using the fitted SARIMAX model.

Args:
    steps (int): The number of steps to forecast.
    exog_future (array-like): The exogenous regressors for the future values.

Returns:
    array-like: The forecasted values.

<a id="time_series.arima.SARIMAX.suggest_order"></a>

#### suggest\_order

```python
@staticmethod
def suggest_order(endog,
                  exog,
                  max_p=3,
                  max_d=2,
                  max_q=3,
                  max_P=2,
                  max_D=1,
                  max_Q=2,
                  max_m=100)
```

Suggest ((p,d,q),(P,D,Q,m)) for SARIMAX.

Regress endog on exog to get residuals, then call SARIMA.suggest_order on residuals.

Args:
    endog (array-like): The endogenous variable.
    exog (array-like): The exogenous regressors.
    max_p, max_d, max_q: Maximum values for non-seasonal components.
    max_P, max_D, max_Q, max_m: Maximum values for seasonal components.

Returns:
    tuple: The optimal orders (p, d, q, P, D, Q, m).

<a id="time_series.arima.SARIMAX.find_best_order"></a>

#### find\_best\_order

```python
@staticmethod
def find_best_order(train_endog,
                    test_endog,
                    train_exog,
                    test_exog,
                    max_p=2,
                    max_d=1,
                    max_q=2,
                    max_P=1,
                    max_D=1,
                    max_Q=1,
                    max_m=100)
```

Grid-search over ((p,d,q),(P,D,Q,m)) to minimize MSE on test set.

Args:
    train_endog (array-like): The training endogenous variable.
    test_endog (array-like): The testing endogenous variable.
    train_exog (array-like): The training exogenous regressors.
    test_exog (array-like): The testing exogenous regressors.
    max_p, max_d, max_q: Maximum values for non-seasonal components.
    max_P, max_D, max_Q, max_m: Maximum values for seasonal components.

Returns:
    tuple: The best orders ((p,d,q),(P,D,Q,m)).

<a id="time_series.decomposition"></a>

# time\_series.decomposition

<a id="time_series.decomposition._centered_moving_average"></a>

#### \_centered\_moving\_average

```python
def _centered_moving_average(series, window)
```

Calculates centered moving average, handling even/odd windows.

<a id="time_series.decomposition.AdditiveDecomposition"></a>

## AdditiveDecomposition Objects

```python
class AdditiveDecomposition()
```

Performs classical additive decomposition of a time series.

Decomposes the series Y into Trend (T), Seasonal (S), and Residual (R) components
such that Y = T + S + R. Assumes seasonality is constant over time.

Attributes:
    period (int): The seasonal period.
    time_series (np.ndarray): The original time series data.
    trend (np.ndarray): The estimated trend component.
    seasonal (np.ndarray): The estimated seasonal component.
    residual (np.ndarray): The estimated residual component.

<a id="time_series.decomposition.AdditiveDecomposition.__init__"></a>

#### \_\_init\_\_

```python
def __init__(period)
```

Initialize the AdditiveDecomposition model.

Args:
    period (int): The seasonal period (e.g., 12 for monthly, 7 for daily). Must be > 1.

<a id="time_series.decomposition.AdditiveDecomposition.__name__"></a>

#### \_\_name\_\_

```python
def __name__()
```

<a id="time_series.decomposition.AdditiveDecomposition.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

<a id="time_series.decomposition.AdditiveDecomposition.fit"></a>

#### fit

```python
def fit(time_series)
```

Perform additive decomposition on the time series.

Args:
    time_series (array-like): The time series data. Must be 1-dimensional and have length >= 2 * period.

Returns:
    tuple: The calculated trend, seasonal, and residual components.

<a id="time_series.decomposition.AdditiveDecomposition.get_components"></a>

#### get\_components

```python
def get_components()
```

Return the calculated components.

<a id="time_series.decomposition.AdditiveDecomposition.reconstruct"></a>

#### reconstruct

```python
def reconstruct()
```

Reconstruct the series from components (Y = T + S + R).

<a id="time_series.decomposition.MultiplicativeDecomposition"></a>

## MultiplicativeDecomposition Objects

```python
class MultiplicativeDecomposition()
```

Performs classical multiplicative decomposition of a time series.

Decomposes the series Y into Trend (T), Seasonal (S), and Residual (R) components
such that Y = T * S * R. Assumes seasonality changes proportionally to the trend.

Handles non-positive values by shifting the data to be positive before decomposition.
Note: This affects the interpretation of the components.

Attributes:
    period (int): The seasonal period.
    time_series (np.ndarray): The original time series data.
    offset (float): The offset added to the series to make it positive (0 if originally positive).
    trend (np.ndarray): The estimated trend component (adjusted back to original scale).
    seasonal (np.ndarray): The estimated seasonal component (from shifted data).
    residual (np.ndarray): The estimated residual component (from shifted data).
    _trend_shifted (np.ndarray): Internal storage of trend from shifted data.

<a id="time_series.decomposition.MultiplicativeDecomposition.__init__"></a>

#### \_\_init\_\_

```python
def __init__(period)
```

Initialize the MultiplicativeDecomposition model.

Args:
    period (int): The seasonal period (e.g., 12 for monthly, 7 for daily). Must be > 1.

<a id="time_series.decomposition.MultiplicativeDecomposition.__name__"></a>

#### \_\_name\_\_

```python
def __name__()
```

<a id="time_series.decomposition.MultiplicativeDecomposition.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

<a id="time_series.decomposition.MultiplicativeDecomposition.fit"></a>

#### fit

```python
def fit(time_series)
```

Perform multiplicative decomposition on the time series.

If the series contains non-positive values, it is shifted before decomposition.
This is done to ensure positivity of the seasonal component, but affects the
interpretation of the components.

Args:
    time_series (array-like): The time series data. Must be 1-dimensional,and have length >= 2 * period.

Returns:
    tuple: The calculated trend, seasonal, and residual components.

<a id="time_series.decomposition.MultiplicativeDecomposition.get_components"></a>

#### get\_components

```python
def get_components()
```

Return the calculated components.

Note: Trend is adjusted back to original scale. Seasonal and Residual
      are derived from the shifted data if an offset was applied.

<a id="time_series.decomposition.MultiplicativeDecomposition.reconstruct"></a>

#### reconstruct

```python
def reconstruct()
```

Reconstruct the series from components.

Accounts for any offset applied during fitting.
Reconstruction formula: Y_recon = T_shifted * S * R - offset
                            = (T + offset) * S * R - offset

<a id="time_series.exponential_smoothing"></a>

# time\_series.exponential\_smoothing

<a id="time_series.exponential_smoothing.mean_squared_error"></a>

#### mean\_squared\_error

<a id="time_series.exponential_smoothing.SimpleExponentialSmoothing"></a>

## SimpleExponentialSmoothing Objects

```python
class SimpleExponentialSmoothing()
```

Simple Exponential Smoothing (SES) for non-seasonal time series without trend.

Forecasts are based on a weighted average of past observations, with weights decreasing exponentially over time.
Forecast is a flat line, this is because SES does not account for trend or seasonality.

Attributes:
    alpha (float): Smoothing parameter for the level (0 <= alpha <= 1).
    level (float): The final estimated level component after fitting.
    fitted_values (np.ndarray): The fitted values (one-step-ahead forecasts).
    model (np.ndarray): The original time series data.

<a id="time_series.exponential_smoothing.SimpleExponentialSmoothing.__init__"></a>

#### \_\_init\_\_

```python
def __init__(alpha)
```

Initialize SES model.

Args:
    alpha (float): Smoothing parameter for the level. Must be between 0 and 1.

<a id="time_series.exponential_smoothing.SimpleExponentialSmoothing.__name__"></a>

#### \_\_name\_\_

```python
def __name__()
```

<a id="time_series.exponential_smoothing.SimpleExponentialSmoothing.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

<a id="time_series.exponential_smoothing.SimpleExponentialSmoothing.fit"></a>

#### fit

```python
def fit(time_series)
```

Fit the SES model to the data.

Args:
    time_series (array-like): The time series data (1-dimensional).

Returns:
    np.ndarray: The fitted values (one-step-ahead forecasts).

<a id="time_series.exponential_smoothing.SimpleExponentialSmoothing.forecast"></a>

#### forecast

```python
def forecast(steps)
```

Generate forecasts for future steps.

Args:
    steps (int): The number of steps to forecast ahead.

Returns:
    np.ndarray: An array of forecasted values.

<a id="time_series.exponential_smoothing.DoubleExponentialSmoothing"></a>

## DoubleExponentialSmoothing Objects

```python
class DoubleExponentialSmoothing()
```

Double Exponential Smoothing (DES) / Holt's Linear Trend Method.

Extends SES to handle time series with a trend component (additive trend).

Attributes:
    alpha (float): Smoothing parameter for the level (0 <= alpha <= 1).
    beta (float): Smoothing parameter for the trend (0 <= beta <= 1).
    level (float): The final estimated level component after fitting.
    trend (float): The final estimated trend component after fitting.
    fitted_values (np.ndarray): The one-step-ahead forecasts made during fitting.
    model (np.ndarray): The original time series data.

<a id="time_series.exponential_smoothing.DoubleExponentialSmoothing.__init__"></a>

#### \_\_init\_\_

```python
def __init__(alpha, beta)
```

Initialize DES model.

Args:
    alpha (float): Smoothing parameter for the level (0 <= alpha <= 1).
    beta (float): Smoothing parameter for the trend (0 <= beta <= 1).

<a id="time_series.exponential_smoothing.DoubleExponentialSmoothing.__name__"></a>

#### \_\_name\_\_

```python
def __name__()
```

<a id="time_series.exponential_smoothing.DoubleExponentialSmoothing.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

<a id="time_series.exponential_smoothing.DoubleExponentialSmoothing.fit"></a>

#### fit

```python
def fit(time_series)
```

Fit the DES model to the data.

Args:
    time_series (array-like): The time series data (1-dimensional). Requires at least 2 points.

Returns:
    np.ndarray: The fitted values (one-step-ahead forecasts).

<a id="time_series.exponential_smoothing.DoubleExponentialSmoothing.forecast"></a>

#### forecast

```python
def forecast(steps)
```

Generate forecasts for future steps.

Args:
    steps (int): The number of steps to forecast ahead.

Returns:
    np.ndarray: An array of forecasted values.

<a id="time_series.exponential_smoothing.DoubleExponentialSmoothing.find_best_alpha_beta"></a>

#### find\_best\_alpha\_beta

```python
def find_best_alpha_beta(train_series,
                         test_series,
                         alpha_values=None,
                         beta_values=None,
                         set_best=False)
```

Find the best alpha and beta values for the DES model.

Args:
    train_series (array-like): The training time series data (1-dimensional).
    test_series (array-like): The testing time series data (1-dimensional).
    alpha_values (list, optional): List of alpha values to evaluate. Defaults to [0.1, 0.2, ..., 0.9].
    beta_values (list, optional): List of beta values to evaluate. Defaults to [0.1, 0.2, ..., 0.9].
    set_best (bool, optional): If True, set the best alpha and beta values to the model. Defaults to False.

Returns:
    tuple: Best alpha and beta values based on mean squared error.

<a id="time_series.exponential_smoothing.TripleExponentialSmoothing"></a>

## TripleExponentialSmoothing Objects

```python
class TripleExponentialSmoothing()
```

Triple Exponential Smoothing (TES) / Holt-Winters Method (Additive Seasonality).

Extends DES to handle time series with both trend and seasonality (additive).

Attributes:
    alpha (float): Smoothing parameter for the level (0 <= alpha <= 1).
    beta (float): Smoothing parameter for the trend (0 <= beta <= 1).
    gamma (float): Smoothing parameter for seasonality (0 <= gamma <= 1).
    period (int): The seasonal period (must be > 1).
    level (float): The final estimated level component after fitting.
    trend (float): The final estimated trend component after fitting.
    season (np.ndarray): The final estimated seasonal components (length `period`).
    fitted_values (np.ndarray): The one-step-ahead forecasts made during fitting.
    model (np.ndarray): The original time series data.

<a id="time_series.exponential_smoothing.TripleExponentialSmoothing.__init__"></a>

#### \_\_init\_\_

```python
def __init__(alpha, beta, gamma, period)
```

Initialize TES model (Additive Seasonality).

Args:
    alpha (float): Smoothing parameter for the level (0 <= alpha <= 1).
    beta (float): Smoothing parameter for the trend (0 <= beta <= 1).
    gamma (float): Smoothing parameter for seasonality (0 <= gamma <= 1).
    period (int): The seasonal period (e.g., 12 for monthly). Must be > 1.

<a id="time_series.exponential_smoothing.TripleExponentialSmoothing.__name__"></a>

#### \_\_name\_\_

```python
def __name__()
```

<a id="time_series.exponential_smoothing.TripleExponentialSmoothing.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

<a id="time_series.exponential_smoothing.TripleExponentialSmoothing._initial_seasonal_components"></a>

#### \_initial\_seasonal\_components

```python
def _initial_seasonal_components(series, m)
```

Estimate initial seasonal components.

<a id="time_series.exponential_smoothing.TripleExponentialSmoothing.fit"></a>

#### fit

```python
def fit(time_series)
```

Fit the TES model (additive seasonality) to the data.

Args:
    time_series (array-like): The time series data (1-dimensional).
                              Length should be >= 2 * period.

Returns:
    np.ndarray: The fitted values for the time series.

<a id="time_series.exponential_smoothing.TripleExponentialSmoothing.forecast"></a>

#### forecast

```python
def forecast(steps)
```

Generate forecasts for future steps (Additive Seasonality).

Args:
    steps (int): The number of steps to forecast ahead.

Returns:
    np.ndarray: An array of forecasted values.

<a id="time_series.exponential_smoothing.TripleExponentialSmoothing.find_best_alpha_beta_gamma"></a>

#### find\_best\_alpha\_beta\_gamma

```python
def find_best_alpha_beta_gamma(train_series,
                               test_series,
                               alpha_values=None,
                               beta_values=None,
                               gamma_values=None,
                               set_best=False)
```

Find the best alpha, beta, and gamma values for the TES model.

Args:
    train_series (array-like): The training time series data (1-dimensional).
    test_series (array-like): The testing time series data (1-dimensional).
    alpha_values (list, optional): List of alpha values to evaluate. Defaults to [0.1, 0.2, ..., 0.9].
    beta_values (list, optional): List of beta values to evaluate. Defaults to [0.1, 0.2, ..., 0.9].
    gamma_values (list, optional): List of gamma values to evaluate. Defaults to [0.1, 0.2, ..., 0.9].
    set_best (bool, optional): If True, set the best alpha, beta, and gamma values to the model. Defaults to False.

Returns:
    tuple: Best alpha, beta, and gamma values based on mean squared error.

<a id="time_series.forecasting"></a>

# time\_series.forecasting

<a id="time_series.forecasting.ForecastingPipeline"></a>

## ForecastingPipeline Objects

```python
class ForecastingPipeline()
```

A customizable pipeline for time series forecasting.

This pipeline allows for the integration of preprocessing steps, a forecasting model,
and evaluation metrics to streamline the forecasting workflow.

Attributes:
    preprocessors (list): A list of preprocessing functions or objects to transform the input data.
    model (object): A forecasting model (e.g., ARIMA, SARIMA, etc.) that implements fit and predict methods.
    evaluators (list): A list of evaluation metrics or functions to assess the model's performance.

Methods:
    add_preprocessor(preprocessor): Add a preprocessing step to the pipeline.
    fit(X, y): Fit the model to the data after applying preprocessing steps.
    predict(X): Make predictions using the fitted model and preprocessing steps.
    evaluate(X, y): Evaluate the model using the provided evaluators.
    summary(): Print a summary of the pipeline configuration.

<a id="time_series.forecasting.ForecastingPipeline.__init__"></a>

#### \_\_init\_\_

```python
def __init__(preprocessors=None, model=None, evaluators=None)
```

Initialize the pipeline with optional preprocessors, model, and evaluators.

Args:
    preprocessors (list, optional): List of preprocessing functions or objects.
    model (object, optional): A forecasting model (e.g., ARIMA, SARIMA, etc.).
    evaluators (list, optional): List of evaluation metrics or functions.

<a id="time_series.forecasting.ForecastingPipeline.add_preprocessor"></a>

#### add\_preprocessor

```python
def add_preprocessor(preprocessor)
```

Add a preprocessing step to the pipeline.

Args:
    preprocessor (callable): A preprocessing function or object.

<a id="time_series.forecasting.ForecastingPipeline.remove_preprocessor"></a>

#### remove\_preprocessor

```python
def remove_preprocessor(preprocessor)
```

Remove a preprocessing step from the pipeline.

Args:
    preprocessor (callable): A preprocessing function or object to remove.

<a id="time_series.forecasting.ForecastingPipeline.add_evaluator"></a>

#### add\_evaluator

```python
def add_evaluator(evaluator)
```

Add an evaluation metric to the pipeline.

Args:
    evaluator (callable): An evaluation metric function.

<a id="time_series.forecasting.ForecastingPipeline.remove_evaluator"></a>

#### remove\_evaluator

```python
def remove_evaluator(evaluator)
```

Remove an evaluation metric from the pipeline.

Args:
    evaluator (callable): An evaluation metric function to remove.

<a id="time_series.forecasting.ForecastingPipeline.add_model"></a>

#### add\_model

```python
def add_model(model)
```

Add a forecasting model to the pipeline.

Args:
    model (object): A forecasting model (e.g., ARIMA, SARIMA, etc.).

<a id="time_series.forecasting.ForecastingPipeline.remove_model"></a>

#### remove\_model

```python
def remove_model(model)
```

Remove a forecasting model from the pipeline.

Args:
    model (object): A forecasting model (e.g., ARIMA, SARIMA, etc.) to remove.

<a id="time_series.forecasting.ForecastingPipeline.fit"></a>

#### fit

```python
def fit(X, y=None)
```

Fit the model to the data.

Args:
    X (array-like): Input features (e.g., time series data).
    y (array-like): Target values (optional). If not provided, X is used as both features and target.

<a id="time_series.forecasting.ForecastingPipeline.predict"></a>

#### predict

```python
def predict(X, steps=1)
```

Make predictions using the fitted model.

Args:
    X (array-like): Input features for prediction.
    steps (int): Number of steps to forecast ahead.

Returns:
    array-like: Predicted values.

<a id="time_series.forecasting.ForecastingPipeline.evaluate"></a>

#### evaluate

```python
def evaluate(predictions, y)
```

Evaluate the model using the provided evaluators.

Args:
    predictions (array-like): Predicted values.
    y (array-like): True target values.

Returns:
    dict: Dictionary of evaluation results.

<a id="time_series.forecasting.ForecastingPipeline.summary"></a>

#### summary

```python
def summary()
```

Print a summary of the pipeline configuration.

<a id="time_series.moving_average"></a>

# time\_series.moving\_average

<a id="time_series.moving_average.SimpleMovingAverage"></a>

## SimpleMovingAverage Objects

```python
class SimpleMovingAverage()
```

Calculates the Simple Moving Average (SMA) of a time series.

SMA smooths out fluctuations by averaging data over a defined window.

Attributes:
    window (int): The number of periods in the moving average window.
    smoothed_values (np.ndarray): The calculated SMA values. NaNs prepended.
    model (np.ndarray): The original time series data.

<a id="time_series.moving_average.SimpleMovingAverage.__init__"></a>

#### \_\_init\_\_

```python
def __init__(window)
```

Initialize SMA calculator.

Args:
    window (int): The size of the moving window. Must be > 0.

<a id="time_series.moving_average.SimpleMovingAverage.__name__"></a>

#### \_\_name\_\_

```python
def __name__()
```

<a id="time_series.moving_average.SimpleMovingAverage.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

<a id="time_series.moving_average.SimpleMovingAverage.fit"></a>

#### fit

```python
def fit(time_series)
```

Calculate the Simple Moving Average for the series.

Args:
    time_series (array-like): The time series data (1-dimensional).

<a id="time_series.moving_average.SimpleMovingAverage.get_smoothed"></a>

#### get\_smoothed

```python
def get_smoothed()
```

Return the calculated moving average series.

<a id="time_series.moving_average.SimpleMovingAverage.forecast"></a>

#### forecast

```python
def forecast(steps)
```

Generate forecasts using the last calculated moving average value.

Note: This is a naive forecast where the future is predicted to be the
last known smoothed value.

Args:
    steps (int): The number of steps to forecast ahead.

Returns:
    np.ndarray: An array of forecasted values (all the same).

<a id="time_series.moving_average.WeightedMovingAverage"></a>

## WeightedMovingAverage Objects

```python
class WeightedMovingAverage()
```

Calculates the Weighted Moving Average (WMA) of a time series.

WMA assigns different weights to data points within the window, typically giving
more importance to recent observations.

Attributes:
    window (int): The number of periods in the moving average window.
    weights (np.ndarray): The weights assigned to observations in the window.
    smoothed_values (np.ndarray): The calculated WMA values. NaNs prepended.
    model (np.ndarray): The original time series data.

<a id="time_series.moving_average.WeightedMovingAverage.__init__"></a>

#### \_\_init\_\_

```python
def __init__(window, weights=None)
```

Initialize WMA calculator.

Args:
    window (int): The size of the moving window. Must be > 0.
    weights (array-like, optional): A sequence of weights for the window.
        Length must match `window`. If None, linear weights giving more
        importance to recent points are used (e.g., [1, 2, 3] for window=3).
        Weights are normalized to sum to 1.

<a id="time_series.moving_average.WeightedMovingAverage.__name__"></a>

#### \_\_name\_\_

```python
def __name__()
```

<a id="time_series.moving_average.WeightedMovingAverage.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

<a id="time_series.moving_average.WeightedMovingAverage.fit"></a>

#### fit

```python
def fit(time_series)
```

Calculate the Weighted Moving Average for the series.

Args:
    time_series (array-like): The time series data (1-dimensional).

<a id="time_series.moving_average.WeightedMovingAverage.get_smoothed"></a>

#### get\_smoothed

```python
def get_smoothed()
```

Return the calculated moving average series.

<a id="time_series.moving_average.WeightedMovingAverage.forecast"></a>

#### forecast

```python
def forecast(steps)
```

Generate forecasts using the last calculated weighted moving average value.

Note: This is a naive forecast.

Args:
    steps (int): The number of steps to forecast ahead.

Returns:
    np.ndarray: An array of forecasted values (all the same).

<a id="time_series.moving_average.ExponentialMovingAverage"></a>

## ExponentialMovingAverage Objects

```python
class ExponentialMovingAverage()
```

Calculates the Exponential Moving Average (EMA) of a time series.

EMA gives more weight to recent observations, making it more responsive to new information.

Attributes:
    alpha (float): The smoothing factor (0 < alpha < 1).
    smoothed_values (np.ndarray): The calculated EMA values. NaNs prepended.
    model (np.ndarray): The original time series data.

<a id="time_series.moving_average.ExponentialMovingAverage.__init__"></a>

#### \_\_init\_\_

```python
def __init__(alpha)
```

Initialize EMA calculator.

Args:
    alpha (float): The smoothing factor (0 < alpha < 1).

<a id="time_series.moving_average.ExponentialMovingAverage.__name__"></a>

#### \_\_name\_\_

```python
def __name__()
```

<a id="time_series.moving_average.ExponentialMovingAverage.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

<a id="time_series.moving_average.ExponentialMovingAverage.fit"></a>

#### fit

```python
def fit(time_series)
```

Calculate the Exponential Moving Average for the series.

Args:
    time_series (array-like): The time series data (1-dimensional).

<a id="time_series.moving_average.ExponentialMovingAverage.get_smoothed"></a>

#### get\_smoothed

```python
def get_smoothed()
```

Return the calculated EMA series.

<a id="time_series.moving_average.ExponentialMovingAverage.forecast"></a>

#### forecast

```python
def forecast(steps)
```

Generate forecasts using the last calculated EMA value.

Note: This is a naive forecast where the future is predicted to be the
last known smoothed value.

Args:
    steps (int): The number of steps to forecast ahead.

Returns:
    np.ndarray: An array of forecasted values (all the same).

<a id="trees"></a>

# trees

<a id="trees.__all__"></a>

#### \_\_all\_\_

<a id="trees.adaBoostClassifier"></a>

# trees.adaBoostClassifier

<a id="trees.adaBoostClassifier.AdaBoostClassifier"></a>

## AdaBoostClassifier Objects

```python
class AdaBoostClassifier()
```

AdaBoost classifier.

Builds an additive model by sequentially fitting weak classifiers (default: decision stumps)
on modified versions of the data. Each subsequent classifier focuses more on samples
that were misclassified by the previous ensemble.

Uses the SAMME algorithm which supports multi-class classification.

Attributes:
    base_estimator_ (object): The base estimator template used for fitting.
    n_estimators (int): The maximum number of estimators at which boosting is terminated.
    learning_rate (float): Weight applied to each classifier's contribution.
    estimators_ (list): The collection of fitted base estimators.
    estimator_weights_ (np.ndarray): Weights for each estimator.
    estimator_errors_ (np.ndarray): Classification error for each estimator.
    classes_ (np.ndarray): The class labels.
    n_classes_ (int): The number of classes.

<a id="trees.adaBoostClassifier.AdaBoostClassifier.__init__"></a>

#### \_\_init\_\_

```python
def __init__(base_estimator=None,
             n_estimators=50,
             learning_rate=1.0,
             random_state=None,
             max_depth=3,
             min_samples_split=2)
```

Initialize the AdaBoostClassifier.

Args:
    base_estimator (object, optional): The base estimator from which the boosted ensemble is built.
                                      Support for sample weighting is required. If None, then
                                      the base estimator is DecisionTreeClassifier(max_depth=1).
    n_estimators (int, optional): The maximum number of estimators at which boosting is terminated.
                                  In case of perfect fit, the learning procedure is stopped early. Defaults to 50.
    learning_rate (float, optional): Weight applied to each classifier's contribution. Defaults to 1.0.
    random_state (int, optional): Controls the random seed given to the base estimator at each boosting iteration.
                                  Defaults to None.
    max_depth (int, optional): The maximum depth of the base estimator. Defaults to 3.
    min_samples_split (int, optional): The minimum number of samples required to split an internal node
                                       when using the default `ClassifierTree` base estimator. Defaults to 2.

<a id="trees.adaBoostClassifier.AdaBoostClassifier._supports_sample_weight"></a>

#### \_supports\_sample\_weight

```python
def _supports_sample_weight(estimator)
```

Check if the estimator's fit method supports sample_weight.

<a id="trees.adaBoostClassifier.AdaBoostClassifier._fit"></a>

#### \_fit

```python
def _fit(X, y)
```

Build a boosted classifier from the training set (X, y).

<a id="trees.adaBoostClassifier.AdaBoostClassifier.fit"></a>

#### fit

```python
def fit(X, y)
```

Build a boosted classifier from the training set (X, y).

<a id="trees.adaBoostClassifier.AdaBoostClassifier.decision_function"></a>

#### decision\_function

```python
def decision_function(X)
```

Compute the decision function of X.

<a id="trees.adaBoostClassifier.AdaBoostClassifier.predict_proba"></a>

#### predict\_proba

```python
def predict_proba(X)
```

Predict class probabilities for X.

<a id="trees.adaBoostClassifier.AdaBoostClassifier.predict"></a>

#### predict

```python
def predict(X)
```

Predict classes for X.

<a id="trees.adaBoostClassifier.AdaBoostClassifier.get_stats"></a>

#### get\_stats

```python
def get_stats(y_true, X=None, y_pred=None, verbose=False)
```

Calculate and optionally print evaluation metrics. Requires either X or y_pred.

<a id="trees.adaBoostClassifier.AdaBoostClassifier._calculate_metrics"></a>

#### \_calculate\_metrics

```python
def _calculate_metrics(y_true, y_pred, y_prob=None)
```

Calculate common classification metrics.

<a id="trees.adaBoostRegressor"></a>

# trees.adaBoostRegressor

<a id="trees.adaBoostRegressor.AdaBoostRegressor"></a>

## AdaBoostRegressor Objects

```python
class AdaBoostRegressor()
```

AdaBoost regressor.

Builds an additive model by sequentially fitting weak regressors (default: decision trees)
on modified versions of the data. The weights of instances are adjusted at each iteration
so that subsequent regressors focus more on instances with larger errors.

Uses the AdaBoost.R2 algorithm.

Attributes:
    base_estimator_ (object): The base estimator template used for fitting.
    n_estimators (int): The maximum number of estimators at which boosting is terminated.
    learning_rate (float): Contribution of each regressor to the final prediction.
    loss (str): The loss function to use when updating the weights ('linear', 'square', 'exponential').
    estimators_ (list): The collection of fitted base estimators.
    estimator_weights_ (np.ndarray): Weights for each estimator (alpha values, specifically log(1/beta)).
    estimator_errors_ (np.ndarray): Loss value for each estimator on the weighted training data.

<a id="trees.adaBoostRegressor.AdaBoostRegressor.__init__"></a>

#### \_\_init\_\_

```python
def __init__(base_estimator=None,
             n_estimators=50,
             learning_rate=1.0,
             loss="linear",
             random_state=None,
             max_depth=3,
             min_samples_split=2)
```

Initialize the AdaBoostRegressor.

Args:
    base_estimator (object, optional): The base estimator from which the boosted ensemble is built.
                                      Support for sample weighting is required. If None, then
                                      the base estimator is DecisionTreeRegressor(max_depth=3).
    n_estimators (int, optional): The maximum number of estimators. Defaults to 50.
    learning_rate (float, optional): Shrinks the contribution of each regressor by learning_rate. Defaults to 1.0.
    loss (str, optional): The loss function to use when updating sample weights ('linear', 'square', 'exponential').
                          Defaults to 'linear'.
    random_state (int, optional): Controls the random seed. Defaults to None.
    max_depth (int, optional): Maximum depth of the base estimator. Defaults to 3.
    min_samples_split (int, optional): Minimum number of samples required to split an internal node. Defaults to 2.

<a id="trees.adaBoostRegressor.AdaBoostRegressor._supports_sample_weight"></a>

#### \_supports\_sample\_weight

```python
def _supports_sample_weight(estimator)
```

Check if the estimator's fit method supports sample_weight.

<a id="trees.adaBoostRegressor.AdaBoostRegressor._fit"></a>

#### \_fit

```python
def _fit(X, y)
```

Build a boosted regressor from the training set (X, y).

<a id="trees.adaBoostRegressor.AdaBoostRegressor.fit"></a>

#### fit

```python
def fit(X, y)
```

Build a boosted regressor from the training set (X, y).

<a id="trees.adaBoostRegressor.AdaBoostRegressor.predict"></a>

#### predict

```python
def predict(X)
```

Predict regression target for X.

<a id="trees.adaBoostRegressor.AdaBoostRegressor.get_stats"></a>

#### get\_stats

```python
def get_stats(y_true, X=None, y_pred=None, verbose=False)
```

Calculate and optionally print evaluation metrics. Requires either X or y_pred.

<a id="trees.adaBoostRegressor.AdaBoostRegressor._calculate_metrics"></a>

#### \_calculate\_metrics

```python
def _calculate_metrics(y_true, y_pred)
```

Calculate common regression metrics.

<a id="trees.gradientBoostedClassifier"></a>

# trees.gradientBoostedClassifier

<a id="trees.gradientBoostedClassifier.GradientBoostedClassifier"></a>

## GradientBoostedClassifier Objects

```python
class GradientBoostedClassifier()
```

A Gradient Boosted Decision Tree Classifier.

This model builds an ensemble of regression trees sequentially. Each tree
is trained to predict the pseudo-residuals (gradients of the loss function)
of the previous model's predictions.

Attributes:
    X (np.ndarray): Training input features of shape (n_samples, n_features).
    y (np.ndarray): Training target class labels of shape (n_samples,).
    n_estimators (int): The number of boosting stages (trees) to perform.
    learning_rate (float): Step size shrinkage to prevent overfitting.
    max_depth (int): Maximum depth of the individual regression tree estimators.
    min_samples_split (int): Minimum number of samples required to split an internal node in a tree.
    random_seed (int or None): Controls the randomness for reproducibility (currently affects feature selection within trees if applicable).
    trees_ (list): List storing the fitted regression tree instances for each boosting stage (and for each class in multiclass).
    classes_ (np.ndarray): The unique class labels found in the target variable `y`.
    n_classes_ (int): The number of unique classes.
    init_estimator_ (float or np.ndarray): The initial prediction model (predicts log-odds).
    loss_ (str): The loss function used ('log_loss' for binary, 'multinomial' for multi-class).

<a id="trees.gradientBoostedClassifier.GradientBoostedClassifier.__init__"></a>

#### \_\_init\_\_

```python
def __init__(X=None,
             y=None,
             n_estimators: int = 100,
             learning_rate: float = 0.1,
             max_depth: int = 3,
             min_samples_split: int = 2,
             random_seed: int = None)
```

Initializes the Gradient Boosted Classifier.

Args:
    X (array-like): Training input features of shape (n_samples, n_features).
    y (array-like): Training target class labels of shape (n_samples,).
    n_estimators (int): Number of boosting stages (trees).
    learning_rate (float): Step size shrinkage to prevent overfitting.
    max_depth (int): Maximum depth of each individual regression tree estimator.
    min_samples_split (int): Minimum samples required to split a node in a tree.
    random_seed (int, optional): Seed for reproducibility. Defaults to None.

<a id="trees.gradientBoostedClassifier.GradientBoostedClassifier.get_params"></a>

#### get\_params

```python
def get_params()
```

Get the parameters of the GradientBoostedClassifier.

<a id="trees.gradientBoostedClassifier.GradientBoostedClassifier._validate_input"></a>

#### \_validate\_input

```python
def _validate_input(X, y)
```

Validates input data X and y.

<a id="trees.gradientBoostedClassifier.GradientBoostedClassifier._init_predict"></a>

#### \_init\_predict

```python
def _init_predict(y)
```

Calculate the initial prediction (log-odds).

<a id="trees.gradientBoostedClassifier.GradientBoostedClassifier.fit"></a>

#### fit

```python
def fit(X=None, y=None, sample_weight=None, verbose=0)
```

Fits the gradient boosted classifier to the training data.

Args:
    X (array-like): Training input features of shape (n_samples, n_features).
    y (array-like): Training target class labels of shape (n_samples,).
    sample_weight (array-like, optional): Sample weights for the training data.
    verbose (int): Controls the verbosity of the fitting process.
                   0 for no output, 1 for basic output.

Returns:
    self: The fitted GradientBoostedClassifier instance.

<a id="trees.gradientBoostedClassifier.GradientBoostedClassifier.decision_function"></a>

#### decision\_function

```python
def decision_function(X)
```

Compute the raw decision scores (log-odds) for samples in X.

Args:
    X (array-like): Input features of shape (n_samples, n_features).

Returns:
    np.ndarray: The raw decision scores. Shape (n_samples,) for binary
                or (n_samples, n_classes) for multi-class.

<a id="trees.gradientBoostedClassifier.GradientBoostedClassifier.predict_proba"></a>

#### predict\_proba

```python
def predict_proba(X)
```

Predict class probabilities for samples in X.

Args:
    X (array-like): Input features of shape (n_samples, n_features).

Returns:
    np.ndarray: Predicted class probabilities. Shape (n_samples, n_classes).
                For binary, columns are [P(class 0), P(class 1)].

<a id="trees.gradientBoostedClassifier.GradientBoostedClassifier.predict"></a>

#### predict

```python
def predict(X)
```

Predicts class labels for input features X.

Args:
    X (array-like): Input features of shape (n_samples, n_features).

Returns:
    np.ndarray: Predicted class labels of shape (n_samples,).

<a id="trees.gradientBoostedClassifier.GradientBoostedClassifier.calculate_metrics"></a>

#### calculate\_metrics

```python
def calculate_metrics(y_true, y_pred, y_prob=None)
```

Calculate common classification metrics.

Args:
    y_true (array-like): True class labels.
    y_pred (array-like): Predicted class labels.
    y_prob (array-like, optional): Predicted probabilities for Log Loss calculation.

Returns:
    dict: A dictionary containing calculated metrics (Accuracy, Precision, Recall, F1 Score, Log Loss if applicable).

<a id="trees.gradientBoostedClassifier.GradientBoostedClassifier.get_stats"></a>

#### get\_stats

```python
def get_stats(y_true, X=None, y_pred=None, verbose=False)
```

Calculate and optionally print evaluation metrics. Requires either X or y_pred.

Args:
    y_true (array-like): True target values.
    X (array-like, optional): Input features to generate predictions if y_pred is not provided.
    y_pred (array-like, optional): Pre-computed predicted class labels.
    verbose (bool): Whether to print the metrics.

Returns:
    dict: A dictionary containing calculated metrics.

<a id="trees.gradientBoostedRegressor"></a>

# trees.gradientBoostedRegressor

<a id="trees.gradientBoostedRegressor.GradientBoostedRegressor"></a>

## GradientBoostedRegressor Objects

```python
class GradientBoostedRegressor()
```

A class to represent a Gradient Boosted Decision Tree Regressor.

Attributes:
    random_seed (int): The random seed for the random number generator.
    num_trees (int): The number of decision trees in the ensemble.
    max_depth (int): The maximum depth of each decision tree.
    learning_rate (float): The learning rate for the gradient boosted model.
    min_samples_split (int): The minimum number of samples required to split a node.
    random_seed (int): The random seed for the random number generator.

Methods:
    fit(X=None, y=None, verbose=0): Fits the gradient boosted decision tree regressor to the training data.
    predict(X): Predicts the target values for the input features.
    calculate_metrics(y_true, y_pred): Calculates the evaluation metrics.
    get_stats(y_true, y_pred, verbose=False): Returns the evaluation metrics.

<a id="trees.gradientBoostedRegressor.GradientBoostedRegressor.__init__"></a>

#### \_\_init\_\_

```python
def __init__(X=None,
             y=None,
             num_trees: int = 100,
             max_depth: int = 3,
             learning_rate: float = 0.1,
             min_samples_split: int = 2,
             random_seed: int = None)
```

Initializes the Gradient Boosted Decision Tree Regressor.

Args:
    X: (np.ndarray), optional - Input feature data (default is None).
    y: (np.ndarray), optional - Target data (default is None).
    num_trees (int): Number of boosting stages (trees).
    max_depth (int): Maximum depth of each individual tree regressor.
    learning_rate (float): Step size shrinkage to prevent overfitting.
    min_samples_split (int): Minimum samples required to split a node.
    random_seed (int): Seed for reproducibility (currently affects feature selection within trees).

<a id="trees.gradientBoostedRegressor.GradientBoostedRegressor.get_params"></a>

#### get\_params

```python
def get_params()
```

Get the parameters of the GradientBoostedRegressor.

<a id="trees.gradientBoostedRegressor.GradientBoostedRegressor.fit"></a>

#### fit

```python
def fit(X=None, y=None, sample_weight=None, verbose=0)
```

Fits the gradient boosted decision tree regressor to the training data.

This method trains the ensemble of decision trees by iteratively fitting each tree to the residuals
of the previous iteration. The residuals are updated after each iteration by subtracting the predictions
made by the current tree from the :target values.

Args:
    X (array-like): Training input features of shape (n_samples, n_features).
    y (array-like): Training target values of shape (n_samples,).
    sample_weight (array-like): Sample weights for each instance (not used in this implementation).
    verbose (int): Whether to print progress messages (e.g., residuals). 0 for no output, 1 for output, >1 for detailed output

Returns:
    self: The fitted GradientBoostedRegressor instance.

<a id="trees.gradientBoostedRegressor.GradientBoostedRegressor.predict"></a>

#### predict

```python
def predict(X)
```

Predicts target values for input features X using the fitted GBR model.

Args:
    X (array-like): Input features of shape (n_samples, n_features).

Returns:
    np.ndarray: Predicted target values of shape (n_samples,).

<a id="trees.gradientBoostedRegressor.GradientBoostedRegressor.calculate_metrics"></a>

#### calculate\_metrics

```python
def calculate_metrics(y_true, y_pred)
```

Calculate common regression metrics.

Args:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.

Returns:
    dict: A dictionary containing calculated metrics (MSE, R^2, MAE, RMSE, MAPE).

<a id="trees.gradientBoostedRegressor.GradientBoostedRegressor.get_stats"></a>

#### get\_stats

```python
def get_stats(y_true, y_pred, verbose=False)
```

Calculate and optionally print evaluation metrics.

Args:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.
    verbose (bool): Whether to print progress messages (e.g., residuals).

Returns:
    dict: A dictionary containing calculated metrics (MSE, R^2, MAE, RMSE, MAPE).

<a id="trees.isolationForest"></a>

# trees.isolationForest

<a id="trees.isolationForest.IsolationUtils"></a>

## IsolationUtils Objects

```python
class IsolationUtils()
```

Utility functions for the Isolation Forest algorithm.

<a id="trees.isolationForest.IsolationUtils.compute_avg_path_length"></a>

#### compute\_avg\_path\_length

```python
@staticmethod
def compute_avg_path_length(size)
```

Computes the average path length of unsuccessful searches in a binary search tree.

Args:
    size: (int) - The size of the tree.

Returns:
    average_path_length: (float) - The average path length.

<a id="trees.isolationForest.IsolationTree"></a>

## IsolationTree Objects

```python
class IsolationTree()
```

IsolationTree is a class that implements an isolation tree, which is a fundamental building block of the Isolation Forest algorithm.

The Isolation Forest is an unsupervised learning method used for anomaly detection.

Attributes:
    max_depth (int): The maximum depth of the tree. Default is 10.
    tree (dict): The learned isolation tree structure.
    force_true_length (bool): If True, the true path length is used for scoring
        instead of the average path length.

Methods:
    __init__(max_depth=10, force_true_length=False):
        Initializes the IsolationTree with the specified maximum depth and
        scoring method.
    fit(X, depth=0):
        Fits the isolation tree to the input data by recursively partitioning
        the data based on randomly selected features and split values.
    path_length(X, tree=None, depth=0):
        Computes the path length for a given sample by traversing the tree
        structure. The path length is used to determine how isolated a sample is.

<a id="trees.isolationForest.IsolationTree.__init__"></a>

#### \_\_init\_\_

```python
def __init__(max_depth=10, force_true_length=False)
```

Initializes the Isolation Forest with specified parameters.

Args:
    max_depth: (int), optional - Maximum depth of the tree (default is 10).
    force_true_length: (bool), optional - If True, use the true path length for scoring (default is False).

Attributes:
    max_depth: (int) - Maximum depth of the tree.
    tree: (object or None) - The tree structure used in the Isolation Forest (default is None).
    force_true_length: (bool) - Indicates whether to use the true path length for scoring.

<a id="trees.isolationForest.IsolationTree.fit"></a>

#### fit

```python
def fit(X, depth=0)
```

Fits the isolation tree to the data.

Args:
    X: (array-like) - The input features.
    depth: (int) - The current depth of the tree (default: 0).

Returns:
    dict: The learned isolation tree.

<a id="trees.isolationForest.IsolationTree.path_length"></a>

#### path\_length

```python
def path_length(X, tree=None, depth=0)
```

Computes the path length for a given sample.

Args:
    X: (array-like) - The input sample.
    tree: (dict) - The current node of the tree (default: None).
    depth: (int) - The current depth of the tree (default: 0).

Returns:
    int: The path length.

<a id="trees.isolationForest.IsolationForest"></a>

## IsolationForest Objects

```python
class IsolationForest()
```

IsolationForest is an implementation of the Isolation Forest algorithm for anomaly detection.

Attributes:
    n_trees (int): The number of isolation trees to build. Default is 100.
    max_samples (int or None): The maximum number of samples to draw for each tree. If None, defaults to the minimum of 256 or the number of samples in the dataset.
    max_depth (int): The maximum depth of each isolation tree. Default is 10.
    n_jobs (int): The number of parallel jobs to run. Set to -1 to use all available cores. Default is 1.
    force_true_length (bool): Whether to force the true path length calculation. Default is False.
    trees (list): A list to store the trained isolation trees.
    classes_ (numpy.ndarray): An array representing the classes (0 for normal, 1 for anomaly).

Methods:
    __init__(n_trees=100, max_samples=None, max_depth=10, n_jobs=1, force_true_length=False):
        Initializes the IsolationForest with the specified parameters.
    fit(X):
        Fits the isolation forest to the data.
            X (array-like): The input features.
    _fit_tree(X):
        Fits a single isolation tree to a subset of the data.
            X (array-like): The input features.
            IsolationTree: A trained isolation tree.
    anomaly_score(X):
        Computes the anomaly scores for given samples.
            X (array-like): The input samples.
            numpy.ndarray: An array of anomaly scores.
    predict(X, threshold=0.5):
        Predicts whether samples are anomalies.
            X (array-like): The input samples.
            threshold (float): The threshold for classifying anomalies (default: 0.5).
            numpy.ndarray: An array of predictions (1 if the sample is an anomaly, 0 otherwise).
    __sklearn_is_fitted__():
        Checks if the model has been fitted.
            bool: True if the model is fitted, False otherwise.

<a id="trees.isolationForest.IsolationForest.__init__"></a>

#### \_\_init\_\_

```python
def __init__(n_trees=100,
             max_samples=None,
             max_depth=10,
             n_jobs=1,
             force_true_length=False)
```

Initializes the IsolationForest with the specified parameters.

Args:
    n_trees: (int), optional - The number of isolation trees to build (default: 100).
    max_samples: (int or None), optional - The maximum number of samples to draw for each tree.
        If None, defaults to the minimum of 256 or the number of samples in the dataset (default: None).
    max_depth: (int), optional - The maximum depth of each isolation tree (default: 10).
    n_jobs: (int), optional - The number of parallel jobs to run.
        Set to -1 to use all available cores (default: 1).
    force_true_length: (bool), optional - Whether to force the true path length calculation (default: False).

Attributes:
    n_trees: (int) - The number of isolation trees.
    max_samples: (int or None) - The maximum number of samples for each tree.
    max_depth: (int) - The maximum depth of the trees.
    force_true_length: (bool) - Indicates whether to use the true path length for scoring.
    trees: (list) - A list to store the trained isolation trees.
    n_jobs: (int) - The number of parallel jobs to run.
    classes_: (np.ndarray) - An array representing the classes (0 for normal, 1 for anomaly).

<a id="trees.isolationForest.IsolationForest.fit"></a>

#### fit

```python
def fit(X, y=None)
```

Fits the isolation forest to the data.

Args:
    X: (array-like) - The input features.
    y: (array-like) - The target labels (not used in this implementation).

<a id="trees.isolationForest.IsolationForest._fit_tree"></a>

#### \_fit\_tree

```python
def _fit_tree(X)
```

<a id="trees.isolationForest.IsolationForest.anomaly_score"></a>

#### anomaly\_score

```python
def anomaly_score(X)
```

Computes the anomaly scores for given samples.

Args:
    X: (array-like) - The input samples.

Returns:
    array: An array of anomaly scores.

<a id="trees.isolationForest.IsolationForest.predict"></a>

#### predict

```python
def predict(X, threshold=0.5)
```

Predicts whether samples are anomalies.

Args:
    X: (array-like) - The input samples.
    threshold: (float) - The threshold for classifying anomalies (default: 0.5).

Returns:
    array: An array of predictions (1 if the sample is an anomaly, 0 otherwise).

<a id="trees.isolationForest.IsolationForest.__sklearn_is_fitted__"></a>

#### \_\_sklearn\_is\_fitted\_\_

```python
def __sklearn_is_fitted__()
```

Checks if the model has been fitted.

<a id="trees.randomForestClassifier"></a>

# trees.randomForestClassifier

This module contains the implementation of a Random Forest Classifier.

The module includes the following classes:
- RandomForest: A class representing a Random Forest model.
- RandomForestWithInfoGain: A class representing a Random Forest model that returns information gain for vis.
- runRandomForest: A class that runs the Random Forest algorithm.

<a id="trees.randomForestClassifier._fit_tree"></a>

#### \_fit\_tree

```python
def _fit_tree(X, y, max_depth, min_samples_split, sample_weight=None)
```

Helper function for parallel tree fitting. Fits a single tree on a bootstrapped sample.

Args:
    X: (array-like) - The input features.
    y: (array-like) - The target labels.
    max_depth: (int) - The maximum depth of the tree.
    min_samples_split: (int) - The minimum samples required to split a node.
    sample_weight: (array-like or None) - The weights for each sample.

Returns:
    ClassifierTree: A fitted tree object.

<a id="trees.randomForestClassifier._classify_oob"></a>

#### \_classify\_oob

```python
def _classify_oob(X, trees, bootstraps)
```

Helper function for parallel out-of-bag predictions. Classifies using out-of-bag samples.

Args:
    X: (array-like) - The input features.
    trees: (list) - The list of fitted trees.
    bootstraps: (list) - The list of bootstrapped indices for each tree.

Returns:
    list: The list of out-of-bag predictions.

<a id="trees.randomForestClassifier.RandomForestClassifier"></a>

## RandomForestClassifier Objects

```python
class RandomForestClassifier()
```

RandomForestClassifier is a custom implementation of a Random Forest classifier.

Attributes:
    n_estimators (int): The number of trees in the forest.
    max_depth (int): The maximum depth of each tree.
    n_jobs (int): The number of jobs to run in parallel. Defaults to -1 (use all available processors).
    random_state (int or None): The seed for random number generation. Defaults to None.
    trees (list): A list of trained decision trees.
    bootstraps (list): A list of bootstrapped indices for out-of-bag (OOB) scoring.
    X (numpy.ndarray or None): The feature matrix used for training.
    y (numpy.ndarray or None): The target labels used for training.
    accuracy (float): The accuracy of the model after fitting.
    precision (float): The precision of the model after fitting.
    recall (float): The recall of the model after fitting.
    f1_score (float): The F1 score of the model after fitting.
    log_loss (float or None): The log loss of the model after fitting (only for binary classification).

Methods:
    __init__(forest_size=100, max_depth=10, n_jobs=-1, random_seed=None, X=None, y=None):
        Initializes the RandomForestClassifier object with the specified parameters.
    fit(X=None, y=None, verbose=False):
        Fits the random forest model to the provided data using parallel processing.
    calculate_metrics(y_true, y_pred):
        Calculates evaluation metrics (accuracy, precision, recall, F1 score, and log loss) for classification.
    predict(X):
        Predicts class labels for the provided data using the trained random forest.
    get_stats(verbose=False):
        Returns the evaluation metrics (accuracy, precision, recall, F1 score, and log loss) as a dictionary.

<a id="trees.randomForestClassifier.RandomForestClassifier.__init__"></a>

#### \_\_init\_\_

```python
def __init__(forest_size=100,
             max_depth=10,
             min_samples_split=2,
             n_jobs=-1,
             random_seed=None,
             X=None,
             y=None)
```

Initializes the RandomForest object.

<a id="trees.randomForestClassifier.RandomForestClassifier.get_params"></a>

#### get\_params

```python
def get_params()
```

Get the parameters of the RandomForestClassifier.

<a id="trees.randomForestClassifier.RandomForestClassifier.fit"></a>

#### fit

```python
def fit(X=None, y=None, sample_weight=None, verbose=False)
```

Fit the random forest with parallel processing.

<a id="trees.randomForestClassifier.RandomForestClassifier.calculate_metrics"></a>

#### calculate\_metrics

```python
def calculate_metrics(y_true, y_pred)
```

Calculate evaluation metrics for classification.

<a id="trees.randomForestClassifier.RandomForestClassifier.predict"></a>

#### predict

```python
def predict(X)
```

Predict class labels for the provided data.

<a id="trees.randomForestClassifier.RandomForestClassifier.predict_proba"></a>

#### predict\_proba

```python
def predict_proba(X)
```

Predict class probabilities for the provided data.

Args:
    X (array-like): The input features.

Returns:
    np.ndarray: A 2D array where each row represents the probability distribution
                over the classes for a record.

<a id="trees.randomForestClassifier.RandomForestClassifier.get_stats"></a>

#### get\_stats

```python
def get_stats(verbose=False)
```

Return the evaluation metrics.

<a id="trees.randomForestRegressor"></a>

# trees.randomForestRegressor

<a id="trees.randomForestRegressor._fit_single_tree"></a>

#### \_fit\_single\_tree

```python
def _fit_single_tree(X, y, max_depth, min_samples_split, sample_weight,
                     tree_index, random_state_base, verbose)
```

Helper function for parallel tree fitting. Fits a single tree on abootstrapped sample.

Args:
    X (np.ndarray): The input features.
    y (np.ndarray): The target labels.
    max_depth (int): The maximum depth of the tree.
    min_samples_split (int): The minimum samples required to split a node.
    sample_weight (array-like): Sample weights for each instance in X.
    tree_index (int): Index of the tree for seeding.
    random_state_base (int): Base random seed.
    verbose (bool): If True, print detailed logs during fitting.

Returns:
    tuple: (tree_index, fitted_tree_instance, bootstrap_indices)

<a id="trees.randomForestRegressor.RandomForestRegressor"></a>

## RandomForestRegressor Objects

```python
class RandomForestRegressor()
```

A class representing a Random Forest model for regression.

Attributes:
    n_estimators (int): The number of trees in the forest.
    max_depth (int): The maximum depth of each tree.
    min_samples_split (int): The minimum number of samples required to split an internal node.
    n_jobs (int): The number of jobs to run in parallel for fitting.
    random_state (int): Seed for random number generation for reproducibility.
    trees (list): List holding the fitted RegressorTree instances.
    X (numpy.ndarray or None): The feature matrix used for training.
    y (numpy.ndarray or None): The target labels used for training.

Methods:
    fit(X=None, y=None, verbose=False): Fits the random forest to the data.
    calculate_metrics(y_true, y_pred): Calculates the evaluation metrics.
    predict(X): Predicts the target values for the input features.
    get_stats(verbose=False): Returns the evaluation metrics.

<a id="trees.randomForestRegressor.RandomForestRegressor.__init__"></a>

#### \_\_init\_\_

```python
def __init__(forest_size=100,
             max_depth=10,
             min_samples_split=2,
             n_jobs=-1,
             random_seed=None,
             X=None,
             y=None)
```

Initialize the Random Forest Regressor.

<a id="trees.randomForestRegressor.RandomForestRegressor.get_params"></a>

#### get\_params

```python
def get_params()
```

Get the parameters of the Random Forest Regressor.

Returns:
    dict: A dictionary containing the parameters of the model.

<a id="trees.randomForestRegressor.RandomForestRegressor.fit"></a>

#### fit

```python
def fit(X=None, y=None, sample_weight=None, verbose=False)
```

Fit the random forest to the training data X and y.

Args:
    X (array-like): Training input features of shape (n_samples, n_features).
    y (array-like): Training target values of shape (n_samples,).
    sample_weight (array-like): Sample weights for each instance in X.
    verbose (bool): Whether to print progress messages.

Returns:
    self: The fitted RandomForestRegressor instance.

<a id="trees.randomForestRegressor.RandomForestRegressor.predict"></a>

#### predict

```python
def predict(X)
```

Predict target values for input features X using the trained random forest.

Args:
    X (array-like): Input features of shape (n_samples, n_features).

Returns:
    np.ndarray: Predicted target values of shape (n_samples,).

<a id="trees.randomForestRegressor.RandomForestRegressor.get_stats"></a>

#### get\_stats

```python
def get_stats(y_true, y_pred, verbose=False)
```

Calculate and optionally print evaluation metrics.

Args:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.
    verbose (bool): Whether to print progress messages (e.g., residuals).

Returns:
    dict: A dictionary containing calculated metrics (MSE, R^2, MAE, RMSE, MAPE).

<a id="trees.randomForestRegressor.RandomForestRegressor.calculate_metrics"></a>

#### calculate\_metrics

```python
def calculate_metrics(y_true, y_pred)
```

Calculate common regression metrics.

Args:
    y_true (array-like): True target values.
    y_pred (array-like): Predicted target values.

Returns:
    dict: A dictionary containing calculated metrics (MSE, R^2, MAE, RMSE, MAPE).

<a id="trees.treeClassifier"></a>

# trees.treeClassifier

<a id="trees.treeClassifier.ClassifierTreeUtility"></a>

## ClassifierTreeUtility Objects

```python
class ClassifierTreeUtility()
```

Utility class for computing entropy, partitioning classes, and calculating information gain.

<a id="trees.treeClassifier.ClassifierTreeUtility.__init__"></a>

#### \_\_init\_\_

```python
def __init__(min_samples_split=2)
```

Initialize the utility class.

<a id="trees.treeClassifier.ClassifierTreeUtility.entropy"></a>

#### entropy

```python
def entropy(class_y, sample_weight=None)
```

Computes the entropy for a given class.

Args:
    class_y: (array-like) - The class labels.
    sample_weight: (array-like) - The sample weights (default: None).

Returns:
    float: The entropy value.

<a id="trees.treeClassifier.ClassifierTreeUtility.partition_classes"></a>

#### partition\_classes

```python
def partition_classes(X, y, split_attribute, split_val, sample_weight=None)
```

Partitions the dataset into two subsets based on a given split attribute and value.

Args:
    X: (array-like) - The input features.
    y: (array-like) - The target labels.
    split_attribute: (int) - The index of the attribute to split on.
    split_val: (float) - The value to split the attribute on.
    sample_weight: (array-like) - The sample weights (default: None).

Returns:
    X_left:  (array-like) - The subset of input features where the split attribute is less than or equal to the split value.
    X_right: (array-like) - The subset of input features where the split attribute is greater than the split value.
    y_left:  (array-like) - The subset of target labels corresponding to X_left.
    y_right: (array-like) - The subset of target labels corresponding to X_right.

<a id="trees.treeClassifier.ClassifierTreeUtility.information_gain"></a>

#### information\_gain

```python
def information_gain(previous_y,
                     current_y,
                     sample_weight_prev=None,
                     sample_weight_current=None)
```

Calculates the information gain between the previous and current values of y.

Args:
    previous_y: (array-like) - The previous values of y.
    current_y: (array-like) - The current values of y.
    sample_weight_prev: (array-like) - The sample weights for the previous y values (default: None).
    sample_weight_current: (array-like) - The sample weights for the current y values (default: None).

Returns:
    float: The information gain between the previous and current values of y.

<a id="trees.treeClassifier.ClassifierTreeUtility.best_split"></a>

#### best\_split

```python
def best_split(X, y, sample_weight=None)
```

Finds the best attribute and value to split the data based on information gain.

Args:
    X: (array-like) - The input features.
    y: (array-like) - The target variable.
    sample_weight: (array-like) - The sample weights (default: None).

Returns:
    dict: A dictionary containing the best split attribute, split value, left and right subsets of X and y,
          and the information gain achieved by the split.

<a id="trees.treeClassifier.ClassifierTree"></a>

## ClassifierTree Objects

```python
class ClassifierTree()
```

A class representing a decision tree.

Args:
    max_depth: (int) - The maximum depth of the decision tree.

Methods:
    learn(X, y, par_node={}, depth=0): Builds the decision tree based on the given training data.
    classify(record): Classifies a record using the decision tree.

<a id="trees.treeClassifier.ClassifierTree.__init__"></a>

#### \_\_init\_\_

```python
def __init__(max_depth=5, min_samples_split=2)
```

Initializes the ClassifierTree with a maximum depth.

<a id="trees.treeClassifier.ClassifierTree.fit"></a>

#### fit

```python
def fit(X, y, sample_weight=None)
```

Fits the decision tree to the training data.

Args:
    X: (array-like) - The input features.
    y: (array-like) - The target labels.
    sample_weight: (array-like) - The sample weights (default: None).

<a id="trees.treeClassifier.ClassifierTree.learn"></a>

#### learn

```python
def learn(X, y, par_node=None, depth=0, sample_weight=None)
```

Builds the decision tree based on the given training data.

Args:
    X: (array-like) - The input features.
    y: (array-like) - The target labels.
    par_node: (dict) - The parent node of the current subtree (default: {}).
    depth: (int) - The current depth of the subtree (default: 0).
    sample_weight: (array-like) - The sample weights (default: None).

Returns:
    dict: The learned decision tree.

<a id="trees.treeClassifier.ClassifierTree.classify"></a>

#### classify

```python
@staticmethod
def classify(tree, record)
```

Classifies a given record using the decision tree.

Args:
    tree: (dict) - The decision tree.
    record: (dict) - A dictionary representing the record to be classified.

Returns:
    The label assigned to the record based on the decision tree.

<a id="trees.treeClassifier.ClassifierTree.predict"></a>

#### predict

```python
def predict(X)
```

Predicts the labels for a given set of records using the decision tree.

Args:
    X: (array-like) - The input features.

Returns:
    list: A list of predicted labels for each record.

<a id="trees.treeClassifier.ClassifierTree.predict_proba"></a>

#### predict\_proba

```python
def predict_proba(X)
```

Predicts the probabilities for a given set of records using the decision tree.

Args:
    X: (array-like) - The input features.

Returns:
    list: A list of dictionaries where each dictionary represents the probability distribution
          over the classes for a record.

<a id="trees.treeRegressor"></a>

# trees.treeRegressor

<a id="trees.treeRegressor.RegressorTreeUtility"></a>

## RegressorTreeUtility Objects

```python
class RegressorTreeUtility()
```

Utility class containing helper functions for building the Regressor Tree.

Handles variance calculation, leaf value calculation, and finding the best split.

<a id="trees.treeRegressor.RegressorTreeUtility.__init__"></a>

#### \_\_init\_\_

```python
def __init__(X, y, min_samples_split, n_features)
```

Initialize the utility class with references to data and parameters.

Args:
    X (np.ndarray): Reference to the feature data.
    y (np.ndarray): Reference to the target data.
    min_samples_split (int): Minimum number of samples required to split a node.
    n_features (int): Total number of features in X.

<a id="trees.treeRegressor.RegressorTreeUtility.calculate_variance"></a>

#### calculate\_variance

```python
def calculate_variance(indices, sample_weight=None)
```

Calculate weighted variance for the subset defined by indices.

<a id="trees.treeRegressor.RegressorTreeUtility.calculate_leaf_value"></a>

#### calculate\_leaf\_value

```python
def calculate_leaf_value(indices, sample_weight=None)
```

Calculate the weighted mean value for a leaf node.

<a id="trees.treeRegressor.RegressorTreeUtility.best_split"></a>

#### best\_split

```python
def best_split(indices, sample_weight=None)
```

Finds the best split for the data subset defined by indices.

<a id="trees.treeRegressor.RegressorTree"></a>

## RegressorTree Objects

```python
class RegressorTree()
```

A class representing a decision tree for regression.

Args:
    max_depth: (int) - The maximum depth of the decision tree.
    min_samples_split: (int) - The minimum number of samples required to split a node.
    n_features: (int) - The number of features in the dataset.
    X: (array-like) - The input features.
    y: (array-like) - The target labels.

Methods:
    fit(X, y, verbose=False): Fits the decision tree to the training data.
    predict(X): Predicts the target values for the input features.
    _traverse_tree(x, node): Traverses the decision tree for a single sample x.
    _leran_recursive(indices, depth): Recursive helper function for learning.

<a id="trees.treeRegressor.RegressorTree.__init__"></a>

#### \_\_init\_\_

```python
def __init__(max_depth=5, min_samples_split=2)
```

Initialize the decision tree.

Args:
    max_depth (int): The maximum depth of the decision tree.
    min_samples_split (int): The minimum number of samples required to split a node.

<a id="trees.treeRegressor.RegressorTree.fit"></a>

#### fit

```python
def fit(X, y, sample_weight=None, verbose=False)
```

Fit the decision tree to the training data.

Args:
    X: (array-like) - The input features.
    y: (array-like) - The target labels.
    sample_weight: (array-like) - The sample weights (default: None).
    verbose: (bool) - If True, print detailed logs during fitting.

Returns:
    dict: The learned decision tree.

<a id="trees.treeRegressor.RegressorTree.predict"></a>

#### predict

```python
def predict(X)
```

Predict the target value for a record or batch of records using the decision tree.

Args:
    X: (array-like) - The input features.

Returns:
    np.ndarray: The predicted target values.

<a id="trees.treeRegressor.RegressorTree._traverse_tree"></a>

#### \_traverse\_tree

```python
def _traverse_tree(x, node)
```

Traverse the tree for a single sample x.

Args:
    x (array-like): The input features.
    node (dict): The current node in the decision tree.

<a id="trees.treeRegressor.RegressorTree._learn_recursive"></a>

#### \_learn\_recursive

```python
def _learn_recursive(indices, depth, sample_weight)
```

Recursive helper function for learning.

Args:
    indices (array-like): The indices of the current node.
    depth (int): The current depth of the decision tree.
    sample_weight (array-like): The sample weights for the current node.

<a id="utils"></a>

# utils

<a id="utils.__all__"></a>

#### \_\_all\_\_

<a id="utils.animator"></a>

# utils.animator

<a id="utils.animator.AnimationBase"></a>

## AnimationBase Objects

```python
class AnimationBase(ABC)
```

Base class for creating animations of machine learning models.

<a id="utils.animator.AnimationBase.__init__"></a>

#### \_\_init\_\_

```python
def __init__(model,
             train_series,
             test_series,
             dynamic_parameter=None,
             static_parameters=None,
             keep_previous=None,
             **kwargs)
```

Initialize the animation base class.

Args:
    model: The forecasting model or any machine learning model.
    train_series: Training time series data.
    test_series: Testing time series data.
    dynamic_parameter: The parameter to update dynamically (e.g., 'window', 'alpha', 'beta').
    static_parameters: Static parameters for the model.
        Should be a dictionary with parameter names as keys and their values.
    keep_previous: Whether to keep all previous lines with reduced opacity.
    **kwargs: Additional customization options (e.g., colors, line styles).

<a id="utils.animator.AnimationBase.setup_plot"></a>

#### setup\_plot

```python
def setup_plot(title,
               xlabel,
               ylabel,
               legend_loc="upper left",
               grid=True,
               figsize=(12, 6))
```

Set up the plot for the animation.

Args:
    title: Title of the plot.
    xlabel: Label for the x-axis.
    ylabel: Label for the y-axis.
    legend_loc: Location of the legend.
    grid: Whether to show grid lines.
    figsize: Size of the figure.

<a id="utils.animator.AnimationBase.update_model"></a>

#### update\_model

```python
@abstractmethod
def update_model(frame)
```

Abstract method to update the model for a given frame. Must be implemented by subclasses.

<a id="utils.animator.AnimationBase.update_plot"></a>

#### update\_plot

```python
@abstractmethod
def update_plot(frame)
```

Abstract method to update the plot for a given frame.Must be implemented by subclasses.

<a id="utils.animator.AnimationBase.animate"></a>

#### animate

```python
def animate(frames, interval=150, blit=True, repeat=False)
```

Create the animation.

Args:
    frames: Range of frames (e.g., window sizes).
    interval: Delay between frames in milliseconds.
    blit: Whether to use blitting for faster rendering.
    repeat: Whether to repeat the animation.

<a id="utils.animator.AnimationBase.save"></a>

#### save

```python
def save(filename, writer="pillow", fps=5, dpi=100)
```

Save the animation to a file.

Args:
    filename: Path to save the animation.
    writer: Writer to use (e.g., 'pillow' for GIF).
    fps: Frames per second.
    dpi: Dots per inch for the saved figure.

<a id="utils.animator.AnimationBase.show"></a>

#### show

```python
def show()
```

Display the animation.

<a id="utils.animator.ForcastingAnimation"></a>

## ForcastingAnimation Objects

```python
class ForcastingAnimation(AnimationBase)
```

Class for creating animations of forecasting models.

<a id="utils.animator.ForcastingAnimation.__init__"></a>

#### \_\_init\_\_

```python
def __init__(model,
             train_series,
             test_series,
             forecast_steps,
             dynamic_parameter=None,
             static_parameters=None,
             keep_previous=False,
             max_previous=None,
             **kwargs)
```

Initialize the forecasting animation class.

Args:
    model: The forecasting model.
    train_series: Training time series data.
    test_series: Testing time series data.
    forecast_steps: Number of steps to forecast.
    dynamic_parameter: The parameter to update dynamically (e.g., 'window', 'alpha', 'beta').
    static_parameters: Static parameters for the model.
        Should be a dictionary with parameter names as keys and their values.
    keep_previous: Whether to keep all previous lines with reduced opacity.
    max_previous: Maximum number of previous lines to keep.
    **kwargs: Additional customization options (e.g., colors, line styles).

<a id="utils.animator.ForcastingAnimation.setup_plot"></a>

#### setup\_plot

```python
def setup_plot(title,
               xlabel,
               ylabel,
               legend_loc="upper left",
               grid=True,
               figsize=(12, 6))
```

Set up the plot for forecasting animation.

<a id="utils.animator.ForcastingAnimation.update_model"></a>

#### update\_model

```python
def update_model(frame)
```

Update the model for the current frame.

Args:
    frame: The current frame (e.g., parameter value).

<a id="utils.animator.ForcastingAnimation.update_plot"></a>

#### update\_plot

```python
def update_plot(frame)
```

Update the plot for the current frame.

Args:
    frame: The current frame (e.g., parameter value).

<a id="utils.animator.RegressionAnimation"></a>

## RegressionAnimation Objects

```python
class RegressionAnimation(AnimationBase)
```

Class for creating animations of regression models.

<a id="utils.animator.RegressionAnimation.__init__"></a>

#### \_\_init\_\_

```python
def __init__(model,
             X,
             y,
             test_size=0.3,
             dynamic_parameter=None,
             static_parameters=None,
             keep_previous=False,
             max_previous=None,
             pca_components=1,
             **kwargs)
```

Initialize the regression animation class.

Args:
    model: The regression model.
    X: Feature matrix (input data).
    y: Target vector (output data).
    test_size: Proportion of the dataset to include in the test split.
    dynamic_parameter: The parameter to update dynamically (e.g., 'alpha', 'beta').
    static_parameters: Additional static parameters for the model.
        Should be a dictionary with parameter names as keys and their values.
    keep_previous: Whether to keep all previous lines with reduced opacity.
    max_previous: Maximum number of previous lines to keep.
    pca_components: Number of components to use for PCA.
    **kwargs: Additional customization options (e.g., colors, line styles).

<a id="utils.animator.RegressionAnimation.setup_plot"></a>

#### setup\_plot

```python
def setup_plot(title,
               xlabel,
               ylabel,
               legend_loc="upper left",
               grid=True,
               figsize=(12, 6))
```

Set up the plot for regression animation.

<a id="utils.animator.RegressionAnimation.update_model"></a>

#### update\_model

```python
def update_model(frame)
```

Update the regression model for the current frame.

Args:
    frame: The current frame (e.g., parameter value).

<a id="utils.animator.RegressionAnimation.update_plot"></a>

#### update\_plot

```python
def update_plot(frame)
```

Update the plot for the current frame.

Args:
    frame: The current frame (e.g., parameter value).

<a id="utils.animator.ClassificationAnimation"></a>

## ClassificationAnimation Objects

```python
class ClassificationAnimation(AnimationBase)
```

Class for creating animations of classification models.

<a id="utils.animator.ClassificationAnimation.__init__"></a>

#### \_\_init\_\_

```python
def __init__(model,
             X,
             y,
             test_size=0.3,
             dynamic_parameter=None,
             static_parameters=None,
             keep_previous=False,
             scaler=None,
             pca_components=2,
             plot_step=0.02,
             **kwargs)
```

Initialize the classification animation class.

Args:
    model: The classification model.
    X: Feature matrix (input data).
    y: Target vector (output data).
    test_size: Proportion of the dataset to include in the test split.
    dynamic_parameter: The parameter to update dynamically (e.g., 'alpha', 'beta').
    static_parameters: Additional static parameters for the model.
        Should be a dictionary with parameter names as keys and their values.
    keep_previous: Whether to keep all previous lines with reduced opacity.
    scaler: Optional scaler for preprocessing the data.
    pca_components: Number of components to use for PCA.
    plot_step: Resolution of the decision boundary mesh.
    **kwargs: Additional customization options (e.g., colors, line styles).

<a id="utils.animator.ClassificationAnimation.setup_plot"></a>

#### setup\_plot

```python
def setup_plot(title,
               xlabel,
               ylabel,
               legend_loc="upper left",
               grid=True,
               figsize=(12, 6))
```

Set up the plot for classification animation.

<a id="utils.animator.ClassificationAnimation.update_model"></a>

#### update\_model

```python
def update_model(frame)
```

Update the classification model for the current frame.

Args:
    frame: The current frame (e.g., parameter value).

<a id="utils.animator.ClassificationAnimation.update_plot"></a>

#### update\_plot

```python
def update_plot(frame)
```

Update the plot for the current frame.

Args:
    frame: The current frame (e.g., parameter value).

<a id="utils.dataAugmentation"></a>

# utils.dataAugmentation

<a id="utils.dataAugmentation._Utils"></a>

## \_Utils Objects

```python
class _Utils()
```

Utility class for data augmentation techniques.

This class provides methods to check if classes are balanced and to separate samples by class.

<a id="utils.dataAugmentation._Utils.check_class_balance"></a>

#### check\_class\_balance

```python
@staticmethod
def check_class_balance(y)
```

Checks the balance of classes in the given array.

Args:
    y: (array-like) - Array of class labels.

Returns:
    tuple: (int, np.ndarray) - A tuple containing the number of unique classes and an array of counts for each class.

<a id="utils.dataAugmentation._Utils.separate_samples"></a>

#### separate\_samples

```python
@staticmethod
def separate_samples(X, y)
```

Separates samples based on their class labels.

Args:
    X: (np.ndarray) - The input data samples.
    y: (np.ndarray) - The class labels corresponding to the input data samples.

Returns:
    dict: (dict) - A dictionary where the keys are unique class labels and the values are arrays of samples belonging to each class.

<a id="utils.dataAugmentation._Utils.get_class_distribution"></a>

#### get\_class\_distribution

```python
@staticmethod
def get_class_distribution(y)
```

Gets the distribution of classes in the given array.

Args:
    y: (array-like) - Array of class labels.

Returns:
    dict: (dict) - A dictionary where the keys are unique class labels and the values are their respective counts.

<a id="utils.dataAugmentation._Utils.get_minority_majority_classes"></a>

#### get\_minority\_majority\_classes

```python
@staticmethod
def get_minority_majority_classes(y)
```

Gets the minority and majority classes from the given array.

Args:
    y: (array-like) - Array of class labels.

Returns:
    tuple: (int, int) - A tuple containing the minority class and the majority class.

<a id="utils.dataAugmentation._Utils.validate_Xy"></a>

#### validate\_Xy

```python
@staticmethod
def validate_Xy(X, y)
```

Validates the input data and labels.

Args:
    X: (array-like) - Feature matrix.
    y: (array-like) - Target vector.

Raises:
    ValueError: If the shapes of X and y do not match or if they are not numpy arrays.

<a id="utils.dataAugmentation.SMOTE"></a>

## SMOTE Objects

```python
class SMOTE()
```

Synthetic Minority Over-sampling Technique (SMOTE) for balancing class distribution.

SMOTE generates synthetic samples for the minority class by interpolating between existing samples.
This helps to create a more balanced dataset, which can improve the performance of machine learning models.

Algorithm Steps:
    - Step 1: Identify the minority class and its samples.
    - Step 2: For each sample in the minority class, find its k nearest neighbors (using Euclidean distance.)
    - Step 3: Randomly select one or more of these neighbors.
    - Step 4: Create synthetic samples by interpolating between the original sample and the selected neighbors.

<a id="utils.dataAugmentation.SMOTE.__init__"></a>

#### \_\_init\_\_

```python
def __init__(random_state=None, k_neighbors=5)
```

Initializes the SMOTE with an optional random state and number of neighbors.

<a id="utils.dataAugmentation.SMOTE.fit_resample"></a>

#### fit\_resample

```python
def fit_resample(X, y, force_equal=False)
```

Resamples the dataset to balance the class distribution by generating synthetic samples.

Args:
    X: (array-like) - Feature matrix.
    y: (array-like) - Target vector.
    force_equal: (bool), optional - If True, resample until classes are equal (default is False).

Returns:
    tuple: (np.ndarray, np.ndarray) - Resampled feature matrix and target vector.

<a id="utils.dataAugmentation.RandomOverSampler"></a>

## RandomOverSampler Objects

```python
class RandomOverSampler()
```

Randomly over-sample the minority class by duplicating examples.

This technique helps to balance the class distribution by randomly duplicating samples from the minority class.
It is a simple yet effective method to address class imbalance in datasets.

Algorithm Steps:
    - Step 1: Identify the minority class and its samples.
    - Step 2: Calculate the number of samples needed to balance the class distribution.
    - Step 3: Randomly select samples from the minority class with replacement.
    - Step 4: Duplicate the selected samples to create a balanced dataset.

<a id="utils.dataAugmentation.RandomOverSampler.__init__"></a>

#### \_\_init\_\_

```python
def __init__(random_state=None)
```

Initializes the RandomOverSampler with an optional random state.

<a id="utils.dataAugmentation.RandomOverSampler.fit_resample"></a>

#### fit\_resample

```python
def fit_resample(X, y)
```

Resamples the dataset to balance the class distribution by duplicating minority class samples.

Args:
    X: (array-like) - Feature matrix.
    y: (array-like) - Target vector.

Returns:
    tuple: (np.ndarray, np.ndarray) - Resampled feature matrix and target vector.

<a id="utils.dataAugmentation.RandomUnderSampler"></a>

## RandomUnderSampler Objects

```python
class RandomUnderSampler()
```

Randomly under-sample the majority class by removing examples.

This technique helps to balance the class distribution by randomly removing samples from the majority class.
It is a simple yet effective method to address class imbalance in datasets.

Algorithm Steps:
    - Step 1: Identify the majority class and its samples.
    - Step 2: Calculate the number of samples to remove to balance the class distribution.
    - Step 3: Randomly select samples from the majority class without replacement.
    - Step 4: Remove the selected samples to create a balanced dataset.

<a id="utils.dataAugmentation.RandomUnderSampler.__init__"></a>

#### \_\_init\_\_

```python
def __init__(random_state=None)
```

Initializes the RandomUnderSampler with an optional random state.

<a id="utils.dataAugmentation.RandomUnderSampler.fit_resample"></a>

#### fit\_resample

```python
def fit_resample(X, y)
```

Resamples the dataset to balance the class distribution by removing majority class samples.

Args:
    X: (array-like) - Feature matrix.
    y: (array-like) - Target vector.

Returns:
    tuple: (np.ndarray, np.ndarray) - Resampled feature matrix and target vector.

<a id="utils.dataAugmentation.Augmenter"></a>

## Augmenter Objects

```python
class Augmenter()
```

General class for data augmentation techniques.

This class allows for the application of multiple augmentation techniques in sequence.

<a id="utils.dataAugmentation.Augmenter.__init__"></a>

#### \_\_init\_\_

```python
def __init__(techniques, verbose=False)
```

Initializes the Augmenter with a list of techniques and verbosity option.

<a id="utils.dataAugmentation.Augmenter.augment"></a>

#### augment

```python
def augment(X, y)
```

Applies multiple augmentation techniques in sequence.

Args:
    X: (np.ndarray) - Feature matrix.
    y: (np.ndarray) - Target vector.

Returns:
    tuple: (np.ndarray, np.ndarray) - Augmented feature matrix and target vector.

<a id="utils.dataPrep"></a>

# utils.dataPrep

<a id="utils.dataPrep.DataPrep"></a>

## DataPrep Objects

```python
class DataPrep()
```

A class for preparing data for machine learning models.

<a id="utils.dataPrep.DataPrep.one_hot_encode"></a>

#### one\_hot\_encode

```python
def one_hot_encode(data, cols)
```

One-hot encodes non-numerical columns in a DataFrame or numpy array.

Drops the original columns after encoding.

Args:
    data: (pandas.DataFrame or numpy.ndarray) - The data to be encoded.
    cols: (list) - The list of column indices to be encoded.

Returns:
    data: (pandas.DataFrame or numpy.ndarray) - The data with one-hot encoded columns.

<a id="utils.dataPrep.DataPrep.find_categorical_columns"></a>

#### find\_categorical\_columns

```python
def find_categorical_columns(data)
```

Finds the indices of non-numerical columns in a DataFrame or numpy array.

Args:
    data: (pandas.DataFrame or numpy.ndarray) - The data to be checked.

Returns:
    categorical_cols: (list) - The list of indices of non-numerical columns.

<a id="utils.dataPrep.DataPrep.write_data"></a>

#### write\_data

```python
def write_data(df, csv_file, print_path=False)
```

Writes the DataFrame to a CSV file.

Args:
    df: (pandas.DataFrame) - The DataFrame to be written.
    csv_file: (str) - The path of the CSV file to write to.
    print_path: (bool), optional - If True, prints the file path (default is False).

<a id="utils.dataPrep.DataPrep.prepare_data"></a>

#### prepare\_data

```python
def prepare_data(csv_file,
                 label_col_index,
                 cols_to_encode=None,
                 write_to_csv=True)
```

Prepares the data by loading a CSV file, one-hot encoding non-numerical columns, and optionally writing the prepared data to a new CSV file.

Args:
    csv_file: (str) - The path of the CSV file to load.
    label_col_index: (int) - The index of the label column.
    cols_to_encode: (list), optional - The list of column indices to one-hot encode (default is None).
    write_to_csv: (bool), optional - Whether to write the prepared data to a new CSV file (default is True).

Returns:
    df: (pandas.DataFrame) - The prepared DataFrame.
    prepared_csv_file: (str) - The path of the prepared CSV file. If write_to_csv is False, returns "N/A".

<a id="utils.dataPrep.DataPrep.df_to_ndarray"></a>

#### df\_to\_ndarray

```python
def df_to_ndarray(df, y_col=0)
```

Converts a DataFrame to a NumPy array.

Args:
    df: (pandas.DataFrame) - The DataFrame to be converted.
    y_col: (int), optional - The index of the label column (default is 0).

Returns:
    X: (numpy.ndarray) - The feature columns as a NumPy array.
    y: (numpy.ndarray) - The label column as a NumPy array.

<a id="utils.dataPrep.DataPrep.k_split"></a>

#### k\_split

```python
def k_split(X, y, k=5)
```

Splits the data into k folds for cross-validation.

Args:
    X: (numpy.ndarray) - The feature columns.
    y: (numpy.ndarray) - The label column.
    k: (int), optional - The number of folds (default is 5).

Returns:
    X_folds: (list) - A list of k folds of feature columns.
    y_folds: (list) - A list of k folds of label columns.

<a id="utils.dataPreprocessing"></a>

# utils.dataPreprocessing

<a id="utils.dataPreprocessing.one_hot_encode"></a>

#### one\_hot\_encode

```python
def one_hot_encode(X, cols=None)
```

One-hot encodes non-numerical columns in a DataFrame or numpy array.

Drops the original columns after encoding.

Args:
    X: (pandas.DataFrame or numpy.ndarray) - The data to be encoded.
    cols: (list), optional - The list of column indices to be encoded (default is None).
        If None, all non-numerical columns will be encoded.

Returns:
    X: (pandas.DataFrame or numpy.ndarray) - The data with one-hot encoded columns.

<a id="utils.dataPreprocessing._find_categorical_columns"></a>

#### \_find\_categorical\_columns

```python
def _find_categorical_columns(X)
```

Finds the indices of non-numerical columns in a DataFrame or numpy array.

Args:
    X: (pandas.DataFrame or numpy.ndarray) - The data to be checked.

Returns:
    categorical_cols: (list) - The list of indices of non-numerical columns.

<a id="utils.dataPreprocessing.normalize"></a>

#### normalize

```python
def normalize(X, norm="l2")
```

Normalizes the input data using the specified norm.

Args:
    X: (numpy.ndarray) - The input data to be normalized.
    norm: (str), optional - The type of norm to use for normalization (default is 'l2').
        Options:
            - 'l2': L2 normalization (Euclidean norm).
            - 'l1': L1 normalization (Manhattan norm).
            - 'max': Max normalization (divides by the maximum absolute value).
            - 'minmax': Min-max normalization (scales to [0, 1]).

Returns:
    X: (numpy.ndarray) - The normalized data.

<a id="utils.dataPreprocessing.Scaler"></a>

## Scaler Objects

```python
class Scaler()
```

A class for scaling data by standardization and normalization.

<a id="utils.dataPreprocessing.Scaler.__init__"></a>

#### \_\_init\_\_

```python
def __init__(method="standard")
```

Initializes the scaler with the specified method.

Args:
    method: (str) - The scaling method to use. Options are 'standard', 'minmax', or 'normalize'.

<a id="utils.dataPreprocessing.Scaler.fit"></a>

#### fit

```python
def fit(X)
```

Fits the scaler to the data.

Args:
    X: (numpy.ndarray) - The data to fit the scaler to.

<a id="utils.dataPreprocessing.Scaler.transform"></a>

#### transform

```python
def transform(X)
```

Transforms the data using the fitted scaler.

Args:
    X: (numpy.ndarray) - The data to transform.

Returns:
    X_transformed: (numpy.ndarray) - The transformed data.

<a id="utils.dataPreprocessing.Scaler.fit_transform"></a>

#### fit\_transform

```python
def fit_transform(X)
```

Fits the scaler to the data and then transforms it.

Args:
    X: (numpy.ndarray) - The data to fit and transform.

Returns:
    X_transformed: (numpy.ndarray) - The transformed data.

<a id="utils.dataPreprocessing.Scaler.inverse_transform"></a>

#### inverse\_transform

```python
def inverse_transform(X)
```

Inverse transforms the data using the fitted scaler.

Args:
    X: (numpy.ndarray) - The data to inverse transform.

Returns:
    X_inverse: (numpy.ndarray) - The inverse transformed data.

<a id="utils.dataSplitting"></a>

# utils.dataSplitting

<a id="utils.dataSplitting.train_test_split"></a>

#### train\_test\_split

```python
def train_test_split(*arrays,
                     test_size=None,
                     train_size=None,
                     random_state=None,
                     shuffle=True,
                     stratify=None)
```

Split arrays or matrices into random train and test subsets.

Parameters
----------
*arrays : sequence of arrays
    Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas DataFrames.

test_size : float or int, default=None
    If float, should be between 0.0 and 1.0 and represent the proportion
    of the dataset to include in the test split. If int, represents the
    absolute number of test samples.

train_size : float or int, default=None
    If float, should be between 0.0 and 1.0 and represent the
    proportion of the dataset to include in the train split. If int,
    represents the absolute number of train samples. If None, the value is
    automatically computed as the complement of the test size (unless both
    are None, in which case test_size defaults to 0.25).

random_state : int, RandomState instance or None, default=None
    Controls the shuffling applied to the data before applying the split.
    Pass an int for reproducible output across multiple function calls.

shuffle : bool, default=True
    Whether or not to shuffle the data before splitting. If shuffle=False
    then stratify must be None.

stratify : array-like, default=None
    If not None, data is split in a stratified fashion, using this as
    the class labels.

Returns:
-------
splitting : list, length=2 * len(arrays)
    List containing train-test split of inputs.

<a id="utils.decomposition"></a>

# utils.decomposition

<a id="utils.decomposition.PCA"></a>

## PCA Objects

```python
class PCA()
```

Principal Component Analysis (PCA) implementation.

<a id="utils.decomposition.PCA.__init__"></a>

#### \_\_init\_\_

```python
def __init__(n_components)
```

Initializes the PCA model.

Args:
    n_components: (int) - Number of principal components to keep.

<a id="utils.decomposition.PCA.fit"></a>

#### fit

```python
def fit(X)
```

Fits the PCA model to the data.

Args:
    X: (np.ndarray) - Input data of shape (n_samples, n_features).

Raises:
    ValueError: If input data is not a 2D numpy array or if n_components exceeds the number of features.

<a id="utils.decomposition.PCA.transform"></a>

#### transform

```python
def transform(X)
```

Applies dimensionality reduction on the input data.

Args:
    X: (np.ndarray) - Input data of shape (n_samples, n_features).

Returns:
    X_transformed: (np.ndarray) - Data transformed into the principal component space of shape (n_samples, n_components).

Raises:
    ValueError: If input data is not a 2D numpy array or if its dimensions do not match the fitted data.

<a id="utils.decomposition.PCA.fit_transform"></a>

#### fit\_transform

```python
def fit_transform(X)
```

Fits the PCA model and applies dimensionality reduction on the input data.

Args:
    X: (np.ndarray) - Input data of shape (n_samples, n_features).

Returns:
    X_transformed: (np.ndarray) - Data transformed into the principal component space of shape (n_samples, n_components).

<a id="utils.decomposition.PCA.get_explained_variance_ratio"></a>

#### get\_explained\_variance\_ratio

```python
def get_explained_variance_ratio()
```

Retrieves the explained variance ratio.

Returns:
    explained_variance_ratio_: (np.ndarray) - Array of explained variance ratios for each principal component.

<a id="utils.decomposition.PCA.get_components"></a>

#### get\_components

```python
def get_components()
```

Retrieves the principal components.

Returns:
    components_: (np.ndarray) - Array of principal components of shape (n_features, n_components).

<a id="utils.decomposition.PCA.inverse_transform"></a>

#### inverse\_transform

```python
def inverse_transform(X_reduced)
```

Reconstructs the original data from the reduced data.

Args:
    X_reduced: (np.ndarray) - Reduced data of shape (n_samples, n_components).

Returns:
    X_original: (np.ndarray) - Reconstructed data of shape (n_samples, n_features).

Raises:
    ValueError: If input data is not a 2D numpy array.

<a id="utils.decomposition.SVD"></a>

## SVD Objects

```python
class SVD()
```

Singular Value Decomposition (SVD) implementation.

<a id="utils.decomposition.SVD.__init__"></a>

#### \_\_init\_\_

```python
def __init__(n_components)
```

Initializes the SVD model.

Args:
    n_components: (int) - Number of singular values and vectors to keep.

<a id="utils.decomposition.SVD.fit"></a>

#### fit

```python
def fit(X)
```

Fits the SVD model to the data.

Args:
    X: (np.ndarray) - Input data of shape (n_samples, n_features).

Raises:
    ValueError: If input data is not a 2D numpy array or if n_components exceeds the minimum dimension of the input data.

<a id="utils.decomposition.SVD.transform"></a>

#### transform

```python
def transform(X)
```

Applies the SVD transformation on the input data.

Args:
    X: (np.ndarray) - Input data of shape (n_samples, n_features).

Returns:
    X_transformed: (np.ndarray) - Data transformed into the singular value space of shape (n_samples, n_components).

Raises:
    ValueError: If input data is not a 2D numpy array.

<a id="utils.decomposition.SVD.fit_transform"></a>

#### fit\_transform

```python
def fit_transform(X)
```

Fits the SVD model and applies the transformation on the input data.

Args:
    X: (np.ndarray) - Input data of shape (n_samples, n_features).

Returns:
    X_transformed: (np.ndarray) - Data transformed into the singular value space of shape (n_samples, n_components).

<a id="utils.decomposition.SVD.get_singular_values"></a>

#### get\_singular\_values

```python
def get_singular_values()
```

Retrieves the singular values.

Returns:
    S: (np.ndarray) - Array of singular values of shape (n_components,).

<a id="utils.decomposition.SVD.get_singular_vectors"></a>

#### get\_singular\_vectors

```python
def get_singular_vectors()
```

Retrieves the singular vectors.

Returns:
    U: (np.ndarray) - Left singular vectors of shape (n_samples, n_components).
    Vt: (np.ndarray) - Right singular vectors of shape (n_components, n_features).

<a id="utils.makeData"></a>

# utils.makeData

<a id="utils.makeData.make_regression"></a>

#### make\_regression

```python
def make_regression(n_samples=100,
                    n_features=100,
                    n_informative=10,
                    n_targets=1,
                    bias=0.0,
                    effective_rank=None,
                    tail_strength=0.5,
                    noise=0.0,
                    shuffle=True,
                    coef=False,
                    random_state=None)
```

Generates a random regression problem.

Args:
    n_samples (int, optional): Number of samples (default is 100).
    n_features (int, optional): Number of features (default is 100).
    n_informative (int, optional): Number of informative features used to build the linear model (default is 10).
    n_targets (int, optional): Number of regression targets (default is 1).
    bias (float, optional): Bias term in the underlying linear model (default is 0.0).
    effective_rank (int or None, optional): Approximate dimension of the data matrix (default is None).
    tail_strength (float, optional): Relative importance of the noisy tail of the singular values profile (default is 0.5).
    noise (float, optional): Standard deviation of the Gaussian noise applied to the output (default is 0.0).
    shuffle (bool, optional): Whether to shuffle the samples and features (default is True).
    coef (bool, optional): If True, returns the coefficients of the underlying linear model (default is False).
    random_state (int or None, optional): Random seed (default is None).

Returns:
    X (np.ndarray): Input samples of shape (n_samples, n_features).
    y (np.ndarray): Output values of shape (n_samples,) or (n_samples, n_targets).
    coef (np.ndarray, optional): Coefficients of the underlying linear model of shape (n_features,) or (n_features, n_targets). Only returned if coef=True.

<a id="utils.makeData.make_classification"></a>

#### make\_classification

```python
def make_classification(n_samples=100,
                        n_features=20,
                        n_informative=2,
                        n_redundant=2,
                        n_repeated=0,
                        n_classes=2,
                        n_clusters_per_class=2,
                        weights=None,
                        flip_y=0.01,
                        class_sep=1.0,
                        hypercube=True,
                        shift=0.0,
                        scale=1.0,
                        shuffle=True,
                        random_state=None)
```

Generates a random n-class classification problem.

Args:
    n_samples (int, optional): Number of samples (default is 100).
    n_features (int, optional): Total number of features (default is 20).
    n_informative (int, optional): Number of informative features (default is 2).
    n_redundant (int, optional): Number of redundant features (default is 2).
    n_repeated (int, optional): Number of duplicated features (default is 0).
    n_classes (int, optional): Number of classes (default is 2).
    n_clusters_per_class (int, optional): Number of clusters per class (default is 2).
    weights (array-like, optional): Proportions of samples assigned to each class (default is None).
    flip_y (float, optional): Fraction of samples whose class is randomly exchanged (default is 0.01).
    class_sep (float, optional): Factor multiplying the hypercube size (default is 1.0).
    hypercube (bool, optional): If True, clusters are placed on the vertices of a hypercube (default is True).
    shift (float, optional): Shift features by the specified value (default is 0.0).
    scale (float, optional): Multiply features by the specified value (default is 1.0).
    shuffle (bool, optional): Shuffle the samples and features (default is True).
    random_state (int or None, optional): Random seed (default is None).

Returns:
    X (np.ndarray): Generated samples of shape (n_samples, n_features).
    y (np.ndarray): Integer labels for class membership of each sample of shape (n_samples,).

<a id="utils.makeData.make_blobs"></a>

#### make\_blobs

```python
def make_blobs(n_samples=100,
               n_features=2,
               centers=None,
               cluster_std=1.0,
               center_box=(-10.0, 10.0),
               shuffle=True,
               random_state=None)
```

Generates isotropic Gaussian blobs for clustering.

Args:
    n_samples (int or array-like, optional): Total number of samples if int, or number of samples per cluster if array-like (default is 100).
    n_features (int, optional): Number of features (default is 2).
    centers (int or array-like, optional): Number of centers to generate, or fixed center locations. If None, 3 centers are generated (default is None).
    cluster_std (float or array-like, optional): Standard deviation of the clusters (default is 1.0).
    center_box (tuple of float, optional): Bounding box for each cluster center when centers are generated at random (default is (-10.0, 10.0)).
    shuffle (bool, optional): Whether to shuffle the samples (default is True).
    random_state (int or None, optional): Random seed (default is None).

Returns:
    X (np.ndarray): Generated samples of shape (n_samples, n_features).
    y (np.ndarray): Integer labels for cluster membership of each sample of shape (n_samples,).
    centers (np.ndarray): Centers of each cluster of shape (n_centers, n_features).

<a id="utils.makeData.make_time_series"></a>

#### make\_time\_series

```python
def make_time_series(n_samples=100,
                     n_timestamps=50,
                     n_features=1,
                     trend="linear",
                     seasonality="sine",
                     seasonality_period=None,
                     noise=0.1,
                     random_state=None)
```

Generates synthetic time series data.

Args:
    n_samples (int, optional): Number of time series samples (default is 100).
    n_timestamps (int, optional): Number of timestamps per sample (default is 50).
    n_features (int, optional): Number of features per timestamp (default is 1).
    trend (str, optional): Type of trend ('linear', 'quadratic', or None) (default is 'linear').
    seasonality (str, optional): Type of seasonality ('sine', 'cosine', or None) (default is 'sine').
    seasonality_period (int, optional): Period of the seasonality (default is None, which uses the length of the time series/2).
    noise (float, optional): Standard deviation of Gaussian noise (default is 0.1).
    random_state (int or None, optional): Random seed (default is None).

Returns:
    X (np.ndarray): Time series data of shape (n_samples, n_timestamps, n_features).

<a id="utils.metrics"></a>

# utils.metrics

<a id="utils.metrics.Metrics"></a>

## Metrics Objects

```python
class Metrics()
```

Implements various regression and classification metrics.

<a id="utils.metrics.Metrics.mean_squared_error"></a>

#### mean\_squared\_error

```python
@classmethod
def mean_squared_error(cls, y_true, y_pred)
```

Calculates the mean squared error between the true and predicted values.

Args:
    y_true: (np.ndarray) - The true values.
    y_pred: (np.ndarray) - The predicted values.

Returns:
    mse: (float) - The mean squared error.

<a id="utils.metrics.Metrics.r_squared"></a>

#### r\_squared

```python
@classmethod
def r_squared(cls, y_true, y_pred)
```

Calculates the R-squared score between the true and predicted values.

Args:
    y_true: (np.ndarray) - The true values.
    y_pred: (np.ndarray) - The predicted values.

Returns:
    r_squared: (float) - The R-squared score.

<a id="utils.metrics.Metrics.mean_absolute_error"></a>

#### mean\_absolute\_error

```python
@classmethod
def mean_absolute_error(cls, y_true, y_pred)
```

Calculates the mean absolute error between the true and predicted values.

Args:
    y_true: (np.ndarray) - The true values.
    y_pred: (np.ndarray) - The predicted values.

Returns:
    mae: (float) - The mean absolute error.

<a id="utils.metrics.Metrics.root_mean_squared_error"></a>

#### root\_mean\_squared\_error

```python
@classmethod
def root_mean_squared_error(cls, y_true, y_pred)
```

Calculates the root mean squared error between the true and predicted values.

Args:
    y_true: (np.ndarray) - The true values.
    y_pred: (np.ndarray) - The predicted values.

Returns:
    rmse: (float) - The root mean squared error.

<a id="utils.metrics.Metrics.mean_absolute_percentage_error"></a>

#### mean\_absolute\_percentage\_error

```python
@classmethod
def mean_absolute_percentage_error(cls, y_true, y_pred)
```

Calculates the mean absolute percentage error between the true and predicted values.

Args:
    y_true: (np.ndarray) - The true values.
    y_pred: (np.ndarray) - The predicted values.

Returns:
    mape: (float) - The mean absolute percentage error as a decimal. Returns np.nan if y_true is all zeros.

<a id="utils.metrics.Metrics.mean_percentage_error"></a>

#### mean\_percentage\_error

```python
@classmethod
def mean_percentage_error(cls, y_true, y_pred)
```

Calculates the mean percentage error between the true and predicted values.

Args:
    y_true: (np.ndarray) - The true values.
    y_pred: (np.ndarray) - The predicted values.

Returns:
    mpe: (float) - The mean percentage error.

<a id="utils.metrics.Metrics.accuracy"></a>

#### accuracy

```python
@classmethod
def accuracy(cls, y_true, y_pred)
```

Calculates the accuracy score between the true and predicted values.

Args:
    y_true: (np.ndarray) - The true values.
    y_pred: (np.ndarray) - The predicted values.

Returns:
    accuracy: (float) - The accuracy score.

<a id="utils.metrics.Metrics.precision"></a>

#### precision

```python
@classmethod
def precision(cls, y_true, y_pred)
```

Calculates the precision score between the true and predicted values.

Args:
    y_true: (np.ndarray) - The true values.
    y_pred: (np.ndarray) - The predicted values.

Returns:
    precision: (float) - The precision score.

<a id="utils.metrics.Metrics.recall"></a>

#### recall

```python
@classmethod
def recall(cls, y_true, y_pred)
```

Calculates the recall score between the true and predicted values.

Args:
    y_true: (np.ndarray) - The true values.
    y_pred: (np.ndarray) - The predicted values.

Returns:
    recall: (float) - The recall score.

<a id="utils.metrics.Metrics.f1_score"></a>

#### f1\_score

```python
@classmethod
def f1_score(cls, y_true, y_pred)
```

Calculates the F1 score between the true and predicted values.

Args:
    y_true: (np.ndarray) - The true values.
    y_pred: (np.ndarray) - The predicted values.

Returns:
    f1_score: (float) - The F1 score.

<a id="utils.metrics.Metrics.log_loss"></a>

#### log\_loss

```python
@classmethod
def log_loss(cls, y_true, y_pred)
```

Calculates the log loss between the true and predicted values.

Args:
    y_true: (np.ndarray) - The true values.
    y_pred: (np.ndarray) - The predicted probabilities.

Returns:
    log_loss: (float) - The log loss.

<a id="utils.metrics.Metrics.confusion_matrix"></a>

#### confusion\_matrix

```python
@classmethod
def confusion_matrix(cls, y_true, y_pred)
```

Calculates the confusion matrix between the true and predicted values.

Args:
    y_true: (np.ndarray) - The true values.
    y_pred: (np.ndarray) - The predicted values.

Returns:
    cm: (np.ndarray) - The confusion matrix.

<a id="utils.metrics.Metrics.show_confusion_matrix"></a>

#### show\_confusion\_matrix

```python
@classmethod
def show_confusion_matrix(cls, y_true, y_pred)
```

Calculates and displays the confusion matrix between the true and predicted values.

Args:
    y_true: (np.ndarray) - The true values.
    y_pred: (np.ndarray) - The predicted values.

Returns:
    cm: (np.ndarray) - The confusion matrix.

<a id="utils.metrics.Metrics.classification_report"></a>

#### classification\_report

```python
@classmethod
def classification_report(cls, y_true, y_pred)
```

Generates a classification report for the true and predicted values.

Args:
    y_true: (np.ndarray) - The true values.
    y_pred: (np.ndarray) - The predicted values.

Returns:
    report: (dict) - The classification report.

<a id="utils.metrics.Metrics.show_classification_report"></a>

#### show\_classification\_report

```python
@classmethod
def show_classification_report(cls, y_true, y_pred)
```

Generates and displays a classification report for the true and predicted values.

Args:
    y_true: (np.ndarray) - The true values.
    y_pred: (np.ndarray) - The predicted values.

Returns:
    report: (dict) - The classification report.

<a id="utils.modelSelection"></a>

# utils.modelSelection

<a id="utils.modelSelection.ModelSelectionUtility"></a>

## ModelSelectionUtility Objects

```python
class ModelSelectionUtility()
```

A utility class for hyperparameter tuning and cross-validation of machine learning models.

<a id="utils.modelSelection.ModelSelectionUtility.get_param_combinations"></a>

#### get\_param\_combinations

```python
@staticmethod
def get_param_combinations(param_grid)
```

Generates all possible combinations of hyperparameters.

Returns:
    param_combinations (list): A list of dictionaries containing hyperparameter combinations.

<a id="utils.modelSelection.ModelSelectionUtility.cross_validate"></a>

#### cross\_validate

```python
@staticmethod
def cross_validate(model,
                   X,
                   y,
                   params,
                   cv=5,
                   metric="mse",
                   direction="minimize",
                   verbose=False)
```

Implements a custom cross-validation for hyperparameter tuning.

Args:
    model: The model Object to be tuned.
    X: (numpy.ndarray) - The feature columns.
    y: (numpy.ndarray) - The label column.
    params: (dict) - The hyperparameters to be tuned.
    cv: (int) - The number of folds for cross-validation. Default is 5.
    metric: (str) - The metric to be used for evaluation. Default is 'mse'.
        - Regression Metrics: 'mse', 'r2', 'mae', 'rmse', 'mape', 'mpe'
        - Classification Metrics: 'accuracy', 'precision', 'recall', 'f1', 'log_loss'
    direction: (str) - The direction to optimize the metric. Default is 'minimize'.
    verbose: (bool) - A flag to display the training progress. Default is False.

Returns:
    tuple: A tuple containing the scores (list) and the trained model.

<a id="utils.modelSelection.GridSearchCV"></a>

## GridSearchCV Objects

```python
class GridSearchCV()
```

Implements a grid search cross-validation for hyperparameter tuning.

<a id="utils.modelSelection.GridSearchCV.__init__"></a>

#### \_\_init\_\_

```python
def __init__(model, param_grid, cv=5, metric="mse", direction="minimize")
```

Initializes the GridSearchCV object.

Args:
    model: The model Object to be tuned.
    param_grid: (list) - A list of dictionaries containing hyperparameters to be tuned.
    cv: (int) - The number of folds for cross-validation. Default is 5.
    metric: (str) - The metric to be used for evaluation. Default is 'mse'.
        - Regression Metrics: 'mse', 'r2', 'mae', 'rmse', 'mape', 'mpe'
        - Classification Metrics: 'accuracy', 'precision', 'recall', 'f1', 'log_loss'
    direction: (str) - The direction to optimize the metric. Default is 'minimize'.

<a id="utils.modelSelection.GridSearchCV.fit"></a>

#### fit

```python
def fit(X, y, verbose=False)
```

Fits the model to the data for all hyperparameter combinations.

Args:
    X: (numpy.ndarray) - The feature columns.
    y: (numpy.ndarray) - The label column.
    verbose: (bool) - A flag to display the training progress. Default is True.

Returns:
    model: The best model with the optimal hyperparameters.

<a id="utils.modelSelection.RandomSearchCV"></a>

## RandomSearchCV Objects

```python
class RandomSearchCV()
```

Implements a random search cross-validation for hyperparameter tuning.

<a id="utils.modelSelection.RandomSearchCV.__init__"></a>

#### \_\_init\_\_

```python
def __init__(model,
             param_grid,
             iter=10,
             cv=5,
             metric="mse",
             direction="minimize")
```

Initializes the RandomSearchCV object.

Args:
    model: The model Object to be tuned.
    param_grid: (list) - A list of dictionaries containing hyperparameters to be tuned.
    iter: (int) - The number of iterations for random search. Default is 10.
    cv: (int) - The number of folds for cross-validation. Default is 5.
    metric: (str) - The metric to be used for evaluation. Default is 'mse'.
        - Regression Metrics: 'mse', 'r2', 'mae', 'rmse', 'mape', 'mpe'
        - Classification Metrics: 'accuracy', 'precision', 'recall', 'f1', 'log_loss'
    direction: (str) - The direction to optimize the metric. Default is 'minimize'.

<a id="utils.modelSelection.RandomSearchCV.fit"></a>

#### fit

```python
def fit(X, y, verbose=False)
```

Fits the model to the data for iter random hyperparameter combinations.

Args:
    X: (numpy.ndarray) - The feature columns.
    y: (numpy.ndarray) - The label column.
    verbose: (bool) - A flag to display the training progress. Default is True.

Returns:
    model: The best model with the optimal hyperparameters.

<a id="utils.modelSelection.segaSearchCV"></a>

## segaSearchCV Objects

```python
class segaSearchCV()
```

Implements a custom search cross-validation for hyperparameter tuning.

<a id="utils.modelSelection.segaSearchCV.__init__"></a>

#### \_\_init\_\_

```python
def __init__(model,
             param_space,
             iter=10,
             cv=5,
             metric="mse",
             direction="minimize")
```

Initializes the segaSearchCV object.

Args:
    model: The model Object to be tuned.
    param_space (list): A list of dictionaries containing hyperparameters to be tuned.
        Should be in the format: [{'param': [type, min, max]}, ...]
    iter (int): The number of iterations for random search. Default is 10.
    cv (int): The number of folds for cross-validation. Default is 5.
    metric (str): The metric to be used for evaluation. Default is 'mse'.
        - Regression Metrics: 'mse', 'r2', 'mae', 'rmse', 'mape', 'mpe'
        - Classification Metrics: 'accuracy', 'precision', 'recall', 'f1', 'log_loss'
    direction (str): The direction to optimize the metric. Default is 'minimize'.

<a id="utils.modelSelection.segaSearchCV.fit"></a>

#### fit

```python
def fit(X, y, verbose=False)
```

Fits the model to the data for iter random hyperparameter combinations.

Args:
    X: (numpy.ndarray)- The feature columns.
    y: (numpy.ndarray)- The label column.
    verbose: (bool) - A flag to display the training progress. Default is True.

<a id="utils.polynomialTransform"></a>

# utils.polynomialTransform

<a id="utils.polynomialTransform.PolynomialTransform"></a>

## PolynomialTransform Objects

```python
class PolynomialTransform()
```

Implements Polynomial Feature Transformation.

Polynomial feature transformation creates new features by raising existing features to a power or creating interaction terms.

Args:
    degree (int): The degree of the polynomial features (default is 2).

Attributes:
    n_samples (int): The number of samples in the input data.
    n_features (int): The number of features in the input data.
    n_output_features (int): The number of output features after transformation.
    combinations (list of tuples): The combinations of features for polynomial terms.

<a id="utils.polynomialTransform.PolynomialTransform.__init__"></a>

#### \_\_init\_\_

```python
def __init__(degree=2)
```

Initialize the PolynomialTransform object.

<a id="utils.polynomialTransform.PolynomialTransform.fit"></a>

#### fit

```python
def fit(X)
```

Fit the model to the data.

Uses itertools.combinations_with_replacement to generate all possible combinations of features(X) of degree n.

<a id="utils.polynomialTransform.PolynomialTransform.transform"></a>

#### transform

```python
def transform(X)
```

Transform the data into polynomial features by computing the product of the features for each combination of features.

<a id="utils.polynomialTransform.PolynomialTransform.fit_transform"></a>

#### fit\_transform

```python
def fit_transform(X)
```

Fit to data, then transform it.

<a id="utils.voting"></a>

# utils.voting

<a id="utils.voting.VotingRegressor"></a>

## VotingRegressor Objects

```python
class VotingRegressor()
```

Implements a voting regressor.

Takes a list of fitted models and their weights and returns a weighted average of the predictions.

<a id="utils.voting.VotingRegressor.__init__"></a>

#### \_\_init\_\_

```python
def __init__(models, model_weights=None)
```

Initialize the VotingRegressor object.

Args:
    models: list of models to be stacked
    model_weights: list of weights for each model. Default is None.

<a id="utils.voting.VotingRegressor.predict"></a>

#### predict

```python
def predict(X)
```

Predict the target variable using the fitted models.

Args:
    X: input features

Returns:
    y_pred: predicted target variable

<a id="utils.voting.VotingRegressor.get_params"></a>

#### get\_params

```python
def get_params()
```

Get the parameters of the VotingRegressor object.

Returns:
    params: dictionary of parameters

<a id="utils.voting.VotingRegressor.show_models"></a>

#### show\_models

```python
def show_models(formula=False)
```

Print the models and their weights.

<a id="utils.voting.VotingClassifier"></a>

## VotingClassifier Objects

```python
class VotingClassifier()
```

Implements a hard voting classifier.

Aggregates predictions from multiple fitted classification models based on
majority vote (optionally weighted).

<a id="utils.voting.VotingClassifier.__init__"></a>

#### \_\_init\_\_

```python
def __init__(estimators, weights=None)
```

Initialize the VotingClassifier object for hard voting.

Args:
    estimators (list): A list of *fitted* classifier objects.
                       Each estimator must have a `predict` method.
    weights (array-like of shape (n_estimators,), optional): Sequence of
        weights (float or int) to weight the occurrences of predicted class
        labels during voting. Uses uniform weights if None. Defaults to None.

<a id="utils.voting.VotingClassifier.predict"></a>

#### predict

```python
def predict(X)
```

Predict class labels for X using hard voting.

Args:
    X (array-like of shape (n_samples, n_features)): The input samples.

Returns:
    maj (np.ndarray of shape (n_samples,)): Predicted class labels based on majority vote.

<a id="utils.voting.VotingClassifier.get_params"></a>

#### get\_params

```python
def get_params(deep=True)
```

Get parameters for this estimator.

Args:
    deep (bool, optional): If True, will return the parameters for this
        estimator and contained subobjects that are estimators. (Not fully implemented for deep=True yet).

Returns:
    params (dict): Parameter names mapped to their values.

<a id="utils.voting.VotingClassifier.show_models"></a>

#### show\_models

```python
def show_models()
```

Print the models and their weights.

<a id="utils.voting.ForecastRegressor"></a>

## ForecastRegressor Objects

```python
class ForecastRegressor()
```

Implements a forcast voting regressor.

Takes a list of fitted models and their weights and returns a weighted average of the predictions.

<a id="utils.voting.ForecastRegressor.__init__"></a>

#### \_\_init\_\_

```python
def __init__(models, model_weights=None)
```

Initialize the ForecastRegressor object.

Args:
    models: list of models to be stacked
    model_weights: list of weights for each model. Default is None.

<a id="utils.voting.ForecastRegressor.forecast"></a>

#### forecast

```python
def forecast(steps)
```

Forecast the target variable using the fitted models.

Args:
    steps: number of steps to forecast

Returns:
    y_pred: predicted target variable

<a id="utils.voting.ForecastRegressor.get_params"></a>

#### get\_params

```python
def get_params()
```

Get the parameters of the ForecastRegressor object.

Returns:
    params: dictionary of parameters

<a id="utils.voting.ForecastRegressor.show_models"></a>

#### show\_models

```python
def show_models(formula=False)
```

Print the models and their weights.

�