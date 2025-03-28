# Table of Contents

* [\_\_init\_\_](#__init__)
  * [\_\_all\_\_](#__init__.__all__)
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
* [linear\_models.discriminantAnalysis](#linear_models.discriminantAnalysis)
  * [LinearDiscriminantAnalysis](#linear_models.discriminantAnalysis.LinearDiscriminantAnalysis)
    * [\_\_init\_\_](#linear_models.discriminantAnalysis.LinearDiscriminantAnalysis.__init__)
    * [fit](#linear_models.discriminantAnalysis.LinearDiscriminantAnalysis.fit)
    * [\_fit\_svd](#linear_models.discriminantAnalysis.LinearDiscriminantAnalysis._fit_svd)
    * [\_fit\_lsqr](#linear_models.discriminantAnalysis.LinearDiscriminantAnalysis._fit_lsqr)
    * [\_fit\_eigen](#linear_models.discriminantAnalysis.LinearDiscriminantAnalysis._fit_eigen)
    * [predict](#linear_models.discriminantAnalysis.LinearDiscriminantAnalysis.predict)
    * [decision\_function](#linear_models.discriminantAnalysis.LinearDiscriminantAnalysis.decision_function)
  * [QuadraticDiscriminantAnalysis](#linear_models.discriminantAnalysis.QuadraticDiscriminantAnalysis)
    * [\_\_init\_\_](#linear_models.discriminantAnalysis.QuadraticDiscriminantAnalysis.__init__)
    * [fit](#linear_models.discriminantAnalysis.QuadraticDiscriminantAnalysis.fit)
    * [predict](#linear_models.discriminantAnalysis.QuadraticDiscriminantAnalysis.predict)
    * [decision\_function](#linear_models.discriminantAnalysis.QuadraticDiscriminantAnalysis.decision_function)
  * [make\_sample\_data](#linear_models.discriminantAnalysis.make_sample_data)
* [linear\_models.linearModels](#linear_models.linearModels)
  * [OrdinaryLeastSquares](#linear_models.linearModels.OrdinaryLeastSquares)
    * [\_\_init\_\_](#linear_models.linearModels.OrdinaryLeastSquares.__init__)
    * [\_\_str\_\_](#linear_models.linearModels.OrdinaryLeastSquares.__str__)
    * [fit](#linear_models.linearModels.OrdinaryLeastSquares.fit)
    * [predict](#linear_models.linearModels.OrdinaryLeastSquares.predict)
    * [get\_formula](#linear_models.linearModels.OrdinaryLeastSquares.get_formula)
  * [Ridge](#linear_models.linearModels.Ridge)
    * [\_\_init\_\_](#linear_models.linearModels.Ridge.__init__)
    * [\_\_str\_\_](#linear_models.linearModels.Ridge.__str__)
    * [fit](#linear_models.linearModels.Ridge.fit)
    * [predict](#linear_models.linearModels.Ridge.predict)
    * [get\_formula](#linear_models.linearModels.Ridge.get_formula)
  * [Lasso](#linear_models.linearModels.Lasso)
    * [\_\_init\_\_](#linear_models.linearModels.Lasso.__init__)
    * [\_\_str\_\_](#linear_models.linearModels.Lasso.__str__)
    * [fit](#linear_models.linearModels.Lasso.fit)
    * [predict](#linear_models.linearModels.Lasso.predict)
    * [get\_formula](#linear_models.linearModels.Lasso.get_formula)
  * [Bayesian](#linear_models.linearModels.Bayesian)
    * [\_\_init\_\_](#linear_models.linearModels.Bayesian.__init__)
    * [\_\_str\_\_](#linear_models.linearModels.Bayesian.__str__)
    * [fit](#linear_models.linearModels.Bayesian.fit)
    * [tune](#linear_models.linearModels.Bayesian.tune)
    * [predict](#linear_models.linearModels.Bayesian.predict)
    * [get\_formula](#linear_models.linearModels.Bayesian.get_formula)
  * [RANSAC](#linear_models.linearModels.RANSAC)
    * [\_\_init\_\_](#linear_models.linearModels.RANSAC.__init__)
    * [\_\_str\_\_](#linear_models.linearModels.RANSAC.__str__)
    * [\_square\_loss](#linear_models.linearModels.RANSAC._square_loss)
    * [\_mean\_square\_loss](#linear_models.linearModels.RANSAC._mean_square_loss)
    * [fit](#linear_models.linearModels.RANSAC.fit)
    * [predict](#linear_models.linearModels.RANSAC.predict)
    * [get\_formula](#linear_models.linearModels.RANSAC.get_formula)
  * [PassiveAggressiveRegressor](#linear_models.linearModels.PassiveAggressiveRegressor)
    * [\_\_init\_\_](#linear_models.linearModels.PassiveAggressiveRegressor.__init__)
    * [\_\_str\_\_](#linear_models.linearModels.PassiveAggressiveRegressor.__str__)
    * [fit](#linear_models.linearModels.PassiveAggressiveRegressor.fit)
    * [predict](#linear_models.linearModels.PassiveAggressiveRegressor.predict)
    * [predict\_all\_steps](#linear_models.linearModels.PassiveAggressiveRegressor.predict_all_steps)
    * [get\_formula](#linear_models.linearModels.PassiveAggressiveRegressor.get_formula)
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
  * [CrossEntropyLoss](#neural_networks.loss.CrossEntropyLoss)
    * [\_\_call\_\_](#neural_networks.loss.CrossEntropyLoss.__call__)
  * [BCEWithLogitsLoss](#neural_networks.loss.BCEWithLogitsLoss)
    * [\_\_call\_\_](#neural_networks.loss.BCEWithLogitsLoss.__call__)
* [neural\_networks.loss\_cupy](#neural_networks.loss_cupy)
  * [CuPyCrossEntropyLoss](#neural_networks.loss_cupy.CuPyCrossEntropyLoss)
    * [\_\_call\_\_](#neural_networks.loss_cupy.CuPyCrossEntropyLoss.__call__)
  * [CuPyBCEWithLogitsLoss](#neural_networks.loss_cupy.CuPyBCEWithLogitsLoss)
    * [\_\_call\_\_](#neural_networks.loss_cupy.CuPyBCEWithLogitsLoss.__call__)
* [neural\_networks.loss\_jit](#neural_networks.loss_jit)
  * [CACHE](#neural_networks.loss_jit.CACHE)
  * [JITCrossEntropyLoss](#neural_networks.loss_jit.JITCrossEntropyLoss)
    * [\_\_init\_\_](#neural_networks.loss_jit.JITCrossEntropyLoss.__init__)
    * [calculate\_loss](#neural_networks.loss_jit.JITCrossEntropyLoss.calculate_loss)
  * [calculate\_cross\_entropy\_loss](#neural_networks.loss_jit.calculate_cross_entropy_loss)
  * [JITBCEWithLogitsLoss](#neural_networks.loss_jit.JITBCEWithLogitsLoss)
    * [\_\_init\_\_](#neural_networks.loss_jit.JITBCEWithLogitsLoss.__init__)
    * [calculate\_loss](#neural_networks.loss_jit.JITBCEWithLogitsLoss.calculate_loss)
  * [calculate\_bce\_with\_logits\_loss](#neural_networks.loss_jit.calculate_bce_with_logits_loss)
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
    * [compile\_numba\_functions](#neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.compile_numba_functions)
* [neural\_networks.numba\_utils](#neural_networks.numba_utils)
  * [CACHE](#neural_networks.numba_utils.CACHE)
  * [calculate\_loss\_from\_outputs\_binary](#neural_networks.numba_utils.calculate_loss_from_outputs_binary)
  * [calculate\_loss\_from\_outputs\_multi](#neural_networks.numba_utils.calculate_loss_from_outputs_multi)
  * [calculate\_cross\_entropy\_loss](#neural_networks.numba_utils.calculate_cross_entropy_loss)
  * [calculate\_bce\_with\_logits\_loss](#neural_networks.numba_utils.calculate_bce_with_logits_loss)
  * [\_compute\_l2\_reg](#neural_networks.numba_utils._compute_l2_reg)
  * [evaluate\_batch](#neural_networks.numba_utils.evaluate_batch)
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
  * [evaluate\_jit](#neural_networks.numba_utils.evaluate_jit)
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
* [trees](#trees)
  * [\_\_all\_\_](#trees.__all__)
* [trees.gradientBoostedRegressor](#trees.gradientBoostedRegressor)
  * [GradientBoostedRegressor](#trees.gradientBoostedRegressor.GradientBoostedRegressor)
    * [\_\_init\_\_](#trees.gradientBoostedRegressor.GradientBoostedRegressor.__init__)
    * [reset](#trees.gradientBoostedRegressor.GradientBoostedRegressor.reset)
    * [fit](#trees.gradientBoostedRegressor.GradientBoostedRegressor.fit)
    * [predict](#trees.gradientBoostedRegressor.GradientBoostedRegressor.predict)
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
  * [RandomForestClassifier](#trees.randomForestClassifier.RandomForestClassifier)
    * [num\_trees](#trees.randomForestClassifier.RandomForestClassifier.num_trees)
    * [decision\_trees](#trees.randomForestClassifier.RandomForestClassifier.decision_trees)
    * [bootstraps\_datasets](#trees.randomForestClassifier.RandomForestClassifier.bootstraps_datasets)
    * [bootstraps\_labels](#trees.randomForestClassifier.RandomForestClassifier.bootstraps_labels)
    * [max\_depth](#trees.randomForestClassifier.RandomForestClassifier.max_depth)
    * [random\_seed](#trees.randomForestClassifier.RandomForestClassifier.random_seed)
    * [forest\_size](#trees.randomForestClassifier.RandomForestClassifier.forest_size)
    * [max\_depth](#trees.randomForestClassifier.RandomForestClassifier.max_depth)
    * [display](#trees.randomForestClassifier.RandomForestClassifier.display)
    * [X](#trees.randomForestClassifier.RandomForestClassifier.X)
    * [y](#trees.randomForestClassifier.RandomForestClassifier.y)
    * [XX](#trees.randomForestClassifier.RandomForestClassifier.XX)
    * [numerical\_cols](#trees.randomForestClassifier.RandomForestClassifier.numerical_cols)
    * [\_\_init\_\_](#trees.randomForestClassifier.RandomForestClassifier.__init__)
    * [reset](#trees.randomForestClassifier.RandomForestClassifier.reset)
    * [\_bootstrapping](#trees.randomForestClassifier.RandomForestClassifier._bootstrapping)
    * [bootstrapping](#trees.randomForestClassifier.RandomForestClassifier.bootstrapping)
    * [fitting](#trees.randomForestClassifier.RandomForestClassifier.fitting)
    * [voting](#trees.randomForestClassifier.RandomForestClassifier.voting)
    * [fit](#trees.randomForestClassifier.RandomForestClassifier.fit)
    * [display\_info\_gains](#trees.randomForestClassifier.RandomForestClassifier.display_info_gains)
    * [plot\_info\_gains\_together](#trees.randomForestClassifier.RandomForestClassifier.plot_info_gains_together)
    * [plot\_info\_gains](#trees.randomForestClassifier.RandomForestClassifier.plot_info_gains)
    * [predict](#trees.randomForestClassifier.RandomForestClassifier.predict)
* [trees.randomForestRegressor](#trees.randomForestRegressor)
  * [RandomForestRegressor](#trees.randomForestRegressor.RandomForestRegressor)
    * [num\_trees](#trees.randomForestRegressor.RandomForestRegressor.num_trees)
    * [decision\_trees](#trees.randomForestRegressor.RandomForestRegressor.decision_trees)
    * [bootstraps\_datasets](#trees.randomForestRegressor.RandomForestRegressor.bootstraps_datasets)
    * [bootstraps\_labels](#trees.randomForestRegressor.RandomForestRegressor.bootstraps_labels)
    * [max\_depth](#trees.randomForestRegressor.RandomForestRegressor.max_depth)
    * [random\_seed](#trees.randomForestRegressor.RandomForestRegressor.random_seed)
    * [forest\_size](#trees.randomForestRegressor.RandomForestRegressor.forest_size)
    * [max\_depth](#trees.randomForestRegressor.RandomForestRegressor.max_depth)
    * [display](#trees.randomForestRegressor.RandomForestRegressor.display)
    * [X](#trees.randomForestRegressor.RandomForestRegressor.X)
    * [y](#trees.randomForestRegressor.RandomForestRegressor.y)
    * [XX](#trees.randomForestRegressor.RandomForestRegressor.XX)
    * [numerical\_cols](#trees.randomForestRegressor.RandomForestRegressor.numerical_cols)
    * [\_\_init\_\_](#trees.randomForestRegressor.RandomForestRegressor.__init__)
    * [reset](#trees.randomForestRegressor.RandomForestRegressor.reset)
    * [\_bootstrapping](#trees.randomForestRegressor.RandomForestRegressor._bootstrapping)
    * [bootstrapping](#trees.randomForestRegressor.RandomForestRegressor.bootstrapping)
    * [fitting](#trees.randomForestRegressor.RandomForestRegressor.fitting)
    * [voting](#trees.randomForestRegressor.RandomForestRegressor.voting)
    * [fit](#trees.randomForestRegressor.RandomForestRegressor.fit)
    * [get\_stats](#trees.randomForestRegressor.RandomForestRegressor.get_stats)
    * [predict](#trees.randomForestRegressor.RandomForestRegressor.predict)
* [trees.treeClassifier](#trees.treeClassifier)
  * [ClassifierTreeUtility](#trees.treeClassifier.ClassifierTreeUtility)
    * [entropy](#trees.treeClassifier.ClassifierTreeUtility.entropy)
    * [partition\_classes](#trees.treeClassifier.ClassifierTreeUtility.partition_classes)
    * [information\_gain](#trees.treeClassifier.ClassifierTreeUtility.information_gain)
    * [best\_split](#trees.treeClassifier.ClassifierTreeUtility.best_split)
  * [ClassifierTree](#trees.treeClassifier.ClassifierTree)
    * [\_\_init\_\_](#trees.treeClassifier.ClassifierTree.__init__)
    * [learn](#trees.treeClassifier.ClassifierTree.learn)
    * [classify](#trees.treeClassifier.ClassifierTree.classify)
* [trees.treeRegressor](#trees.treeRegressor)
  * [RegressorTreeUtility](#trees.treeRegressor.RegressorTreeUtility)
    * [calculate\_variance](#trees.treeRegressor.RegressorTreeUtility.calculate_variance)
    * [partition\_classes](#trees.treeRegressor.RegressorTreeUtility.partition_classes)
    * [information\_gain](#trees.treeRegressor.RegressorTreeUtility.information_gain)
    * [best\_split](#trees.treeRegressor.RegressorTreeUtility.best_split)
  * [RegressorTree](#trees.treeRegressor.RegressorTree)
    * [\_\_init\_\_](#trees.treeRegressor.RegressorTree.__init__)
    * [learn](#trees.treeRegressor.RegressorTree.learn)
    * [predict](#trees.treeRegressor.RegressorTree.predict)
* [utils](#utils)
  * [\_\_all\_\_](#utils.__all__)
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

<a id="__init__"></a>

# \_\_init\_\_

<a id="__init__.__all__"></a>

#### \_\_all\_\_

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

This class implements the K-Means clustering algorithm along with methods for evaluating the optimal number of clusters 
and visualizing the clustering results.

Parameters:
- X: The data matrix (numpy array).
- n_clusters: The number of clusters.
- max_iter: The maximum number of iterations.
- tol: The tolerance to declare convergence.

Methods:
- __init__: Initializes the KMeans object with parameters such as the data matrix, number of clusters, maximum iterations, 
            and convergence tolerance.
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

Parameters:
- X: The data matrix (numpy array, pandas DataFrame, or list).
- n_clusters: The number of clusters.
- max_iter: The maximum number of iterations.
- tol: The tolerance to declare convergence.

<a id="clustering.clustering.KMeans._handle_categorical"></a>

#### \_handle\_categorical

```python
def _handle_categorical(X)
```

Handle categorical columns by one-hot encoding.

Parameters:
- X: The input data with potential categorical columns.

Returns:
- X_processed: The processed data with categorical columns encoded.

<a id="clustering.clustering.KMeans._convert_to_ndarray"></a>

#### \_convert\_to\_ndarray

```python
def _convert_to_ndarray(X)
```

Convert input data to a NumPy ndarray and handle categorical columns.

Parameters:
- X: The input data, which can be a list, DataFrame, or ndarray.

Returns:
- X_ndarray: The converted and processed input data as a NumPy ndarray.

<a id="clustering.clustering.KMeans.initialize_centroids"></a>

#### initialize\_centroids

```python
def initialize_centroids()
```

Randomly initialize the centroids.

Returns:
- centroids: The initialized centroids.

<a id="clustering.clustering.KMeans.assign_clusters"></a>

#### assign\_clusters

```python
def assign_clusters(centroids)
```

Assign clusters based on the nearest centroid.

Parameters:
- centroids: The current centroids.

Returns:
- labels: The cluster assignments for each data point.

<a id="clustering.clustering.KMeans.update_centroids"></a>

#### update\_centroids

```python
def update_centroids()
```

Update the centroids based on the current cluster assignments.

Returns:
- centroids: The updated centroids.

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

Parameters:
- new_X: The data matrix to predict (numpy array).

Returns:
- labels: The predicted cluster labels.

<a id="clustering.clustering.KMeans.elbow_method"></a>

#### elbow\_method

```python
def elbow_method(max_k=10)
```

Implement the elbow method to determine the optimal number of clusters.

Parameters:
- max_k: The maximum number of clusters to test.

Returns:
- distortions: A list of distortions for each k.

<a id="clustering.clustering.KMeans.calinski_harabasz_index"></a>

#### calinski\_harabasz\_index

```python
def calinski_harabasz_index(X, labels, centroids)
```

Calculate the Calinski-Harabasz Index for evaluating clustering performance.

Parameters:
- X: The data matrix (numpy array).
- labels: The cluster labels for each data point.
- centroids: The centroids of the clusters.

Returns:
- ch_index: The computed Calinski-Harabasz Index.

<a id="clustering.clustering.KMeans.davies_bouldin_index"></a>

#### davies\_bouldin\_index

```python
def davies_bouldin_index(X, labels, centroids)
```

Calculate the Davies-Bouldin Index for evaluating clustering performance.

Parameters:
- X: The data matrix (numpy array).
- labels: The cluster labels for each data point.
- centroids: The centroids of the clusters.

Returns:
- db_index: The computed Davies-Bouldin Index.

<a id="clustering.clustering.KMeans.silhouette_score"></a>

#### silhouette\_score

```python
def silhouette_score(X, labels)
```

Calculate the silhouette score for evaluating clustering performance.

Parameters:
- X: The data matrix (numpy array).
- labels: The cluster labels for each data point.

Returns:
- silhouette_score: The computed silhouette score.

<a id="clustering.clustering.KMeans.find_optimal_clusters"></a>

#### find\_optimal\_clusters

```python
def find_optimal_clusters(max_k=10, true_k=None, save_dir=None)
```

Find the optimal number of clusters using various evaluation metrics and plot the results.

Parameters:
- X: The data matrix (numpy array).
- max_k: The maximum number of clusters to consider.
- true_k: The true number of clusters in the data.
- save_dir: The directory to save the plot (optional).

Returns:
- ch_optimal_k: The optimal number of clusters based on the Calinski-Harabasz Index.
- db_optimal_k: The optimal number of clusters based on the Davies-Bouldin Index.
- silhouette_optimal_k: The optimal number of clusters based on the Silhouette Score.

<a id="clustering.clustering.DBSCAN"></a>

## DBSCAN Objects

```python
class DBSCAN()
```

This class implements the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) algorithm.

Parameters:
- X: The data matrix (numpy array).
- eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
- min_samples: The number of samples in a neighborhood for a point to be considered as a core point.

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

Parameters:
- X: The data matrix (numpy array).
- eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
- min_samples: The number of samples in a neighborhood for a point to be considered as a core point.
- compile_numba: Whether to compile the distance calculations using Numba for performance.
    If not compiled, the first call to the numba fitting function will take longer, but subsequent calls will be faster.

<a id="clustering.clustering.DBSCAN._handle_categorical"></a>

#### \_handle\_categorical

```python
def _handle_categorical(X)
```

Handle categorical columns by one-hot encoding.

Parameters:
- X: The input data with potential categorical columns.

Returns:
- X_processed: The processed data with categorical columns encoded.

<a id="clustering.clustering.DBSCAN._convert_to_ndarray"></a>

#### \_convert\_to\_ndarray

```python
def _convert_to_ndarray(X)
```

Convert input data to a NumPy ndarray and handle categorical columns.

Parameters:
- X: The input data, which can be a list, DataFrame, or ndarray.

Returns:
- X_ndarray: The converted and processed input data as a NumPy ndarray.

<a id="clustering.clustering.DBSCAN._custom_distance_matrix"></a>

#### \_custom\_distance\_matrix

```python
def _custom_distance_matrix(X1, X2, metric='euclidean')
```

Calculate the pairwise distance matrix between two sets of data points using a custom distance calculation method.

Parameters:
- X1: The first data matrix (numpy array).
- X2: The second data matrix (numpy array).
- metric: The distance metric to use ('euclidean', 'manhattan', or 'cosine').

Returns:
- dist_matrix: The pairwise distance matrix between data points in X1 and X2.

<a id="clustering.clustering.DBSCAN.fit"></a>

#### fit

```python
def fit(metric='euclidean', numba=False)
```

Fit the DBSCAN model to the data.

Algorithm Steps:
1. Calculate the distance matrix between all points in the dataset.
2. Identify core points based on the minimum number of neighbors within eps distance.
3. Assign cluster labels using depth-first search (DFS) starting from core points.

Parameters:
- metric: The distance metric to use ('euclidean', 'manhattan', or 'cosine').
- numba: Whether to use numba for faster computation.

Returns:
- labels: The cluster labels for each data point.

<a id="clustering.clustering.DBSCAN.predict"></a>

#### predict

```python
def predict(new_X)
```

Predict the cluster labels for new data points.
Note: DBSCAN does not naturally support predicting new data points.

Parameters:
- new_X: The data matrix to predict (numpy array).

Returns:
- labels: The predicted cluster labels (-1 for noise).

<a id="clustering.clustering.DBSCAN.fit_predict"></a>

#### fit\_predict

```python
def fit_predict(numba=False)
```

Fit the DBSCAN model to the data and return the cluster labels.

Returns:
- labels: The cluster labels for the data.

<a id="clustering.clustering.DBSCAN.silhouette_score"></a>

#### silhouette\_score

```python
def silhouette_score()
```

Calculate the silhouette score for evaluating clustering performance.

Returns:
- silhouette_score: The computed silhouette score.

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

Parameters:
- min: The minimum eps value to start the search.
- max: The maximum eps value to end the search.
- precision: The precision of the search.
- return_scores: Whether to return a dictionary of (eps, score) pairs.
- verbose: Whether to print the silhouette score for each eps value.

Returns:
- eps: The optimal eps value.
- scores_dict (optional): A dictionary of (eps, score) pairs if return_scores is True.

<a id="clustering._dbscan_jit_utils"></a>

# clustering.\_dbscan\_jit\_utils

<a id="clustering._dbscan_jit_utils._identify_core_points"></a>

#### \_identify\_core\_points

```python
@njit(parallel=True, fastmath=True)
def _identify_core_points(dist_matrix, eps, min_samples)
```

Identify core points based on the distance matrix, eps, and min_samples.

Parameters:
- dist_matrix: Pairwise distance matrix.
- eps: Maximum distance for neighbors.
- min_samples: Minimum number of neighbors to be a core point.

Returns:
- core_points: Boolean array indicating core points.

<a id="clustering._dbscan_jit_utils._assign_clusters"></a>

#### \_assign\_clusters

```python
@njit(parallel=False, fastmath=True)
def _assign_clusters(dist_matrix, core_points, eps)
```

Assign cluster labels using depth-first search (DFS) starting from core points.

Parameters:
- dist_matrix: Pairwise distance matrix.
- core_points: Boolean array indicating core points.
- eps: Maximum distance for neighbors.

Returns:
- labels: Cluster labels for each data point.

<a id="linear_models"></a>

# linear\_models

<a id="linear_models.__all__"></a>

#### \_\_all\_\_

<a id="linear_models.discriminantAnalysis"></a>

# linear\_models.discriminantAnalysis

<a id="linear_models.discriminantAnalysis.LinearDiscriminantAnalysis"></a>

## LinearDiscriminantAnalysis Objects

```python
class LinearDiscriminantAnalysis(object)
```

Implements Linear Discriminant Analysis.
A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes' rule.

Parameters:
- solver : {'svd', 'lsqr', 'eigen'}, default='svd'
    Solver to use for the LDA. 
        'svd' is the default and recommended solver. 
        'lsqr' is a faster alternative that can be used when the number of features is large. 
        'eigen' is an alternative solver that can be used when the number of features is small.
- priors : array-like, shape (n_classes,), default=None
    Prior probabilities of the classes. If None, the priors are uniform.

<a id="linear_models.discriminantAnalysis.LinearDiscriminantAnalysis.__init__"></a>

#### \_\_init\_\_

```python
def __init__(solver='svd', priors=None)
```

Initialize the Linear Discriminant Analysis model with the specified solver and prior probabilities.

Parameters:
- solver : {'svd', 'lsqr', 'eigen'}, default='svd'
    Solver to use for the LDA. 
        'svd' is the default and recommended solver. 
        'lsqr' is a faster alternative that can be used when the number of features is large. 
        'eigen' is an alternative solver that can be used when the number of features is small.
- priors : array-like, shape (n_classes,), default=None
    Prior probabilities of the classes. If None, the priors are uniform.

<a id="linear_models.discriminantAnalysis.LinearDiscriminantAnalysis.fit"></a>

#### fit

```python
def fit(X, y)
```

Fit the model according to the given training data.
This method computes the mean and covariance of each class, and the prior probabilities of each class.

Parameters:
- X : array-like, shape (n_samples, n_features): Training data, where n_samples is the number of samples and n_features is the number of features.
- y : array-like, shape (n_samples,): Target values, i.e., class labels.

<a id="linear_models.discriminantAnalysis.LinearDiscriminantAnalysis._fit_svd"></a>

#### \_fit\_svd

```python
def _fit_svd(X, y)
```

Fit the model using Singular Value Decomposition.
Singular Value Decomposition (SVD) is a matrix factorization technique that decomposes a matrix into three other matrices. 
In the context of LDA, SVD is used to find the linear combinations of features that best separate the classes.

Parameters:
- X : array-like, shape (n_samples, n_features): Training data, where n_samples is the number of samples and n_features is the number of features.
- y : array-like, shape (n_samples,): Target values, i.e., class labels.

<a id="linear_models.discriminantAnalysis.LinearDiscriminantAnalysis._fit_lsqr"></a>

#### \_fit\_lsqr

```python
def _fit_lsqr(X, y)
```

Fit the model using LSQR (Least Squares).
LSQR (Least Squares) is a method for solving linear equations. 
In the context of LDA, LSQR is used to find the linear combinations of features that best separate the classes by solving a least squares problem.

Parameters:
- X : array-like, shape (n_samples, n_features): Training data, where n_samples is the number of samples and n_features is the number of features.
- y : array-like, shape (n_samples,): Target values, i.e., class labels.

<a id="linear_models.discriminantAnalysis.LinearDiscriminantAnalysis._fit_eigen"></a>

#### \_fit\_eigen

```python
def _fit_eigen(X, y)
```

Fit the model using eigenvalue decomposition.
Eigenvalue decomposition is a method for decomposing a matrix into its eigenvalues and eigenvectors.
In the context of LDA, eigenvalue decomposition is used to find the linear combinations of features that best separate the classes by solving a generalized eigenvalue problem.

Parameters:
- X : array-like, shape (n_samples, n_features): Training data, where n_samples is the number of samples and n_features is the number of features.
- y : array-like, shape (n_samples,): Target values, i.e., class labels.

<a id="linear_models.discriminantAnalysis.LinearDiscriminantAnalysis.predict"></a>

#### predict

```python
def predict(X)
```

Perform classification on an array of test vectors X.

Parameters:
- X : array-like, shape (n_samples, n_features): Test data, where n_samples is the number of samples and n_features is the number of features.

Returns:
- array, shape (n_samples,): Predicted class labels for the input samples.

<a id="linear_models.discriminantAnalysis.LinearDiscriminantAnalysis.decision_function"></a>

#### decision\_function

```python
def decision_function(X)
```

Apply decision function to an array of samples. 
The decision function is the log-likelihood of each class.

Parameters:
- X : array-like, shape (n_samples, n_features): Test data, where n_samples is the number of samples and n_features is the number of features.

Returns:
- array, shape (n_samples, n_classes): Log-likelihood of each class for the input samples.

<a id="linear_models.discriminantAnalysis.QuadraticDiscriminantAnalysis"></a>

## QuadraticDiscriminantAnalysis Objects

```python
class QuadraticDiscriminantAnalysis(object)
```

Implements Quadratic Discriminant Analysis.
The quadratic term allows for more flexibility in modeling the class conditional
A classifier with a quadratic decision boundary, generated by fitting class conditional densities to the data and using Bayes' rule.

Parameters:
- priors : array-like, shape (n_classes,), default=None
    Prior probabilities of the classes. If None, the priors are uniform.
- reg_param : float, default=0.0
    Regularization parameter. If greater than 0, the covariance matrices are regularized by adding a scaled identity matrix to them.

<a id="linear_models.discriminantAnalysis.QuadraticDiscriminantAnalysis.__init__"></a>

#### \_\_init\_\_

```python
def __init__(priors=None, reg_param=0.0)
```

Initialize the Quadratic Discriminant Analysis model with the specified prior probabilities and regularization parameter.

Parameters:
- priors : array-like, shape (n_classes,), default=None
    Prior probabilities of the classes. If None, the priors are uniform.
- reg_param : float, default=0.0

<a id="linear_models.discriminantAnalysis.QuadraticDiscriminantAnalysis.fit"></a>

#### fit

```python
def fit(X, y)
```

Fit the model according to the given training data. Uses the means and covariance matrices of each class.

Parameters:
- X : array-like, shape (n_samples, n_features): Training data, where n_samples is the number of samples and n_features is the number of features.
- y : array-like, shape (n_samples,): Target values, i.e., class labels.

<a id="linear_models.discriminantAnalysis.QuadraticDiscriminantAnalysis.predict"></a>

#### predict

```python
def predict(X)
```

Perform classification on an array of test vectors X.

Parameters:
- X : array-like, shape (n_samples, n_features): Test data, where n_samples is the number of samples and n_features is the number of features.

Returns:
- array, shape (n_samples,): Predicted class labels for the input samples.

<a id="linear_models.discriminantAnalysis.QuadraticDiscriminantAnalysis.decision_function"></a>

#### decision\_function

```python
def decision_function(X)
```

Apply decision function to an array of samples.
The decision function is the log-likelihood of each class.

Parameters:
- X : array-like, shape (n_samples, n_features): Test data, where n_samples is the number of samples and n_features is the number of features.

Returns:
- array, shape (n_samples, n_classes): Log-likelihood of each class for the input samples.

<a id="linear_models.discriminantAnalysis.make_sample_data"></a>

#### make\_sample\_data

```python
def make_sample_data(n_samples,
                     n_features,
                     cov_class_1,
                     cov_class_2,
                     shift=[1, 1],
                     seed=0)
```

Make data for testing, for testing LDA and QDA. 
Data points for class 1 are generated by multiplying a random matrix with the 
    covariance matrix of class 1, and data points for class 2 are generated by multiplying 
    a random matrix with the covariance matrix of class 2 and adding [1, 1].

<a id="linear_models.linearModels"></a>

# linear\_models.linearModels

<a id="linear_models.linearModels.OrdinaryLeastSquares"></a>

## OrdinaryLeastSquares Objects

```python
class OrdinaryLeastSquares(object)
```

Ordinary Least Squares (OLS) linear regression model.

Parameters:
- fit_intercept : bool, default=True
    Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations.

Attributes:
- coef_ : ndarray of shape (n_features,) or (n_features + 1,)
    Estimated coefficients for the linear regression problem. If `fit_intercept` is True, the first element is the intercept.
- intercept_ : float
    Independent term in the linear model. Set to 0.0 if `fit_intercept` is False.

Methods: 
- fit(X, y): Fit the linear model to the data.
- predict(X): Predict using the linear model.
- get_formula(): Returns the formula of the model as a string.

<a id="linear_models.linearModels.OrdinaryLeastSquares.__init__"></a>

#### \_\_init\_\_

```python
def __init__(fit_intercept=True) -> None
```

Initialize the OrdinaryLeastSquares object.

Parameters:
- fit_intercept : bool, default=True
    Whether to calculate the intercept for this model.

<a id="linear_models.linearModels.OrdinaryLeastSquares.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

<a id="linear_models.linearModels.OrdinaryLeastSquares.fit"></a>

#### fit

```python
def fit(X, y)
```

Fit linear model.

Parameters:
- X : array-like of shape (n_samples, n_features): Training data.
- y : array-like of shape (n_samples,) or (n_samples, n_targets): Target values.

Returns:
- self : object

<a id="linear_models.linearModels.OrdinaryLeastSquares.predict"></a>

#### predict

```python
def predict(X)
```

Predict using the linear model.

Parameters:
- X : array-like of shape (n_samples, n_features): Samples.

Returns:
- y_pred : array-like of shape (n_samples,): Predicted values.

<a id="linear_models.linearModels.OrdinaryLeastSquares.get_formula"></a>

#### get\_formula

```python
def get_formula()
```

Returns the formula of the model as a string.

Returns:
- formula : str: The formula of the model.

<a id="linear_models.linearModels.Ridge"></a>

## Ridge Objects

```python
class Ridge(object)
```

This class implements Ridge Regression using Coordinate Descent.
Ridge regression implements L2 regularization, which helps to prevent overfitting by adding a penalty term to the loss function.

Parameters:
- alpha : float, default=1.0
    Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates.
- fit_intercept : bool, default=True
    Whether to calculate the intercept for this model.
- max_iter : int, default=10000
    Maximum number of iterations for the coordinate descent solver.
- tol : float, default=1e-4
    Tolerance for the optimization. The optimization stops when the change in the coefficients is less than this tolerance.

Attributes:
- coef_ : ndarray of shape (n_features,) or (n_features + 1,)
    Estimated coefficients for the linear regression problem. If `fit_intercept` is True, the first element is the intercept.
- intercept_ : float
    Independent term in the linear model. Set to 0.0 if `fit_intercept` is False.

Methods:
- fit(X, y): Fit the linear model to the data.
- predict(X): Predict using the linear model.
- get_formula(): Returns the formula of the model as a string.

<a id="linear_models.linearModels.Ridge.__init__"></a>

#### \_\_init\_\_

```python
def __init__(alpha=1.0,
             fit_intercept=True,
             max_iter=10000,
             tol=1e-4,
             compile_numba=False)
```

This class implements Ridge Regression using Coordinate Descent.
Ridge regression implements L2 regularization, which helps to prevent overfitting by adding a penalty term to the loss function.

Parameters:
- alpha : float, default=1.0
    Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates.
- fit_intercept : bool, default=True
    Whether to calculate the intercept for this model.
- max_iter : int, default=10000
    Maximum number of iterations for the coordinate descent solver.
- tol : float, default=1e-4
    Tolerance for the optimization. The optimization stops when the change in the coefficients is less than this tolerance.
- compile_numba : bool, default=False
    Whether to precompile the numba functions. If True, the numba fitting functions will be compiled before use. 
    If not compiled, the first call to the numba fitting function will take longer, but subsequent calls will be faster.

<a id="linear_models.linearModels.Ridge.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

<a id="linear_models.linearModels.Ridge.fit"></a>

#### fit

```python
def fit(X, y, numba=False)
```

Fit the model to the data using coordinate descent.

Parameters:
    - X : array-like of shape (n_samples, n_features): Training data.
    - y : array-like of shape (n_samples,) or (n_samples, n_targets): Target values.
    - numba : Whether to use numba for faster computation. Default is False.

<a id="linear_models.linearModels.Ridge.predict"></a>

#### predict

```python
def predict(X)
```

Predict using the linear model.  

Parameters:
- X : array-like of shape (n_samples, n_features): Samples.

<a id="linear_models.linearModels.Ridge.get_formula"></a>

#### get\_formula

```python
def get_formula()
```

Computes the formula of the model.

Returns:
- formula : str: The formula of the model.

<a id="linear_models.linearModels.Lasso"></a>

## Lasso Objects

```python
class Lasso(object)
```

This class implements Lasso Regression using Coordinate Descent.
Lasso regression implements L1 regularization, which helps to prevent overfitting by adding a penalty term to the loss function.

Parameters:
- alpha : float, default=1.0
    Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates.
- fit_intercept : bool, default=True
    Whether to calculate the intercept for this model.
- max_iter : int, default=10000
    Maximum number of iterations for the coordinate descent solver.
- tol : float, default=1e-4
    Tolerance for the optimization. The optimization stops when the change in the coefficients is less than this tolerance.

Attributes:
- coef_ : ndarray of shape (n_features,) or (n_features + 1,)
    Estimated coefficients for the linear regression problem. If `fit_intercept` is True, the first element is the intercept.
- intercept_ : float
    Independent term in the linear model. Set to 0.0 if `fit_intercept` is False.

<a id="linear_models.linearModels.Lasso.__init__"></a>

#### \_\_init\_\_

```python
def __init__(alpha=1.0,
             fit_intercept=True,
             max_iter=10000,
             tol=1e-4,
             compile_numba=False)
```

This class implements Lasso Regression using Coordinate Descent.
Lasso regression implements L1 regularization, which helps to prevent overfitting by adding a penalty term to the loss function.

Parameters:
- alpha : float, default=1.0
    Regularization strength; must be a positive float. Regularization improves the conditioning of the problem and reduces the variance of the estimates.
- fit_intercept : bool, default=True
    Whether to calculate the intercept for this model.
- max_iter : int, default=10000
    Maximum number of iterations for the coordinate descent solver.
- tol : float, default=1e-4
    Tolerance for the optimization. The optimization stops when the change in the coefficients is less than this tolerance.
- compile_numba : bool, default=False
    Whether to precompile the numba functions. If True, the numba fitting functions will be compiled before use. 
    If not compiled, the first call to the numba fitting function will take longer, but subsequent calls will be faster.

<a id="linear_models.linearModels.Lasso.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

<a id="linear_models.linearModels.Lasso.fit"></a>

#### fit

```python
def fit(X, y, numba=False)
```

Fit the model to the data using coordinate descent.

Parameters:
    - X : array-like of shape (n_samples, n_features): Training data.
    - y : array-like of shape (n_samples,) or (n_samples, n_targets): Target values.
    - numba : Whether to use numba for faster computation. Default is False.

<a id="linear_models.linearModels.Lasso.predict"></a>

#### predict

```python
def predict(X)
```

Predict using the linear model.

Parameters:
- X : array-like of shape (n_samples, n_features): Samples.

<a id="linear_models.linearModels.Lasso.get_formula"></a>

#### get\_formula

```python
def get_formula()
```

Computes the formula of the model.

Returns:
- formula : str: The formula of the model.

<a id="linear_models.linearModels.Bayesian"></a>

## Bayesian Objects

```python
class Bayesian(object)
```

This class implements Bayesian Regression using Coordinate Descent.
Bayesian regression implements both L1 and L2 regularization, which helps to prevent overfitting by adding a penalty term to the loss function.

Parameters:
- max_iter: int, default=300
    The maximum number of iterations to perform.
- tol: float, default=0.001
    The convergence threshold. The algorithm will stop if the coefficients change less than the threshold.
- alpha_1: float, default=1e-06
    The shape parameter for the prior on the weights.
- alpha_2: float, default=1e-06
    The scale parameter for the prior on the weights.
- lambda_1: float, default=1e-06
    The shape parameter for the prior on the noise.
- lambda_2: float, default=1e-06
    The scale parameter for the prior on the noise.        
- fit_intercept: bool, default=True
     Whether to calculate the intercept for this model.

Attributes:
- intercept_: float
    The intercept of the model.
- coef_: ndarray of shape (n_features,) or (n_features + 1,)
    Estimated coefficients for the linear regression problem. If `fit_intercept` is True, the first element is the intercept.
- n_iter_: int  
     The number of iterations performed.
- alpha_: float
    The precision of the weights.
- lambda_: float
    The precision of the noise.
- sigma_: ndarray of shape (n_features, n_features)
    The posterior covariance of the weights.

<a id="linear_models.linearModels.Bayesian.__init__"></a>

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

This class implements Bayesian Regression using Coordinate Descent.
Bayesian regression implements both L1 and L2 regularization, which helps to prevent overfitting by adding a penalty term to the loss function.

Parameters:
- max_iter: int, default=300
    The maximum number of iterations to perform.
- tol: float, default=0.001
    The convergence threshold. The algorithm will stop if the coefficients change less than the threshold.
- alpha_1: float, default=1e-06
    The shape parameter for the prior on the weights.
- alpha_2: float, default=1e-06
    The scale parameter for the prior on the weights.
- lambda_1: float, default=1e-06
    The shape parameter for the prior on the noise.
- lambda_2: float, default=1e-06
    The scale parameter for the prior on the noise.        
- fit_intercept: bool, default=True
    Whether to calculate the intercept for this model.

<a id="linear_models.linearModels.Bayesian.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

<a id="linear_models.linearModels.Bayesian.fit"></a>

#### fit

```python
def fit(X, y)
```

Fit the model to the data.

Parameters:
- X : array-like of shape (n_samples, n_features): Training data.
- y : array-like of shape (n_samples,) or (n_samples, n_targets): Target values.

<a id="linear_models.linearModels.Bayesian.tune"></a>

#### tune

```python
def tune(X, y, beta1=0.9, beta2=0.999, iter=1000)
```

Automatically tune the hyperparameters alpha_1, alpha_2, lambda_1, lambda_2.
Loops through the parameter space, and returns the best hyperparameters based on the mean squared error.
Compues gradients using ADAM optimizer.

Parameters:
- X : array-like of shape (n_samples, n_features): Training data.
- y : array-like of shape (n_samples,) or (n_samples, n_targets): Target values.
- beta1: float, default=0.9
    The exponential decay rate for the first moment estimates.
- beta2: float, default=0.999
    The exponential decay rate for the second moment estimates.
- iter: int, default=1000
    The number of iterations to perform.

Returns:
- best_alpha_1: float
    The best value of alpha_1.
- best_alpha_2: float
    The best value of alpha_2.
- best_lambda_1: float
    The best value of lambda_1.
- best_lambda_2: float
    The best value of lambda_2.

<a id="linear_models.linearModels.Bayesian.predict"></a>

#### predict

```python
def predict(X)
```

Predict using the linear model. Computes the dot product of X and the coefficients.

Parameters:
- X : array-like of shape (n_samples, n_features): Samples.

<a id="linear_models.linearModels.Bayesian.get_formula"></a>

#### get\_formula

```python
def get_formula()
```

Computes the formula of the model.

Returns:
- formula : str: The formula of the model.

<a id="linear_models.linearModels.RANSAC"></a>

## RANSAC Objects

```python
class RANSAC(object)
```

Implements RANSAC (RANdom SAmple Consensus) algorithm for robust linear regression.
This uses the RANSAC algorithm to fit a linear model to the data, while ignoring outliers.

Parameters:
- n: int, default=10
    Number of data points to estimate parameters.
- k: int, default=100
    Maximum iterations allowed.
- t: float, default=0.05
     Threshold value to determine if points are fit well, in terms of residuals.
- d: int, default=10
    Number of close data points required to assert model fits well. 
- model: object, default=None
    The model to use for fitting. If None, uses Ordinary Least Squares.
- auto_scale_t: bool, default=False
    - Whether to automatically scale the threshold until a model is fit.
- scale_t_factor: float, default=2
    - Factor by which to scale the threshold until a model is fit.
- auto_scale_n: bool, default=False
    - Whether to automatically scale the number of data points until a model is fit.
- scale_n_factor: float, default=2
    - Factor by which to scale the number of data points until a model is fit.

Attributes:
- best_fit: object
    The best model fit.
- best_error: float
    The best error achieved by the model.
- best_n: int
    The best number of data points used to fit the model.
- best_t: float
    The best threshold value used to determine if points are fit well, in terms of residuals.
- best_model: object
    The best model fit.

<a id="linear_models.linearModels.RANSAC.__init__"></a>

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

Implements RANSAC (RANdom SAmple Consensus) algorithm for robust linear regression.
This uses the RANSAC algorithm to fit a linear model to the data, while ignoring outliers.

Parameters:
- n: int, default=10
    Number of data points to estimate parameters.
- k: int, default=100
    Maximum iterations allowed.
- t: float, default=0.05
    Threshold value to determine if points are fit well, in terms of residuals.
- d: int, default=10
    Number of close data points required to assert model fits well. 
- model: object, default=None
    The model to use for fitting. If None, uses Ordinary Least Squares.
- auto_scale_t: bool, default=False
    - Whether to automatically scale the threshold until a model is fit.
- scale_t_factor: float, default=2
    - Factor by which to scale the threshold until a model is fit.
- auto_scale_n: bool, default=False
    - Whether to automatically scale the number of data points until a model is fit.
- scale_n_factor: float, default=2
    - Factor by which to scale the number of data points until a model is fit.

<a id="linear_models.linearModels.RANSAC.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

<a id="linear_models.linearModels.RANSAC._square_loss"></a>

#### \_square\_loss

```python
def _square_loss(y_true, y_pred)
```

Compute the square loss.

<a id="linear_models.linearModels.RANSAC._mean_square_loss"></a>

#### \_mean\_square\_loss

```python
def _mean_square_loss(y_true, y_pred)
```

Compute the mean square loss.

<a id="linear_models.linearModels.RANSAC.fit"></a>

#### fit

```python
def fit(X, y)
```

Fit the model to the data, using RANSAC.

Parameters:
- X : array-like of shape (n_samples, n_features): Training data.
- y : array-like of shape (n_samples,) or (n_samples, n_targets): Target values.

<a id="linear_models.linearModels.RANSAC.predict"></a>

#### predict

```python
def predict(X)
```

Predict using the best fit model.

Parameters:
- X : array-like of shape (n_samples, n_features): Samples.

<a id="linear_models.linearModels.RANSAC.get_formula"></a>

#### get\_formula

```python
def get_formula()
```

Computes the formula of the model if fit, else returns "No model fit available"

<a id="linear_models.linearModels.PassiveAggressiveRegressor"></a>

## PassiveAggressiveRegressor Objects

```python
class PassiveAggressiveRegressor(object)
```

Implements Passive Aggressive Regression using the Passive Aggressive algorithm.
The algorithm is a type of online learning algorithm that updates the model parameters based on the current sample.
If the prediction is within a certain tolerance, the model parameters are updated.

Parameters:
- C: float, default=1.0
    Regularization parameter/step size.
- max_iter: int, default=1000
    The maximum number of passes over the training data.
- tol: float, default=1e-3
    The stopping criterion.

Attributes:
- coef_: ndarray of shape (n_features,) or (n_features + 1,)
    Estimated coefficients for the linear regression problem. If `fit_intercept` is True, the first element is the intercept.
- intercept_: float
    Independent term in the linear model. Set to 0.0 if `fit_intercept` is False.
- n_iter_: int
    The number of iterations performed.
- steps_: list of tuples of shape (n_features,) or (n_features + 1,)
    The weights and intercept at each iteration if save_steps is True.
- save_steps: bool, default=False
    - Whether to save the weights and intercept at each iteration.

<a id="linear_models.linearModels.PassiveAggressiveRegressor.__init__"></a>

#### \_\_init\_\_

```python
def __init__(C=1.0, max_iter=1000, tol=1e-3)
```

Implements Passive Aggressive Regression using the Passive Aggressive algorithm.
The algorithm is a type of online learning algorithm that updates the model parameters based on the current sample.
If the prediction is within a certain tolerance, the model parameters are updated.

Parameters:
- C: float, default=1.0
    Regularization parameter/step size.
- max_iter: int, default=1000
    The maximum number of passes over the training data.
- tol: float, default=1e-3
    The stopping criterion.

<a id="linear_models.linearModels.PassiveAggressiveRegressor.__str__"></a>

#### \_\_str\_\_

```python
def __str__()
```

<a id="linear_models.linearModels.PassiveAggressiveRegressor.fit"></a>

#### fit

```python
def fit(X, y, save_steps=False, verbose=False)
```

Fit the model to the data.
Save the weights and the intercept at each iteration if save_steps is True.

Parameters:
- X : array-like of shape (n_samples, n_features): Training data.
- y : array-like of shape (n_samples,) or (n_samples, n_targets): Target values.
- save_steps: bool, default=False
- verbose: bool, default=False

<a id="linear_models.linearModels.PassiveAggressiveRegressor.predict"></a>

#### predict

```python
def predict(X)
```

Predict using the linear model. Dot product of X and the coefficients.

<a id="linear_models.linearModels.PassiveAggressiveRegressor.predict_all_steps"></a>

#### predict\_all\_steps

```python
def predict_all_steps(X)
```

Predict using the linear model at each iteration. (save_steps=True)

<a id="linear_models.linearModels.PassiveAggressiveRegressor.get_formula"></a>

#### get\_formula

```python
def get_formula()
```

Computes the formula of the model.

Returns:
- formula : str: The formula of the model.

<a id="linear_models._lasso_jit_utils"></a>

# linear\_models.\_lasso\_jit\_utils

<a id="linear_models._lasso_jit_utils._fit_numba_no_intercept"></a>

#### \_fit\_numba\_no\_intercept

```python
@njit(parallel=True, fastmath=True)
def _fit_numba_no_intercept(X, y, alpha, max_iter, tol)
```

Fit the model to the data using coordinate descent with numba (no intercept) for Lasso.

Parameters:
    - X : array-like of shape (n_samples, n_features): Training data.
    - y : array-like of shape (n_samples,): Target values.
    - alpha : float: Regularization strength.
    - max_iter : int: Maximum number of iterations.
    - tol : float: Tolerance for convergence.

Returns:
    - coef_ : ndarray of shape (n_features,): Estimated coefficients.

<a id="linear_models._lasso_jit_utils._fit_numba_intercept"></a>

#### \_fit\_numba\_intercept

```python
@njit(parallel=True, fastmath=True)
def _fit_numba_intercept(X, y, alpha, max_iter, tol)
```

Fit the model to the data using coordinate descent with numba (with intercept) for Lasso.

Parameters:
    - X : array-like of shape (n_samples, n_features): Training data.
    - y : array-like of shape (n_samples,): Target values.
    - alpha : float: Regularization strength.
    - max_iter : int: Maximum number of iterations.
    - tol : float: Tolerance for convergence.

Returns:
    - coef_ : ndarray of shape (n_features,): Estimated coefficients.
    - intercept_ : float: Estimated intercept.

<a id="linear_models._ridge_jit_utils"></a>

# linear\_models.\_ridge\_jit\_utils

<a id="linear_models._ridge_jit_utils._fit_numba_no_intercept"></a>

#### \_fit\_numba\_no\_intercept

```python
@njit(parallel=True, fastmath=True)
def _fit_numba_no_intercept(X, y, alpha, max_iter, tol)
```

Fit the model to the data using coordinate descent with numba (no intercept).

Parameters:
    - X : array-like of shape (n_samples, n_features): Training data.
    - y : array-like of shape (n_samples,): Target values.
    - alpha : float: Regularization strength.
    - max_iter : int: Maximum number of iterations.
    - tol : float: Tolerance for convergence.

Returns:
    - coef_ : ndarray of shape (n_features,): Estimated coefficients.

<a id="linear_models._ridge_jit_utils._fit_numba_intercept"></a>

#### \_fit\_numba\_intercept

```python
@njit(parallel=True, fastmath=True)
def _fit_numba_intercept(X, y, alpha, max_iter, tol)
```

Fit the model to the data using coordinate descent with numba (with intercept).

Parameters:
    - X : array-like of shape (n_samples, n_features): Training data.
    - y : array-like of shape (n_samples,): Target values.
    - alpha : float: Regularization strength.
    - max_iter : int: Maximum number of iterations.
    - tol : float: Tolerance for convergence.

Returns:
    - coef_ : ndarray of shape (n_features,): Estimated coefficients.
    - intercept_ : float: Estimated intercept.

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

<a id="nearest_neighbors.base.KNeighborsBase.__init__"></a>

#### \_\_init\_\_

```python
def __init__(n_neighbors=5,
             distance_metric='euclidean',
             one_hot_encode=False,
             fp_precision=np.float64,
             numba=False)
```

Initialize the KNeighborsBase class.
Parameters:
- n_neighbors: int, default=5. The number of neighbors to use for the KNN algorithm.
- distance_metric: str, default='euclidean'. The distance metric to use for calculating distances.
- one_hot_encode: bool, default=False. Whether to apply one-hot encoding to the categorical columns.
- fp_precision: data type, default=np.float64. The floating point precision to use for the calculations.
- numba: bool, default=True. Whether to use numba for speeding up the calculations.

<a id="nearest_neighbors.base.KNeighborsBase.fit"></a>

#### fit

```python
def fit(X, y)
```

Fit the model using the training data.
Parameters:
- X: array-like, shape (n_samples, n_features). The training data.
- y: array-like, shape (n_samples,). The target values.

<a id="nearest_neighbors.base.KNeighborsBase.get_distance_indices"></a>

#### get\_distance\_indices

```python
def get_distance_indices(X)
```

Compute the distances and return the indices of the nearest points im the training data.
Parameters:
- X: array-like, shape (n_samples, n_features). The input data.
Returns:
- indices: array, shape (n_samples, n_neighbors). The indices of the nearest neighbors.

<a id="nearest_neighbors.base.KNeighborsBase._data_precision"></a>

#### \_data\_precision

```python
def _data_precision(X, y=None)
```

Set the floating point precision for the input data.
Parameters:
- X: array-like, shape (n_samples, n_features). The training data.
- y: array-like, shape (n_samples,). The target values.

<a id="nearest_neighbors.base.KNeighborsBase._check_data"></a>

#### \_check\_data

```python
def _check_data(X, y)
```

Check if the input data is valid.
Parameters:
- X: array-like, shape (n_samples, n_features). The input data.
- y: array-like, shape (n_samples,). The target values.

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

The @abstractmethod decorator indicates that this 
method must be implemented by any subclass of KNNBase.

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
Parameters: 
- X: array-like, shape (n_samples, n_features). The input data for which to predict the class labels.
Returns:
- predictions: array, shape (n_samples,). The predicted class labels for the input data.

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
Parameters: 
- X: array-like, shape (n_samples, n_features). The input data for which to predict the class labels.
Returns:
- predictions: array, shape (n_samples,). The predicted class labels for the input data.

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
Parameters:
- distances: 2D array of shape (n_samples, n_train_samples), precomputed distances.
- y_train: 1D array of shape (n_train_samples,), training labels.
- n_neighbors: int, number of nearest neighbors to consider.
Returns:
- predictions: 1D array of shape (n_samples,), predicted values.

<a id="nearest_neighbors._nearest_neighbors_jit_utils._numba_predict_classifier"></a>

#### \_numba\_predict\_classifier

```python
@njit(parallel=True, fastmath=True)
def _numba_predict_classifier(distances, y_train, n_neighbors)
```

Numba-optimized helper function for KNN classification predictions.
Parameters:
- distances: 2D array of shape (n_samples, n_train_samples), precomputed distances.
- y_train: 1D array of shape (n_train_samples,), training labels.
- n_neighbors: int, number of nearest neighbors to consider.
Returns:
- predictions: 1D array of shape (n_samples,), predicted class labels.

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

ReLU (Rectified Linear Unit) activation function: f(z) = max(0, z)
Returns the input directly if it's positive, otherwise returns 0.

<a id="neural_networks.activations.Activation.relu_derivative"></a>

#### relu\_derivative

```python
@staticmethod
def relu_derivative(z)
```

Derivative of the ReLU function: f'(z) = 1 if z > 0, else 0
Returns 1 for positive input, and 0 for negative input.

<a id="neural_networks.activations.Activation.leaky_relu"></a>

#### leaky\_relu

```python
@staticmethod
def leaky_relu(z, alpha=0.01)
```

Leaky ReLU activation function: f(z) = z if z > 0, else alpha * z
Allows a small, non-zero gradient when the input is negative to address the dying ReLU problem.

<a id="neural_networks.activations.Activation.leaky_relu_derivative"></a>

#### leaky\_relu\_derivative

```python
@staticmethod
def leaky_relu_derivative(z, alpha=0.01)
```

Derivative of the Leaky ReLU function: f'(z) = 1 if z > 0, else alpha
Returns 1 for positive input, and alpha for negative input.

<a id="neural_networks.activations.Activation.tanh"></a>

#### tanh

```python
@staticmethod
def tanh(z)
```

Hyperbolic tangent (tanh) activation function: f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
Maps input to the range [-1, 1], typically used for normalized input.

<a id="neural_networks.activations.Activation.tanh_derivative"></a>

#### tanh\_derivative

```python
@staticmethod
def tanh_derivative(z)
```

Derivative of the tanh function: f'(z) = 1 - tanh(z)^2
Used for backpropagation through the tanh activation.

<a id="neural_networks.activations.Activation.sigmoid"></a>

#### sigmoid

```python
@staticmethod
def sigmoid(z)
```

Sigmoid activation function: f(z) = 1 / (1 + exp(-z))
Maps input to the range [0, 1], commonly used for binary classification.

<a id="neural_networks.activations.Activation.sigmoid_derivative"></a>

#### sigmoid\_derivative

```python
@staticmethod
def sigmoid_derivative(z)
```

Derivative of the sigmoid function: f'(z) = sigmoid(z) * (1 - sigmoid(z))
Used for backpropagation through the sigmoid activation.

<a id="neural_networks.activations.Activation.softmax"></a>

#### softmax

```python
@staticmethod
def softmax(z)
```

Softmax activation function: f(z)_i = exp(z_i) / sum(exp(z_j)) for all j
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

Parameters:
    figure_size (tuple): Size of the figure (width, height)
    dpi (int): DPI for rendering

<a id="neural_networks.animation.TrainingAnimator.initialize"></a>

#### initialize

```python
def initialize(metrics_to_track, has_validation=False)
```

Initialize the animation with specified metrics.

Parameters:
    metrics_to_track (list): List of metrics to track
    has_validation (bool): Whether validation metrics are available

<a id="neural_networks.animation.TrainingAnimator.update_metrics"></a>

#### update\_metrics

```python
def update_metrics(epoch_metrics, validation=False)
```

Update the stored metrics with new values.

Parameters:
    epoch_metrics (dict): Dictionary containing metric values
    validation (bool): Whether these are validation metrics

<a id="neural_networks.animation.TrainingAnimator.animate_training_metrics"></a>

#### animate\_training\_metrics

```python
def animate_training_metrics(interval=200,
                             blit=True,
                             save_path=None,
                             save_format='mp4',
                             fps=10,
                             dpi=300)
```

Create an animation of the training metrics.

Parameters:
    interval (int): Delay between frames in milliseconds
    blit (bool): Whether to use blitting for efficient animation
    save_path (str, optional): Path to save the animation
    save_format (str): Format to save animation ('mp4', 'gif', etc.)
    fps (int): Frames per second for the saved video
    dpi (int): DPI for the saved animation

Returns:
    animation.FuncAnimation: Animation object

<a id="neural_networks.animation.TrainingAnimator.setup_training_video"></a>

#### setup\_training\_video

```python
def setup_training_video(filepath, fps=10, dpi=None)
```

Set up a video writer to capture training progress in real-time.

Parameters:
    filepath (str): Path to save the video
    fps (int): Frames per second
    dpi (int, optional): DPI for rendering

<a id="neural_networks.animation.TrainingAnimator.add_training_frame"></a>

#### add\_training\_frame

```python
def add_training_frame()
```

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

<a id="neural_networks.cupy_utils.apply_dropout"></a>

#### apply\_dropout

```python
def apply_dropout(X, dropout_rate)
```

<a id="neural_networks.cupy_utils.fused_relu"></a>

#### fused\_relu

```python
@fuse()
def fused_relu(x)
```

<a id="neural_networks.cupy_utils.fused_sigmoid"></a>

#### fused\_sigmoid

```python
@fuse()
def fused_sigmoid(x)
```

<a id="neural_networks.cupy_utils.fused_leaky_relu"></a>

#### fused\_leaky\_relu

```python
@fuse()
def fused_leaky_relu(x, alpha=0.01)
```

<a id="neural_networks.cupy_utils.forward_cupy"></a>

#### forward\_cupy

```python
def forward_cupy(X, weights, biases, activations, dropout_rate, training,
                 is_binary)
```

<a id="neural_networks.cupy_utils.backward_cupy"></a>

#### backward\_cupy

```python
def backward_cupy(layer_outputs, y, weights, activations, reg_lambda,
                  is_binary, dWs, dbs)
```

<a id="neural_networks.cupy_utils.logsumexp"></a>

#### logsumexp

```python
def logsumexp(a, axis=None, keepdims=False)
```

<a id="neural_networks.cupy_utils.calculate_cross_entropy_loss"></a>

#### calculate\_cross\_entropy\_loss

```python
def calculate_cross_entropy_loss(logits, targets)
```

<a id="neural_networks.cupy_utils.calculate_bce_with_logits_loss"></a>

#### calculate\_bce\_with\_logits\_loss

```python
def calculate_bce_with_logits_loss(logits, targets)
```

<a id="neural_networks.cupy_utils.calculate_loss_from_outputs_binary"></a>

#### calculate\_loss\_from\_outputs\_binary

```python
def calculate_loss_from_outputs_binary(outputs, y, weights, reg_lambda)
```

<a id="neural_networks.cupy_utils.calculate_loss_from_outputs_multi"></a>

#### calculate\_loss\_from\_outputs\_multi

```python
def calculate_loss_from_outputs_multi(outputs, y, weights, reg_lambda)
```

<a id="neural_networks.cupy_utils.evaluate_batch"></a>

#### evaluate\_batch

```python
def evaluate_batch(y_hat, y_true, is_binary)
```

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

<a id="neural_networks.layers.DenseLayer.backward"></a>

#### backward

```python
def backward(dA, reg_lambda)
```

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
    input_shape (tuple): Shape of the input data (excluding batch size).
    output_size (int): Size of the flattened output vector.
    input_cache (np.ndarray): Cached input for backpropagation.
    input_size (int): Size of the input (same as input_shape).
    output_size (int): Size of the output (same as output_size).

<a id="neural_networks.layers.FlattenLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__()
```

<a id="neural_networks.layers.FlattenLayer.forward"></a>

#### forward

```python
def forward(X)
```

Flattens the input tensor.

Args:
    X (np.ndarray): Input data of shape (batch_size, channels, height, width)
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
    d_out (np.ndarray): Gradient of the loss with respect to the layer output,
                      shape (batch_size, out_channels, h_out, w_out)
    reg_lambda (float, optional): Regularization parameter.
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

<a id="neural_networks.layers.RNNLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(input_size, hidden_size, activation="tanh")
```

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

<a id="neural_networks.layers_cupy.CuPyActivation.relu"></a>

#### relu

```python
@staticmethod
def relu(z)
```

ReLU (Rectified Linear Unit) activation function: f(z) = max(0, z)
Returns the input directly if it's positive, otherwise returns 0.

<a id="neural_networks.layers_cupy.CuPyActivation.relu_derivative"></a>

#### relu\_derivative

```python
@staticmethod
def relu_derivative(z)
```

Derivative of the ReLU function: f'(z) = 1 if z > 0, else 0
Returns 1 for positive input, and 0 for negative input.

<a id="neural_networks.layers_cupy.CuPyActivation.leaky_relu"></a>

#### leaky\_relu

```python
@staticmethod
def leaky_relu(z, alpha=0.01)
```

Leaky ReLU activation function: f(z) = z if z > 0, else alpha * z
Allows a small, non-zero gradient when the input is negative to address the dying ReLU problem.

<a id="neural_networks.layers_cupy.CuPyActivation.leaky_relu_derivative"></a>

#### leaky\_relu\_derivative

```python
@staticmethod
def leaky_relu_derivative(z, alpha=0.01)
```

Derivative of the Leaky ReLU function: f'(z) = 1 if z > 0, else alpha
Returns 1 for positive input, and alpha for negative input.

<a id="neural_networks.layers_cupy.CuPyActivation.tanh"></a>

#### tanh

```python
@staticmethod
def tanh(z)
```

Hyperbolic tangent (tanh) activation function: f(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
Maps input to the range [-1, 1], typically used for normalized input.

<a id="neural_networks.layers_cupy.CuPyActivation.tanh_derivative"></a>

#### tanh\_derivative

```python
@staticmethod
def tanh_derivative(z)
```

Derivative of the tanh function: f'(z) = 1 - tanh(z)^2
Used for backpropagation through the tanh activation.

<a id="neural_networks.layers_cupy.CuPyActivation.sigmoid"></a>

#### sigmoid

```python
@staticmethod
def sigmoid(z)
```

Sigmoid activation function: f(z) = 1 / (1 + exp(-z))
Maps input to the range [0, 1], commonly used for binary classification.

<a id="neural_networks.layers_cupy.CuPyActivation.sigmoid_derivative"></a>

#### sigmoid\_derivative

```python
@staticmethod
def sigmoid_derivative(z)
```

Derivative of the sigmoid function: f'(z) = sigmoid(z) * (1 - sigmoid(z))
Used for backpropagation through the sigmoid activation.

<a id="neural_networks.layers_cupy.CuPyActivation.softmax"></a>

#### softmax

```python
@staticmethod
def softmax(z)
```

Softmax activation function: f(z)_i = exp(z_i) / sum(exp(z_j)) for all j
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

<a id="neural_networks.layers_jit.JITDenseLayer.backward"></a>

#### backward

```python
def backward(dA, reg_lambda)
```

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
Fixed to avoid reshape contiguity issues.

<a id="neural_networks.layers_jit.JITConvLayer._col2im"></a>

#### \_col2im

```python
def _col2im(dcol, x_shape)
```

Convert column back to image format for the backward pass.
Fixed to avoid reshape contiguity issues.

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

<a id="neural_networks.layers_jit.JITRNNLayer.__init__"></a>

#### \_\_init\_\_

```python
def __init__(input_size, hidden_size, activation="tanh")
```

<a id="neural_networks.loss"></a>

# neural\_networks.loss

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

<a id="neural_networks.loss_jit.JITCrossEntropyLoss"></a>

## JITCrossEntropyLoss Objects

```python
class JITCrossEntropyLoss()
```

<a id="neural_networks.loss_jit.JITCrossEntropyLoss.__init__"></a>

#### \_\_init\_\_

```python
def __init__()
```

<a id="neural_networks.loss_jit.JITCrossEntropyLoss.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(logits, targets)
```

<a id="neural_networks.loss_jit.calculate_cross_entropy_loss"></a>

#### calculate\_cross\_entropy\_loss

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_cross_entropy_loss(logits, targets)
```

<a id="neural_networks.loss_jit.JITBCEWithLogitsLoss"></a>

## JITBCEWithLogitsLoss Objects

```python
class JITBCEWithLogitsLoss()
```

<a id="neural_networks.loss_jit.JITBCEWithLogitsLoss.__init__"></a>

#### \_\_init\_\_

```python
def __init__()
```

<a id="neural_networks.loss_jit.JITBCEWithLogitsLoss.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(logits, targets)
```

<a id="neural_networks.loss_jit.calculate_bce_with_logits_loss"></a>

#### calculate\_bce\_with\_logits\_loss

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_bce_with_logits_loss(logits, targets)
```

<a id="neural_networks.neuralNetworkBase"></a>

# neural\_networks.neuralNetworkBase

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase"></a>

## NeuralNetworkBase Objects

```python
class NeuralNetworkBase()
```

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.__init__"></a>

#### \_\_init\_\_

```python
def __init__(layers, dropout_rate=0.0, reg_lambda=0.0, activations=None)
```

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.initialize_layers"></a>

#### initialize\_layers

```python
def initialize_layers()
```

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.forward"></a>

#### forward

```python
def forward(X, training=True)
```

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.backward"></a>

#### backward

```python
def backward(y)
```

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

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.evaluate"></a>

#### evaluate

```python
def evaluate(X, y)
```

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.predict"></a>

#### predict

```python
def predict(X)
```

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(X, y)
```

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.apply_dropout"></a>

#### apply\_dropout

```python
def apply_dropout(X)
```

Applies dropout to the activation X.
Args:
    X (ndarray): Activation values.
Returns:
    ndarray: Activation values after applying dropout.

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.compute_l2_reg"></a>

#### compute\_l2\_reg

```python
def compute_l2_reg(weights)
```

Computes the L2 regularization term.
Args:
    weights (list): List of weight matrices.
Returns:
    float: L2 regularization term.

<a id="neural_networks.neuralNetworkBase.NeuralNetworkBase.calculate_precision_recall_f1"></a>

#### calculate\_precision\_recall\_f1

```python
def calculate_precision_recall_f1(X, y)
```

Calculates precision, recall, and F1 score.
Parameters:
    - X (ndarray): Input data
    - y (ndarray): Target labels
Returns:
    - precision (float): Precision score
    - recall (float): Recall score
    - f1 (float): F1 score

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

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.__init__"></a>

#### \_\_init\_\_

```python
def __init__(layers, dropout_rate=0.2, reg_lambda=0.01, activations=None)
```

Initializes the Numba backend neural network.
Args:
    layers (list): List of layer sizes or Layer objects.
    dropout_rate (float): Dropout rate for regularization.
    reg_lambda (float): L2 regularization parameter.
    activations (list): List of activation functions for each layer.

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
    X (ndarray): Input data of shape (batch_size, input_size).
    training (bool): Whether the network is in training mode (applies dropout).
Returns: 
    ndarray: Output predictions of shape (batch_size, output_size).

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.backward"></a>

#### backward

```python
def backward(y)
```

Performs backward propagation to calculate the gradients.
Parameters: 
    y (ndarray): Target labels of shape (m, output_size).

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
          save_path='training_animation.mp4',
          fps=1,
          dpi=100,
          frame_every=1)
```

Trains the neural network model.
Parameters:
    - X_train (ndarray): Training data features.
    - y_train (ndarray): Training data labels.
    - X_val (ndarray): Validation data features, optional.
    - y_val (ndarray): Validation data labels, optional.
    - optimizer (Optimizer): Optimizer for updating parameters (default: Adam, lr=0.0001).
    - epochs (int): Number of training epochs (default: 100).
    - batch_size (int): Batch size for mini-batch gradient descent (default: 32).
    - early_stopping_threshold (int): Patience for early stopping (default: 10).
    - lr_scheduler (Scheduler): Learning rate scheduler (default: None).
    - p (bool): Whether to print training progress (default: True).
    - use_tqdm (bool): Whether to use tqdm for progress bar (default: True).
    - n_jobs (int): Number of jobs for parallel processing (default: 1).
    - track_metrics (bool): Whether to track training metrics (default: False).
    - track_adv_metrics (bool): Whether to track advanced metrics (default: False).
    - save_animation (bool): Whether to save the animation of metrics (default: False).
    - save_path (str): Path to save the animation file. File extension must be .mp4 or .gif (default: 'training_animation.mp4').
    - fps (int): Frames per second for the saved animation (default: 1).
    - dpi (int): DPI for the saved animation (default: 100).
    - frame_every (int): Capture frame every N epochs (to reduce file size) (default: 1).

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.evaluate"></a>

#### evaluate

```python
def evaluate(X, y)
```

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.predict"></a>

#### predict

```python
def predict(X)
```

<a id="neural_networks.neuralNetworkBaseBackend.BaseBackendNeuralNetwork.calculate_loss"></a>

#### calculate\_loss

```python
def calculate_loss(X, y)
```

Calculates the loss with L2 regularization.
Parameters:
    - X (ndarray): Input data
    - y (ndarray): Target labels
Returns: 
    float: The calculated loss value

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
Parameters:
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
                                 save_path='training_animation.mp4',
                                 fps=1,
                                 dpi=100,
                                 frame_every=1)
```

Trains the neural network model while capturing training metrics in real-time animation.

Parameters:
    - X_train, y_train: Training data
    - X_val, y_val: Validation data (optional)
    - optimizer: Optimizer for updating parameters
    - epochs: Number of training epochs
    - batch_size: Batch size for mini-batch gradient descent
    - early_stopping_threshold: Patience for early stopping
    - lr_scheduler: Learning rate scheduler
    - save_path: Path to save the animation file
    - fps: Frames per second for the saved animation
    - dpi: DPI for the saved animation
    - writer: Animation writer ('ffmpeg', 'pillow', etc.)
    - frame_every: Capture frame every N epochs (to reduce file size)

Returns:
    - None

<a id="neural_networks.neuralNetworkCuPyBackend"></a>

# neural\_networks.neuralNetworkCuPyBackend

<a id="neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork"></a>

## CuPyBackendNeuralNetwork Objects

```python
class CuPyBackendNeuralNetwork(NeuralNetworkBase)
```

<a id="neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.__init__"></a>

#### \_\_init\_\_

```python
def __init__(layers, dropout_rate=0.2, reg_lambda=0.01, activations=None)
```

Initializes the Numba backend neural network.
Args:
    layers (list): List of layer sizes or Layer objects.
    dropout_rate (float): Dropout rate for regularization.
    reg_lambda (float): L2 regularization parameter.
    activations (list): List of activation functions for each layer.

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
          save_path='training_animation.mp4',
          fps=1,
          dpi=100,
          frame_every=1)
```

Trains the neural network model.
Args:
    - X_train (ndarray): Training data features.
    - y_train (ndarray): Training data labels.
    - X_val (ndarray): Validation data features, optional.
    - y_val (ndarray): Validation data labels, optional.
    - optimizer (Optimizer): Optimizer for updating parameters (default: JITAdam, lr=0.0001).
    - epochs (int): Number of training epochs (default: 100).
    - batch_size (int): Batch size for mini-batch gradient descent (default: 32).
    - early_stopping_threshold (int): Patience for early stopping (default: 10).
    - lr_scheduler (Scheduler): Learning rate scheduler (default: None).
    - p (bool): Whether to print training progress (default: True).
    - use_tqdm (bool): Whether to use tqdm for progress bar (default: True).
    - n_jobs (int): Number of jobs for parallel processing (default: 1).
    - track_metrics (bool): Whether to track training metrics (default: False).
    - track_adv_metrics (bool): Whether to track advanced metrics (default: False).
    - save_animation (bool): Whether to save the animation of metrics (default: False).
    - save_path (str): Path to save the animation file. File extension must be .mp4 or .gif (default: 'training_animation.mp4').
    - fps (int): Frames per second for the saved animation (default: 1).
    - dpi (int): DPI for the saved animation (default: 100).
    - frame_every (int): Capture frame every N epochs (to reduce file size) (default: 1).

<a id="neural_networks.neuralNetworkCuPyBackend.CuPyBackendNeuralNetwork.evaluate"></a>

#### evaluate

```python
def evaluate(X, y)
```

Evaluates the model performance.
Parameters:
    - X (ndarray): Input data (NumPy or CuPy array)
    - y (ndarray): Target labels (NumPy or CuPy array)
Returns:
    - accuracy (float): Model accuracy
    - predicted (ndarray): Predicted labels (NumPy array)

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

<a id="neural_networks.neuralNetworkNumbaBackend.NumbaBackendNeuralNetwork.__init__"></a>

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
          save_path='training_animation.mp4',
          fps=1,
          dpi=100,
          frame_every=1)
```

Trains the neural network model.
Args:
    - X_train (ndarray): Training data features.
    - y_train (ndarray): Training data labels.
    - X_val (ndarray): Validation data features, optional.
    - y_val (ndarray): Validation data labels, optional.
    - optimizer (Optimizer): Optimizer for updating parameters (default: JITAdam, lr=0.0001).
    - epochs (int): Number of training epochs (default: 100).
    - batch_size (int): Batch size for mini-batch gradient descent (default: 32).
    - early_stopping_threshold (int): Patience for early stopping (default: 10).
    - lr_scheduler (Scheduler): Learning rate scheduler (default: None).
    - p (bool): Whether to print training progress (default: True).
    - use_tqdm (bool): Whether to use tqdm for progress bar (default: True).
    - n_jobs (int): Number of jobs for parallel processing (default: 1).
    - track_metrics (bool): Whether to track training metrics (default: False).
    - track_adv_metrics (bool): Whether to track advanced metrics (default: False).
    - save_animation (bool): Whether to save the animation of metrics (default: False).
    - save_path (str): Path to save the animation file. File extension must be .mp4 or .gif (default: 'training_animation.mp4').
    - fps (int): Frames per second for the saved animation (default: 1).
    - dpi (int): DPI for the saved animation (default: 100).
    - frame_every (int): Capture frame every N epochs (to reduce file size) (default: 1).

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
Parameters:
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

<a id="neural_networks.numba_utils.calculate_loss_from_outputs_multi"></a>

#### calculate\_loss\_from\_outputs\_multi

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_loss_from_outputs_multi(outputs, y, weights, reg_lambda)
```

<a id="neural_networks.numba_utils.calculate_cross_entropy_loss"></a>

#### calculate\_cross\_entropy\_loss

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_cross_entropy_loss(logits, targets)
```

<a id="neural_networks.numba_utils.calculate_bce_with_logits_loss"></a>

#### calculate\_bce\_with\_logits\_loss

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def calculate_bce_with_logits_loss(logits, targets)
```

<a id="neural_networks.numba_utils._compute_l2_reg"></a>

#### \_compute\_l2\_reg

```python
@njit(fastmath=True, nogil=True, parallel=True, cache=CACHE)
def _compute_l2_reg(weights)
```

<a id="neural_networks.numba_utils.evaluate_batch"></a>

#### evaluate\_batch

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def evaluate_batch(y_hat, y_true, is_binary)
```

<a id="neural_networks.numba_utils.relu"></a>

#### relu

```python
@njit(fastmath=True, cache=CACHE)
def relu(z)
```

<a id="neural_networks.numba_utils.relu_derivative"></a>

#### relu\_derivative

```python
@njit(fastmath=True, cache=CACHE)
def relu_derivative(z)
```

<a id="neural_networks.numba_utils.leaky_relu"></a>

#### leaky\_relu

```python
@njit(fastmath=True, cache=CACHE)
def leaky_relu(z, alpha=0.01)
```

<a id="neural_networks.numba_utils.leaky_relu_derivative"></a>

#### leaky\_relu\_derivative

```python
@njit(fastmath=True, cache=CACHE)
def leaky_relu_derivative(z, alpha=0.01)
```

<a id="neural_networks.numba_utils.tanh"></a>

#### tanh

```python
@njit(fastmath=True, cache=CACHE)
def tanh(z)
```

<a id="neural_networks.numba_utils.tanh_derivative"></a>

#### tanh\_derivative

```python
@njit(fastmath=True, cache=CACHE)
def tanh_derivative(z)
```

<a id="neural_networks.numba_utils.sigmoid"></a>

#### sigmoid

```python
@njit(fastmath=True, cache=CACHE)
def sigmoid(z)
```

<a id="neural_networks.numba_utils.sigmoid_derivative"></a>

#### sigmoid\_derivative

```python
@njit(fastmath=True, cache=CACHE)
def sigmoid_derivative(z)
```

<a id="neural_networks.numba_utils.softmax"></a>

#### softmax

```python
@njit(parallel=True, fastmath=True, cache=CACHE)
def softmax(z)
```

<a id="neural_networks.numba_utils.sum_reduce"></a>

#### sum\_reduce

```python
@njit(fastmath=True, cache=CACHE)
def sum_reduce(arr)
```

<a id="neural_networks.numba_utils.sum_axis0"></a>

#### sum\_axis0

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def sum_axis0(arr)
```

<a id="neural_networks.numba_utils.apply_dropout_jit"></a>

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

<a id="neural_networks.numba_utils.compute_l2_reg"></a>

#### compute\_l2\_reg

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def compute_l2_reg(weights)
```

<a id="neural_networks.numba_utils.one_hot_encode"></a>

#### one\_hot\_encode

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def one_hot_encode(y, num_classes)
```

<a id="neural_networks.numba_utils.process_batches_binary"></a>

#### process\_batches\_binary

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def process_batches_binary(X_shuffled, y_shuffled, batch_size, layers,
                           dropout_rate, dropout_layer_indices, reg_lambda,
                           dWs_acc, dbs_acc)
```

<a id="neural_networks.numba_utils.process_batches_multi"></a>

#### process\_batches\_multi

```python
@njit(fastmath=True, nogil=True, cache=CACHE)
def process_batches_multi(X_shuffled, y_shuffled, batch_size, layers,
                          dropout_rate, dropout_layer_indices, reg_lambda,
                          dWs_acc, dbs_acc)
```

<a id="neural_networks.numba_utils.evaluate_jit"></a>

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

<a id="neural_networks.optimizers.AdamOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the first and second moment estimates for each layer's weights.
Args: layers (list): List of layers in the neural network.
Returns: None

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

<a id="neural_networks.optimizers.SGDOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the velocity for each layer's weights.
Args: layers (list): List of layers in the neural network.
Returns: None

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
Returns: None

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

<a id="neural_networks.optimizers.AdadeltaOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the running averages for each layer's weights.
Args: layers (list): List of layers in the neural network.
Returns: None

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
Returns: None

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

<a id="neural_networks.optimizers_cupy.CuPyAdamOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

<a id="neural_networks.optimizers_cupy.CuPyAdamOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

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

<a id="neural_networks.optimizers_cupy.CuPySGDOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

<a id="neural_networks.optimizers_cupy.CuPySGDOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

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

<a id="neural_networks.optimizers_cupy.CuPyAdadeltaOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

<a id="neural_networks.optimizers_cupy.CuPyAdadeltaOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

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

<a id="neural_networks.optimizers_jit.JITAdamOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the first and second moment estimates for each layer's weights.
Args: layers (list): List of layers in the neural network.
Returns: None

<a id="neural_networks.optimizers_jit.JITAdamOptimizer.update"></a>

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

<a id="neural_networks.optimizers_jit.JITAdamOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

<a id="neural_networks.optimizers_jit.adam_update_layers"></a>

#### adam\_update\_layers

```python
@njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
def adam_update_layers(m, v, t, layers, dWs, dbs, learning_rate, beta1, beta2,
                       epsilon, reg_lambda)
```

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

<a id="neural_networks.optimizers_jit.JITSGDOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the velocity for each layer's weights.
Args: layers (list): List of layers in the neural network.
Returns: None

<a id="neural_networks.optimizers_jit.JITSGDOptimizer.update"></a>

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
Returns: None

<a id="neural_networks.optimizers_jit.JITSGDOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

<a id="neural_networks.optimizers_jit.sgd_update_layers"></a>

#### sgd\_update\_layers

```python
@njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
def sgd_update_layers(velocity, layers, dWs, dbs, learning_rate, momentum,
                      reg_lambda)
```

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

<a id="neural_networks.optimizers_jit.JITAdadeltaOptimizer.initialize"></a>

#### initialize

```python
def initialize(layers)
```

Initializes the running averages for each layer's weights.
Args: layers (list): List of layers in the neural network.
Returns: None

<a id="neural_networks.optimizers_jit.JITAdadeltaOptimizer.update"></a>

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
Returns: None

<a id="neural_networks.optimizers_jit.JITAdadeltaOptimizer.update_layers"></a>

#### update\_layers

```python
def update_layers(layers, dWs, dbs)
```

<a id="neural_networks.optimizers_jit.adadelta_update_layers"></a>

#### adadelta\_update\_layers

```python
@njit(parallel=True, fastmath=True, nogil=True, cache=CACHE)
def adadelta_update_layers(E_g2, E_delta_x2, layers, dWs, dbs, learning_rate,
                           rho, epsilon, reg_lambda)
```

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

<a id="neural_networks.schedulers.lr_scheduler_step.__repr__"></a>

#### \_\_repr\_\_

```python
def __repr__()
```

<a id="neural_networks.schedulers.lr_scheduler_step.step"></a>

#### step

```python
def step(epoch)
```

Adjusts the learning rate based on the current epoch. Decays the learning rate by lr_decay every lr_decay_epoch epochs.
Args:
    epoch (int): The current epoch number.
Returns: None

<a id="neural_networks.schedulers.lr_scheduler_step.reduce"></a>

#### reduce

```python
def reduce()
```

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

<a id="neural_networks.schedulers.lr_scheduler_exp.__repr__"></a>

#### \_\_repr\_\_

```python
def __repr__()
```

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

<a id="neural_networks.schedulers.lr_scheduler_plateau.__repr__"></a>

#### \_\_repr\_\_

```python
def __repr__()
```

<a id="neural_networks.schedulers.lr_scheduler_plateau.step"></a>

#### step

```python
def step(epoch, loss)
```

Updates the learning rate based on the loss value.
Args:
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
Parameters:
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
Parameters:
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
Parameters:
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
Parameters:
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
Parameters:
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
Parameters:
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

<a id="svm.baseSVM.BaseSVM.__init__"></a>

#### \_\_init\_\_

```python
def __init__(C=1.0,
             tol=1e-4,
             max_iter=1000,
             learning_rate=0.01,
             kernel='linear',
             degree=3,
             gamma='scale',
             coef0=0.0,
             regression=False)
```

Initialize the BaseSVM class with kernel support.

Parameters:
- C: float, regularization parameter
- tol: float, tolerance for stopping criteria
- max_iter: int, maximum number of iterations
- learning_rate: float, step size for optimization
- kernel: str, 'linear', 'poly', 'rbf', or 'sigmoid'
- degree: int, degree for polynomial kernel
- gamma: str or float, kernel coefficient ('scale', 'auto', or float)
- coef0: float, independent term in poly and sigmoid kernels
- regression: bool, whether to use regression (SVR) or classification (SVC) (default: False)

<a id="svm.baseSVM.BaseSVM.fit"></a>

#### fit

```python
def fit(X, y=None)
```

Fit the SVM model.

Parameters:
- X: array of shape (n_samples, n_features)
- y: array of shape (n_samples,)

<a id="svm.baseSVM.BaseSVM._fit"></a>

#### \_fit

```python
def _fit(X, y)
```

Abstract method to be implemented by subclasses for training.

Parameters:
    X (array-like of shape (n_samples, n_features)): Training vectors.
    y (array-like of shape (n_samples,)): Target values.

Raises:
    NotImplementedError: If the method is not overridden by subclasses.

<a id="svm.baseSVM.BaseSVM._compute_kernel"></a>

#### \_compute\_kernel

```python
def _compute_kernel(X1, X2)
```

Compute the kernel function between X1 and X2.

Parameters:
- X1: array of shape (n_samples1, n_features)
- X2: array of shape (n_samples2, n_features)

Returns:
- Kernel matrix of shape (n_samples1, n_samples2)

<a id="svm.baseSVM.BaseSVM.decision_function"></a>

#### decision\_function

```python
def decision_function(X)
```

Compute the decision function for input samples.

Parameters:
- X: array of shape (n_samples, n_features)

Returns:
- Decision values of shape (n_samples,)

<a id="svm.baseSVM.BaseSVM.predict"></a>

#### predict

```python
def predict(X)
```

Predict class labels for input samples.

Parameters:
- X: array of shape (n_samples, n_features)

Returns:
- Predicted labels of shape (n_samples,)

<a id="svm.baseSVM.BaseSVM.score"></a>

#### score

```python
def score(X, y)
```

Compute the mean accuracy of the model on the given test data.

Parameters:
    X (array-like of shape (n_samples, n_features)): Test samples.
    y (array-like of shape (n_samples,)): True class labels.

Returns:
    score (float): Mean accuracy of predictions.

Raises:
    NotImplementedError: If the method is not overridden by subclasses.

<a id="svm.baseSVM.BaseSVM.get_params"></a>

#### get\_params

```python
def get_params(deep=True)
```

Get the hyperparameters of the model.

Parameters:
    deep (bool, default=True): If True, returns parameters of subobjects as well.

Returns:
    params (dict): Dictionary of hyperparameter names and values.

<a id="svm.baseSVM.BaseSVM.set_params"></a>

#### set\_params

```python
def set_params(**parameters)
```

Set hyperparameters of the model.

Parameters:
    **parameters (dict): Hyperparameter names and values.

Returns:
    self (BaseSVM): The updated estimator instance.

<a id="svm.baseSVM.BaseSVM.__sklearn_is_fitted__"></a>

#### \_\_sklearn\_is\_fitted\_\_

```python
def __sklearn_is_fitted__()
```

Check if the model has been fitted. For compatibility with sklearn.

Returns:
    fitted (bool): True if the model has been fitted, otherwise False.

<a id="svm.generalizedSVM"></a>

# svm.generalizedSVM

<a id="svm.generalizedSVM.GeneralizedSVR"></a>

## GeneralizedSVR Objects

```python
class GeneralizedSVR(BaseSVM)
```

<a id="svm.generalizedSVM.GeneralizedSVR.__init__"></a>

#### \_\_init\_\_

```python
def __init__(C=1.0,
             tol=1e-4,
             max_iter=1000,
             learning_rate=0.01,
             epsilon=0.1,
             kernel='linear',
             degree=3,
             gamma='scale',
             coef0=0.0)
```

<a id="svm.generalizedSVM.GeneralizedSVR._fit"></a>

#### \_fit

```python
def _fit(X, y)
```

Fit the SVR model using gradient descent with support for multiple kernels.

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

Compute raw decision function values.

<a id="svm.generalizedSVM.GeneralizedSVR.score"></a>

#### score

```python
def score(X, y)
```

Compute the coefficient of determination (R▓ score).

<a id="svm.generalizedSVM.GeneralizedSVC"></a>

## GeneralizedSVC Objects

```python
class GeneralizedSVC(BaseSVM)
```

<a id="svm.generalizedSVM.GeneralizedSVC.__init__"></a>

#### \_\_init\_\_

```python
def __init__(C=1.0,
             tol=1e-4,
             max_iter=1000,
             learning_rate=0.01,
             kernel='linear',
             degree=3,
             gamma='scale',
             coef0=0.0)
```

<a id="svm.generalizedSVM.GeneralizedSVC._fit"></a>

#### \_fit

```python
def _fit(X, y)
```

Fit the SVC model using gradient descent with support for multiple kernels.

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

Compute raw decision function values.

<a id="svm.linerarSVM"></a>

# svm.linerarSVM

<a id="svm.linerarSVM.LinearSVC"></a>

## LinearSVC Objects

```python
class LinearSVC(BaseSVM)
```

<a id="svm.linerarSVM.LinearSVC.__init__"></a>

#### \_\_init\_\_

```python
def __init__(C=1.0, tol=1e-4, max_iter=1000, learning_rate=0.01)
```

<a id="svm.linerarSVM.LinearSVC._fit"></a>

#### \_fit

```python
def _fit(X, y)
```

Implement the fitting procedure for LinearSVC using gradient descent.

Parameters:
    X (array-like of shape (n_samples, n_features)): Training vectors.
    y (array-like of shape (n_samples,)): Target labels in {-1, 1}.

Returns:
    self (LinearSVC): The fitted instance.

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

Parameters:
    X (array-like of shape (n_samples, n_features)): Input samples.

Returns:
    y_pred (array of shape (n_samples,)): Predicted class labels {-1, 1}.

<a id="svm.linerarSVM.LinearSVC._predict_multiclass"></a>

#### \_predict\_multiclass

```python
def _predict_multiclass(X)
```

Predict class labels for multi-class classification using one-vs-rest strategy.

Parameters:
    X (array-like of shape (n_samples, n_features)): Input samples.

Returns:
    predicted_labels (array of shape (n_samples,)): Predicted class labels.

<a id="svm.linerarSVM.LinearSVC.decision_function"></a>

#### decision\_function

```python
def decision_function(X)
```

Compute raw decision function values before thresholding.

Parameters:
    X (array-like of shape (n_samples, n_features)): Input samples.

Returns:
    scores (array of shape (n_samples,)): Decision function values.

<a id="svm.linerarSVM.LinearSVC._score_binary"></a>

#### \_score\_binary

```python
def _score_binary(X, y)
```

Compute the mean accuracy of predictions for binary classification.

Parameters:
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

Parameters:
    X (array-like of shape (n_samples, n_features)): Test samples.
    y (array-like of shape (n_samples,)): True labels.

Returns:
    score (float): Mean accuracy of predictions.

<a id="svm.linerarSVM.LinearSVR"></a>

## LinearSVR Objects

```python
class LinearSVR(BaseSVM)
```

<a id="svm.linerarSVM.LinearSVR.__init__"></a>

#### \_\_init\_\_

```python
def __init__(C=1.0, tol=1e-4, max_iter=1000, learning_rate=0.01, epsilon=0.1)
```

<a id="svm.linerarSVM.LinearSVR._fit"></a>

#### \_fit

```python
def _fit(X, y)
```

Implement the fitting procedure for LinearSVR using the epsilon-insensitive loss.

Parameters:
    X (array-like of shape (n_samples, n_features)): Training vectors.
    y (array-like of shape (n_samples,)): Target values.

Returns:
    self (LinearSVR): The fitted instance.

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

Parameters:
    X (array-like of shape (n_samples, n_features)): Input samples.

Returns:
    y_pred (array of shape (n_samples,)): Predicted values.

<a id="svm.linerarSVM.LinearSVR.decision_function"></a>

#### decision\_function

```python
def decision_function(X)
```

Compute raw decision function values.

Parameters:
    X (array-like of shape (n_samples, n_features)): Input samples.

Returns:
    scores (array of shape (n_samples,)): Predicted values.

<a id="svm.linerarSVM.LinearSVR.score"></a>

#### score

```python
def score(X, y)
```

Compute the coefficient of determination (R▓ score).

Parameters:
    X (array-like of shape (n_samples, n_features)): Test samples.
    y (array-like of shape (n_samples,)): True target values.

Returns:
    score (float): R▓ score of predictions.

<a id="svm.oneClassSVM"></a>

# svm.oneClassSVM

<a id="svm.oneClassSVM.OneClassSVM"></a>

## OneClassSVM Objects

```python
class OneClassSVM(BaseSVM)
```

<a id="svm.oneClassSVM.OneClassSVM.__init__"></a>

#### \_\_init\_\_

```python
def __init__(C=1.0,
             tol=1e-4,
             max_iter=1000,
             learning_rate=0.01,
             kernel='linear',
             degree=3,
             gamma='scale',
             coef0=0.0)
```

<a id="svm.oneClassSVM.OneClassSVM._fit"></a>

#### \_fit

```python
def _fit(X, y=None)
```

Fit the OneClassSVM model using gradient descent for anomaly detection.

Parameters:
    X (array-like of shape (n_samples, n_features)): Training vectors.
    y (array-like of shape (n_samples,)): Target values (ignored).

Returns:
    self (OneClassSVM): The fitted instance.

Algorithm:
    - Initialize weights w and bias b.
    - Use gradient descent to minimize the One-Class SVM objective:
      (1/2) ||w||^2 + b + C * sum(max(0, -(w^T x_i + b))).
    - Update w and b based on subgradients.
    - Stop when gradients are below tolerance or max iterations reached.

<a id="svm.oneClassSVM.OneClassSVM.decision_function"></a>

#### decision\_function

```python
def decision_function(X)
```

<a id="svm.oneClassSVM.OneClassSVM.predict"></a>

#### predict

```python
def predict(X)
```

<a id="svm.oneClassSVM.OneClassSVM.score"></a>

#### score

```python
def score(X, y)
```

Compute the mean accuracy of predictions.

Parameters:
    X (array-like of shape (n_samples, n_features)): Test samples.
    y (array-like of shape (n_samples,)): True labels (+1 for inliers, -1 for outliers).

Returns:
    score (float): Mean accuracy of predictions.

<a id="svm.oneClassSVM.OneClassSVM.__sklearn_is_fitted__"></a>

#### \_\_sklearn\_is\_fitted\_\_

```python
def __sklearn_is_fitted__()
```

Check if the model has been fitted. For compatibility with sklearn.

Returns:
    fitted (bool): True if the model has been fitted, otherwise False.

<a id="trees"></a>

# trees

<a id="trees.__all__"></a>

#### \_\_all\_\_

<a id="trees.gradientBoostedRegressor"></a>

# trees.gradientBoostedRegressor

<a id="trees.gradientBoostedRegressor.GradientBoostedRegressor"></a>

## GradientBoostedRegressor Objects

```python
class GradientBoostedRegressor(object)
```

A class to represent a Gradient Boosted Decision Tree Regressor.

Attributes:
    random_seed (int): The random seed for the random number generator.
    num_trees (int): The number of decision trees in the ensemble.
    max_depth (int): The maximum depth of each decision tree.
    display (bool): A flag to display the decision tree.
    X (list): A list of input data features.
    y (list): A list of target values.
    XX (list): A list of input data features and target values.
    numerical_cols (set): A set of indices of numeric attributes (columns).

Methods:
    __init__(file_loc, num_trees=5, random_seed=0, max_depth=10): Initializes the GBDT object.
    reset(): Resets the GBDT object.
    fit(): Fits the GBDT model to the training data.
    predict(): Predicts the target values for the input data.
    get_stats(y_predicted): Calculates various evaluation metrics for the predicted target values.

<a id="trees.gradientBoostedRegressor.GradientBoostedRegressor.__init__"></a>

#### \_\_init\_\_

```python
def __init__(X=None,
             y=None,
             num_trees: int = 10,
             max_depth: int = 10,
             random_seed: int = 0)
```

<a id="trees.gradientBoostedRegressor.GradientBoostedRegressor.reset"></a>

#### reset

```python
def reset()
```

<a id="trees.gradientBoostedRegressor.GradientBoostedRegressor.fit"></a>

#### fit

```python
def fit(X=None, y=None, stats=False)
```

Fits the gradient boosted decision tree regressor to the training data.

This method trains the ensemble of decision trees by iteratively fitting each tree to the residuals
of the previous iteration. The residuals are updated after each iteration by subtracting the predictions
made by the current tree from the target values.

Args:
    X (numpy.ndarray): An array of input data features. Default is None.
    y (numpy.ndarray): An array of target values. Default is None.
    stats (bool): A flag to decide whether to return stats or not. Default is False.

Returns:
    None

<a id="trees.gradientBoostedRegressor.GradientBoostedRegressor.predict"></a>

#### predict

```python
def predict(X=None)
```

Predicts the target values for the input data using the gradient boosted decision tree regressor.

Parameters:
- X (numpy.ndarray): An array of input data features. Default is None.

Returns:
    predictions (numpy.ndarray): An array of predicted target values for the input data.

<a id="trees.gradientBoostedRegressor.GradientBoostedRegressor.get_stats"></a>

#### get\_stats

```python
def get_stats(y_predicted)
```

Calculates various evaluation metrics for the predicted target values.

Args:
    y_predicted (numpy.ndarray): An array of predicted target values.

Returns:
    dict: A dictionary containing the evaluation metrics.
        - MSE (float): Mean Squared Error
        - R^2 (float): R-squared Score
        - MAPE (float): Mean Absolute Percentage Error
        - MAE (float): Mean Absolute Error
        - RMSE (float): Root Mean Squared Error

<a id="trees.isolationForest"></a>

# trees.isolationForest

<a id="trees.isolationForest.IsolationUtils"></a>

## IsolationUtils Objects

```python
class IsolationUtils(object)
```

Utility functions for the Isolation Forest algorithm.

<a id="trees.isolationForest.IsolationUtils.compute_avg_path_length"></a>

#### compute\_avg\_path\_length

```python
@staticmethod
def compute_avg_path_length(size)
```

Computes the average path length of unsuccessful searches in a binary search tree.

Parameters:
- size (int): The size of the tree.

Returns:
- float: The average path length.

<a id="trees.isolationForest.IsolationTree"></a>

## IsolationTree Objects

```python
class IsolationTree(object)
```

An isolation tree for anomaly detection.

<a id="trees.isolationForest.IsolationTree.__init__"></a>

#### \_\_init\_\_

```python
def __init__(max_depth=10, force_true_length=False)
```

<a id="trees.isolationForest.IsolationTree.fit"></a>

#### fit

```python
def fit(X, depth=0)
```

Fits the isolation tree to the data.

Parameters:
- X (array-like): The input features.
- depth (int): The current depth of the tree (default: 0).

Returns:
- dict: The learned isolation tree.

<a id="trees.isolationForest.IsolationTree.path_length"></a>

#### path\_length

```python
def path_length(X, tree=None, depth=0)
```

Computes the path length for a given sample.

Parameters:
- X (array-like): The input sample.
- tree (dict): The current node of the tree (default: None).
- depth (int): The current depth of the tree (default: 0).

Returns:
- int: The path length.

<a id="trees.isolationForest.IsolationForest"></a>

## IsolationForest Objects

```python
class IsolationForest(object)
```

Isolation Forest for anomaly detection.

<a id="trees.isolationForest.IsolationForest.__init__"></a>

#### \_\_init\_\_

```python
def __init__(n_trees=100,
             max_samples=None,
             max_depth=10,
             n_jobs=1,
             force_true_length=False)
```

<a id="trees.isolationForest.IsolationForest.fit"></a>

#### fit

```python
def fit(X)
```

Fits the isolation forest to the data.

Parameters:
- X (array-like): The input features.

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

Parameters:
- X (array-like): The input samples.

Returns:
- array: An array of anomaly scores.

<a id="trees.isolationForest.IsolationForest.predict"></a>

#### predict

```python
def predict(X, threshold=0.5)
```

Predicts whether samples are anomalies.

Parameters:
- X (array-like): The input samples.
- threshold (float): The threshold for classifying anomalies (default: 0.5).

Returns:
- array: An array of predictions (1 if the sample is an anomaly, 0 otherwise).

<a id="trees.isolationForest.IsolationForest.__sklearn_is_fitted__"></a>

#### \_\_sklearn\_is\_fitted\_\_

```python
def __sklearn_is_fitted__()
```

<a id="trees.randomForestClassifier"></a>

# trees.randomForestClassifier

This module contains the implementation of a Random Forest Classifier.

The module includes the following classes:
- RandomForest: A class representing a Random Forest model.
- RandomForestWithInfoGain: A class representing a Random Forest model that returns information gain for vis.
- runRandomForest: A class that runs the Random Forest algorithm.

<a id="trees.randomForestClassifier.RandomForestClassifier"></a>

## RandomForestClassifier Objects

```python
class RandomForestClassifier(object)
```

Random Forest classifier.

Attributes:
    num_trees (int): The number of decision trees in the random forest.
    decision_trees (list): List of decision trees in the random forest.
    bootstraps_datasets (list): List of bootstrapped datasets for each tree.
    bootstraps_labels (list): List of true class labels corresponding to records in the bootstrapped datasets.
    max_depth (int): The maximum depth of each decision tree.

Methods:
    __init__(self, num_trees, max_depth): Initializes the RandomForest object.
    _reset(self): Resets the RandomForest object.
    _bootstrapping(self, XX, n): Performs bootstrapping on the dataset.
    bootstrapping(self, XX): Initializes the bootstrapped datasets for each tree.
    fitting(self): Fits the decision trees to the bootstrapped datasets.
    voting(self, X): Performs voting to classify the input records.

<a id="trees.randomForestClassifier.RandomForestClassifier.num_trees"></a>

#### num\_trees

Number of decision trees in the random forest

<a id="trees.randomForestClassifier.RandomForestClassifier.decision_trees"></a>

#### decision\_trees

List of decision trees in the random forest

<a id="trees.randomForestClassifier.RandomForestClassifier.bootstraps_datasets"></a>

#### bootstraps\_datasets

List of bootstrapped datasets for each tree

<a id="trees.randomForestClassifier.RandomForestClassifier.bootstraps_labels"></a>

#### bootstraps\_labels

List of true class labels corresponding to records in the bootstrapped datasets

<a id="trees.randomForestClassifier.RandomForestClassifier.max_depth"></a>

#### max\_depth

Maximum depth of each decision tree

<a id="trees.randomForestClassifier.RandomForestClassifier.random_seed"></a>

#### random\_seed

Random seed for reproducibility

<a id="trees.randomForestClassifier.RandomForestClassifier.forest_size"></a>

#### forest\_size

Number of trees in the random forest

<a id="trees.randomForestClassifier.RandomForestClassifier.max_depth"></a>

#### max\_depth

Maximum depth of each decision tree

<a id="trees.randomForestClassifier.RandomForestClassifier.display"></a>

#### display

Flag to display additional information about info gain

<a id="trees.randomForestClassifier.RandomForestClassifier.X"></a>

#### X

Data features

<a id="trees.randomForestClassifier.RandomForestClassifier.y"></a>

#### y

Data labels

<a id="trees.randomForestClassifier.RandomForestClassifier.XX"></a>

#### XX

Contains both data features and data labels

<a id="trees.randomForestClassifier.RandomForestClassifier.numerical_cols"></a>

#### numerical\_cols

Number of numeric attributes (columns)

<a id="trees.randomForestClassifier.RandomForestClassifier.__init__"></a>

#### \_\_init\_\_

```python
def __init__(X=None,
             y=None,
             max_depth=5,
             forest_size=5,
             display=False,
             random_seed=0)
```

Initializes the RandomForest object.

Args:
    num_trees (int): The number of decision trees in the random forest.
    max_depth (int): The maximum depth of each decision tree.

<a id="trees.randomForestClassifier.RandomForestClassifier.reset"></a>

#### reset

```python
def reset()
```

Resets the random forest object.

<a id="trees.randomForestClassifier.RandomForestClassifier._bootstrapping"></a>

#### \_bootstrapping

```python
def _bootstrapping(XX, n)
```

Performs bootstrapping on the dataset.

Args:
    XX (list): The dataset.
    n (int): The number of samples to be selected.

Returns:
    tuple: A tuple containing the bootstrapped dataset and the corresponding labels.

<a id="trees.randomForestClassifier.RandomForestClassifier.bootstrapping"></a>

#### bootstrapping

```python
def bootstrapping(XX)
```

Initializes the bootstrapped datasets for each tree.

Args:
    XX (list): The dataset.

<a id="trees.randomForestClassifier.RandomForestClassifier.fitting"></a>

#### fitting

```python
def fitting()
```

Fits the decision trees to the bootstrapped datasets.

<a id="trees.randomForestClassifier.RandomForestClassifier.voting"></a>

#### voting

```python
def voting(X)
```

Performs voting to classify the input records.

Args:
    X (list): The input records.

Returns:
    list: The predicted class labels for the input records.

<a id="trees.randomForestClassifier.RandomForestClassifier.fit"></a>

#### fit

```python
def fit(X=None, y=None, verbose=False)
```

Runs the random forest algorithm.

Returns:
    tuple: A tuple containing the random forest object and the accuracy of the random forest algorithm.

Raises:
    FileNotFoundError: If the file specified by file_loc does not exist.

<a id="trees.randomForestClassifier.RandomForestClassifier.display_info_gains"></a>

#### display\_info\_gains

```python
def display_info_gains()
```

Displays the information gains of each decision tree.

<a id="trees.randomForestClassifier.RandomForestClassifier.plot_info_gains_together"></a>

#### plot\_info\_gains\_together

```python
def plot_info_gains_together()
```

Plots the information gains of all decision trees together.

<a id="trees.randomForestClassifier.RandomForestClassifier.plot_info_gains"></a>

#### plot\_info\_gains

```python
def plot_info_gains()
```

Plots the information gain of each decision tree separately.

<a id="trees.randomForestClassifier.RandomForestClassifier.predict"></a>

#### predict

```python
def predict(X)
```

Predicts the class labels for the input records.

Args:
    X (list): The input records.

Returns:
    list: The predicted class labels for the input records.

<a id="trees.randomForestRegressor"></a>

# trees.randomForestRegressor

This module contains the implementation of a Random Forest Regressor.

The module includes the following classes:
- RandomForest: A class representing a Random Forest model.
- runRandomForest: A class that runs the Random Forest algorithm.

<a id="trees.randomForestRegressor.RandomForestRegressor"></a>

## RandomForestRegressor Objects

```python
class RandomForestRegressor(object)
```

A class representing a Random Forest model.

Attributes:
    num_trees (int): The number of decision trees in the random forest.
    decision_trees (list): A list of decision trees in the random forest.
    bootstraps_datasets (list): A list of bootstrapped datasets for each tree.
    bootstraps_labels (list): A list of corresponding labels for each bootstrapped dataset.
    max_depth (int): The maximum depth of each decision tree.

Methods:
    __init__(num_trees, max_depth): Initializes the RandomForest object.
    _bootstrapping(XX, n): Performs bootstrapping on the dataset.
    bootstrapping(XX): Initializes the bootstrapped datasets for each tree.
    fitting(): Fits the decision trees to the bootstrapped datasets.
    voting(X): Performs voting to predict the target values for the input records.
    user(): Returns the user's GTUsername.

<a id="trees.randomForestRegressor.RandomForestRegressor.num_trees"></a>

#### num\_trees

Number of decision trees in the random forest

<a id="trees.randomForestRegressor.RandomForestRegressor.decision_trees"></a>

#### decision\_trees

List of decision trees in the random forest

<a id="trees.randomForestRegressor.RandomForestRegressor.bootstraps_datasets"></a>

#### bootstraps\_datasets

List of bootstrapped datasets for each tree

<a id="trees.randomForestRegressor.RandomForestRegressor.bootstraps_labels"></a>

#### bootstraps\_labels

List of true class labels corresponding to records in the bootstrapped datasets

<a id="trees.randomForestRegressor.RandomForestRegressor.max_depth"></a>

#### max\_depth

Maximum depth of each decision tree

<a id="trees.randomForestRegressor.RandomForestRegressor.random_seed"></a>

#### random\_seed

Random seed for reproducibility

<a id="trees.randomForestRegressor.RandomForestRegressor.forest_size"></a>

#### forest\_size

Number of trees in the random forest

<a id="trees.randomForestRegressor.RandomForestRegressor.max_depth"></a>

#### max\_depth

Maximum depth of each decision tree

<a id="trees.randomForestRegressor.RandomForestRegressor.display"></a>

#### display

Flag to display additional information about info gain

<a id="trees.randomForestRegressor.RandomForestRegressor.X"></a>

#### X

Data features

<a id="trees.randomForestRegressor.RandomForestRegressor.y"></a>

#### y

Data labels

<a id="trees.randomForestRegressor.RandomForestRegressor.XX"></a>

#### XX

Contains both data features and data labels

<a id="trees.randomForestRegressor.RandomForestRegressor.numerical_cols"></a>

#### numerical\_cols

Number of numeric attributes (columns)

<a id="trees.randomForestRegressor.RandomForestRegressor.__init__"></a>

#### \_\_init\_\_

```python
def __init__(X=None, y=None, forest_size=10, random_seed=0, max_depth=10)
```

Initializes the RandomForest object.

Args:
    X (ndarray): The input data features.
    y (ndarray): The target values.
    forest_size (int): The number of decision trees in the random forest.
    random_seed (int): The random seed for reproducibility.
    max_depth (int): The maximum depth of each decision tree.

<a id="trees.randomForestRegressor.RandomForestRegressor.reset"></a>

#### reset

```python
def reset()
```

Resets the random forest object.

<a id="trees.randomForestRegressor.RandomForestRegressor._bootstrapping"></a>

#### \_bootstrapping

```python
def _bootstrapping(XX, n)
```

Performs bootstrapping on the dataset.

Args:
    XX (list): The dataset.
    n (int): The number of samples to be selected.

Returns:
    tuple: A tuple containing the bootstrapped dataset and the corresponding labels.

<a id="trees.randomForestRegressor.RandomForestRegressor.bootstrapping"></a>

#### bootstrapping

```python
def bootstrapping(XX)
```

Initializes the bootstrapped datasets for each tree.

Args:
    XX (list): The dataset.

<a id="trees.randomForestRegressor.RandomForestRegressor.fitting"></a>

#### fitting

```python
def fitting()
```

Fits the decision trees to the bootstrapped datasets.

<a id="trees.randomForestRegressor.RandomForestRegressor.voting"></a>

#### voting

```python
def voting(X)
```

Performs voting to predict the target values for the input records.

Args:
    X (list): The input records.

Returns:
    list: The predicted target values for the input records.

<a id="trees.randomForestRegressor.RandomForestRegressor.fit"></a>

#### fit

```python
def fit(X=None, y=None, verbose=False)
```

Runs the random forest algorithm.

<a id="trees.randomForestRegressor.RandomForestRegressor.get_stats"></a>

#### get\_stats

```python
def get_stats(verbose=True)
```

Returns the evaluation metrics.

<a id="trees.randomForestRegressor.RandomForestRegressor.predict"></a>

#### predict

```python
def predict(X=None)
```

Predicts the target values for the input data.

<a id="trees.treeClassifier"></a>

# trees.treeClassifier

<a id="trees.treeClassifier.ClassifierTreeUtility"></a>

## ClassifierTreeUtility Objects

```python
class ClassifierTreeUtility(object)
```

Utility class for computing entropy, partitioning classes, and calculating information gain.

<a id="trees.treeClassifier.ClassifierTreeUtility.entropy"></a>

#### entropy

```python
def entropy(class_y)
```

Computes the entropy for a given class.

Parameters:
- class_y (array-like): The class labels.

Returns:
- float: The entropy value.

<a id="trees.treeClassifier.ClassifierTreeUtility.partition_classes"></a>

#### partition\_classes

```python
def partition_classes(X, y, split_attribute, split_val)
```

Partitions the dataset into two subsets based on a given split attribute and value.

Parameters:
- X (array-like): The input features.
- y (array-like): The target labels.
- split_attribute (int): The index of the attribute to split on.
- split_val (float): The value to split the attribute on.

Returns:
- X_left  (array-like): The subset of input features where the split attribute is less than or equal to the split value.
- X_right (array-like): The subset of input features where the split attribute is greater than the split value.
- y_left  (array-like): The subset of target labels corresponding to X_left.
- y_right (array-like): The subset of target labels corresponding to X_right.

<a id="trees.treeClassifier.ClassifierTreeUtility.information_gain"></a>

#### information\_gain

```python
def information_gain(previous_y, current_y)
```

Calculates the information gain between the previous and current values of y.

Parameters:
- previous_y (array-like): The previous values of y.
- current_y (array-like): The current values of y.

Returns:
- float: The information gain between the previous and current values of y.

<a id="trees.treeClassifier.ClassifierTreeUtility.best_split"></a>

#### best\_split

```python
def best_split(X, y)
```

Finds the best attribute and value to split the data based on information gain.

Parameters:
- X (array-like): The input features.
- y (array-like): The target variable.

Returns:
- dict: A dictionary containing the best split attribute, split value, left and right subsets of X and y,
        and the information gain achieved by the split.

<a id="trees.treeClassifier.ClassifierTree"></a>

## ClassifierTree Objects

```python
class ClassifierTree(object)
```

A class representing a decision tree.

Parameters:
- max_depth (int): The maximum depth of the decision tree.

Methods:
- learn(X, y, par_node={}, depth=0): Builds the decision tree based on the given training data.
- classify(record): Classifies a record using the decision tree.

<a id="trees.treeClassifier.ClassifierTree.__init__"></a>

#### \_\_init\_\_

```python
def __init__(max_depth=5)
```

<a id="trees.treeClassifier.ClassifierTree.learn"></a>

#### learn

```python
def learn(X, y, par_node={}, depth=0)
```

Builds the decision tree based on the given training data.

Parameters:
- X (array-like): The input features.
- y (array-like): The target labels.
- par_node (dict): The parent node of the current subtree (default: {}).
- depth (int): The current depth of the subtree (default: 0).

Returns:
- dict: The learned decision tree.

<a id="trees.treeClassifier.ClassifierTree.classify"></a>

#### classify

```python
def classify(record)
```

Classifies a given record using the decision tree.

Parameters:
- record: A dictionary representing the record to be classified.

Returns:
- The label assigned to the record based on the decision tree.

<a id="trees.treeRegressor"></a>

# trees.treeRegressor

<a id="trees.treeRegressor.RegressorTreeUtility"></a>

## RegressorTreeUtility Objects

```python
class RegressorTreeUtility(object)
```

Utility class for computing variance, partitioning classes, and calculating information gain.

<a id="trees.treeRegressor.RegressorTreeUtility.calculate_variance"></a>

#### calculate\_variance

```python
def calculate_variance(y)
```

Calculate the variance of a dataset.
Variance is used as the measure of impurity in the case of regression.

<a id="trees.treeRegressor.RegressorTreeUtility.partition_classes"></a>

#### partition\_classes

```python
def partition_classes(X, y, split_attribute, split_val)
```

Partitions the dataset into two subsets based on a given split attribute and value.

Parameters:
- X (array-like): The input features.
- y (array-like): The target labels.
- split_attribute (int): The index of the attribute to split on.
- split_val (float): The value to split the attribute on.

Returns:
- X_left (array-like): The subset of input features where the split attribute is less than or equal to the split value.
- X_right (array-like): The subset of input features where the split attribute is greater than the split value.
- y_left (array-like): The subset of target labels corresponding to X_left.
- y_right (array-like): The subset of target labels corresponding to X_right.

<a id="trees.treeRegressor.RegressorTreeUtility.information_gain"></a>

#### information\_gain

```python
def information_gain(previous_y, current_y)
```

Calculate the information gain from a split by subtracting the variance of
child nodes from the variance of the parent node.

<a id="trees.treeRegressor.RegressorTreeUtility.best_split"></a>

#### best\_split

```python
def best_split(X, y)
```

Finds the best attribute and value to split the data based on information gain.

Parameters:
- X (array-like): The input features.
- y (array-like): The target variable.

Returns:
- dict: A dictionary containing the best split attribute, split value, left and right subsets of X and y,
        and the information gain achieved by the split.

<a id="trees.treeRegressor.RegressorTree"></a>

## RegressorTree Objects

```python
class RegressorTree(object)
```

A class representing a decision tree for regression.

Parameters:
- max_depth (int): The maximum depth of the decision tree.

Methods:
- learn(X, y, par_node={}, depth=0): Builds the decision tree based on the given training data.
- classify(record): Predicts the target value for a record using the decision tree.

<a id="trees.treeRegressor.RegressorTree.__init__"></a>

#### \_\_init\_\_

```python
def __init__(max_depth=5)
```

<a id="trees.treeRegressor.RegressorTree.learn"></a>

#### learn

```python
def learn(X, y, par_node={}, depth=0)
```

Builds the decision tree based on the given training data.

Parameters:
- X (array-like): The input features.
- y (array-like): The target labels.
- par_node (dict): The parent node of the current subtree (default: {}).
- depth (int): The current depth of the subtree (default: 0).

Returns:
- dict: The learned decision tree.

<a id="trees.treeRegressor.RegressorTree.predict"></a>

#### predict

```python
def predict(record)
```

Predicts a given record using the decision tree.

Parameters:
- record: A dictionary representing the record to be classified.

Returns:
- Returns the mean of the target values.

<a id="utils"></a>

# utils

<a id="utils.__all__"></a>

#### \_\_all\_\_

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

Check the balance of classes in the given array.
Parameters:
    y (array-like): Array of class labels.
Returns:
    tuple: A tuple containing the number of unique classes and an array of counts for each class.

<a id="utils.dataAugmentation._Utils.separate_samples"></a>

#### separate\_samples

```python
@staticmethod
def separate_samples(X, y)
```

Separates samples based on their class labels.
Parameters:
    X (numpy.ndarray): The input data samples.
    y (numpy.ndarray): The class labels corresponding to the input data samples.
Returns:
    dict: A dictionary where the keys are unique class labels and the values are arrays of samples belonging to each class.

<a id="utils.dataAugmentation._Utils.get_class_distribution"></a>

#### get\_class\_distribution

```python
@staticmethod
def get_class_distribution(y)
```

Get the distribution of classes in the given array.
Parameters:
    y (array-like): Array of class labels.
Returns:
    dict: A dictionary where the keys are unique class labels and the values are their respective counts.

<a id="utils.dataAugmentation._Utils.get_minority_majority_classes"></a>

#### get\_minority\_majority\_classes

```python
@staticmethod
def get_minority_majority_classes(y)
```

Get the minority and majority classes from the given array.
Parameters:
    y (array-like): Array of class labels.
Returns:
    tuple: A tuple containing the minority class and the majority class.

<a id="utils.dataAugmentation._Utils.validate_Xy"></a>

#### validate\_Xy

```python
@staticmethod
def validate_Xy(X, y)
```

Validate the input data and labels.
Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
Raises:
    ValueError: If the shapes of X and y do not match.

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

<a id="utils.dataAugmentation.SMOTE.fit_resample"></a>

#### fit\_resample

```python
def fit_resample(X, y, force_equal=False)
```

Resample the dataset to balance the class distribution by generating synthetic samples.
Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    force_equal (bool): If True, resample until classes are equal. Default is False.
Returns:
    X_resampled (array-like): Resampled feature matrix.
    y_resampled (array-like): Resampled target vector.

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

<a id="utils.dataAugmentation.RandomOverSampler.fit_resample"></a>

#### fit\_resample

```python
def fit_resample(X, y)
```

Resample the dataset to balance the class distribution by duplicating minority class samples.
Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
Returns:
    X_resampled (array-like): Resampled feature matrix.
    y_resampled (array-like): Resampled target vector.

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

<a id="utils.dataAugmentation.RandomUnderSampler.fit_resample"></a>

#### fit\_resample

```python
def fit_resample(X, y)
```

Resample the dataset to balance the class distribution by removing majority class samples.
Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
Returns:
    X_resampled (array-like): Resampled feature matrix.
    y_resampled (array-like): Resampled target vector.

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

<a id="utils.dataAugmentation.Augmenter.augment"></a>

#### augment

```python
def augment(X, y)
```

<a id="utils.dataPrep"></a>

# utils.dataPrep

<a id="utils.dataPrep.DataPrep"></a>

## DataPrep Objects

```python
class DataPrep(object)
```

A class for preparing data for machine learning models.

<a id="utils.dataPrep.DataPrep.one_hot_encode"></a>

#### one\_hot\_encode

```python
def one_hot_encode(data, cols)
```

One-hot encodes non-numerical columns in a DataFrame or numpy array.
Drops the original columns after encoding.

Parameters:
- data (pandas.DataFrame or numpy.ndarray): The data to be encoded.
- cols (list): The list of column indices to be encoded.

Returns:
- data (pandas.DataFrame or numpy.ndarray): The data with one-hot encoded columns.

<a id="utils.dataPrep.DataPrep.find_categorical_columns"></a>

#### find\_categorical\_columns

```python
def find_categorical_columns(data)
```

Finds the indices of non-numerical columns in a DataFrame or numpy array.

Parameters:
- data (pandas.DataFrame or numpy.ndarray): The data to be checked.

Returns:
- categorical_cols (list): The list of indices of non-numerical columns.

<a id="utils.dataPrep.DataPrep.write_data"></a>

#### write\_data

```python
def write_data(df, csv_file, print_path=False)
```

Writes the DataFrame to a CSV file.

Parameters:
- df (pandas.DataFrame): The DataFrame to be written.
- csv_file (str): The path of the CSV file to write to.

<a id="utils.dataPrep.DataPrep.prepare_data"></a>

#### prepare\_data

```python
def prepare_data(csv_file,
                 label_col_index,
                 cols_to_encode=[],
                 write_to_csv=True)
```

Prepares the data by loading a CSV file, one-hot encoding non-numerical columns,
and optionally writing the prepared data to a new CSV file.

Parameters:
- csv_file (str): The path of the CSV file to load.
- label_col_index (int): The index of the label column.
- cols_to_encode (list): The list of column indices to one-hot encode. Default is an empty list.
- write_to_csv (bool): Whether to write the prepared data to a new CSV file. Default is True.

Returns:
- df (pandas.DataFrame): The prepared DataFrame.
- prepared_csv_file (str): The path of the prepared CSV file. If write_to_csv is False, returns "N/A".

<a id="utils.dataPrep.DataPrep.df_to_ndarray"></a>

#### df\_to\_ndarray

```python
def df_to_ndarray(df, y_col=0)
```

Converts a DataFrame to a NumPy array.

Parameters:
- df (pandas.DataFrame): The DataFrame to be converted.
- y_col (int): The index of the label column. Default is 0.

Returns:
- X (numpy.ndarray): The feature columns as a NumPy array.
- y (numpy.ndarray): The label column as a NumPy array.

<a id="utils.dataPrep.DataPrep.k_split"></a>

#### k\_split

```python
def k_split(X, y, k=5)
```

Splits the data into k folds for cross-validation.

Parameters:
- X (numpy.ndarray): The feature columns.
- y (numpy.ndarray): The label column.
- k (int): The number of folds. Default is 5.

Returns:
- X_folds (list): A list of k folds of feature columns.
- y_folds (list): A list of k folds of label columns.

<a id="utils.dataPreprocessing"></a>

# utils.dataPreprocessing

<a id="utils.dataPreprocessing.one_hot_encode"></a>

#### one\_hot\_encode

```python
def one_hot_encode(X, cols=None)
```

One-hot encodes non-numerical columns in a DataFrame or numpy array.
Drops the original columns after encoding.

Parameters:
    - X (pandas.DataFrame or numpy.ndarray): The data to be encoded.
    - cols (list): The list of column indices to be encoded. Defaults to None, which means all non-numerical columns will be encoded.
        If None, the function will automatically detect non-numerical columns.

Returns:
    - X (pandas.DataFrame or numpy.ndarray): The data with one-hot encoded columns.

<a id="utils.dataPreprocessing._find_categorical_columns"></a>

#### \_find\_categorical\_columns

```python
def _find_categorical_columns(X)
```

Finds the indices of non-numerical columns in a DataFrame or numpy array.

Parameters:
- X (pandas.DataFrame or numpy.ndarray): The data to be checked.

Returns:
- categorical_cols (list): The list of indices of non-numerical columns.

<a id="utils.dataPreprocessing.normalize"></a>

#### normalize

```python
def normalize(X, norm='l2')
```

Normalizes the input data using the specified norm.
Uses L2 normalization by default.

Parameters:
    - X (numpy.ndarray): The input data to be normalized.
    - norm (str): The type of norm to use for normalization. Default is 'l2'.
        - 'l2': L2 normalization (Euclidean norm).
        - 'l1': L1 normalization (Manhattan norm).
        - 'max': Max normalization (divides by the maximum absolute value).
        - 'minmax': Min-max normalization (scales to [0, 1]).

Returns:
    - X (numpy.ndarray): The normalized data.

<a id="utils.dataPreprocessing.Scaler"></a>

## Scaler Objects

```python
class Scaler()
```

A class for scaling data by standardization and normalization.

<a id="utils.dataPreprocessing.Scaler.__init__"></a>

#### \_\_init\_\_

```python
def __init__(method='standard')
```

Initializes the scaler with the specified method.

Parameters:
    - method (str): The scaling method to use. Options are 'standard', 'minmax', or 'normalize'.

<a id="utils.dataPreprocessing.Scaler.fit"></a>

#### fit

```python
def fit(X)
```

Fits the scaler to the data.

Parameters:
    - X (numpy.ndarray): The data to fit the scaler to.

<a id="utils.dataPreprocessing.Scaler.transform"></a>

#### transform

```python
def transform(X)
```

Transforms the data using the fitted scaler.

Parameters:
    - X (numpy.ndarray): The data to transform.

Returns:
    - X_transformed (numpy.ndarray): The transformed data.

<a id="utils.dataPreprocessing.Scaler.fit_transform"></a>

#### fit\_transform

```python
def fit_transform(X)
```

Fits the scaler to the data and then transforms it.

Parameters:
    - X (numpy.ndarray): The data to fit and transform.

Returns:
    - X_transformed (numpy.ndarray): The transformed data.

<a id="utils.dataPreprocessing.Scaler.inverse_transform"></a>

#### inverse\_transform

```python
def inverse_transform(X)
```

Inverse transforms the data using the fitted scaler.

Parameters:
    - X (numpy.ndarray): The data to inverse transform.

Returns:
    - X_inverse (numpy.ndarray): The inverse transformed data.

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

Returns
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

Principal Component Analysis (PCA) implementation.
Uses the eigendecomposition of the covariance matrix to project the data onto a lower-dimensional space.

Parameters:
- n_components: number of principal components to keep

<a id="utils.decomposition.PCA.fit"></a>

#### fit

```python
def fit(X)
```

Fit the PCA model to the data X.

Parameters:
- X: numpy array of shape (n_samples, n_features)
     The data to fit the model to.

<a id="utils.decomposition.PCA.transform"></a>

#### transform

```python
def transform(X)
```

Apply the dimensionality reduction on X.

Parameters:
- X: numpy array of shape (n_samples, n_features)
     The data to transform.

Returns:
- X_transformed: numpy array of shape (n_samples, n_components)
                 The data transformed into the principal component space.

<a id="utils.decomposition.PCA.fit_transform"></a>

#### fit\_transform

```python
def fit_transform(X)
```

Fit the PCA model to the data X and apply the dimensionality reduction on X.

Parameters:
- X: numpy array of shape (n_samples, n_features)
     The data to fit the model to and transform.

Returns:
- X_transformed: numpy array of shape (n_samples, n_components)
                 The data transformed into the principal component space.

<a id="utils.decomposition.PCA.get_explained_variance_ratio"></a>

#### get\_explained\_variance\_ratio

```python
def get_explained_variance_ratio()
```

<a id="utils.decomposition.PCA.get_explained_variance_ratio"></a>

#### get\_explained\_variance\_ratio

```python
def get_explained_variance_ratio()
```

<a id="utils.decomposition.PCA.get_components"></a>

#### get\_components

```python
def get_components()
```

<a id="utils.decomposition.PCA.inverse_transform"></a>

#### inverse\_transform

```python
def inverse_transform(X_reduced)
```

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

Singular Value Decomposition (SVD) implementation.

Parameters:
- n_components: number of singular values and vectors to keep

<a id="utils.decomposition.SVD.fit"></a>

#### fit

```python
def fit(X)
```

Fit the SVD model to the data X.

Parameters:
- X: numpy array of shape (n_samples, n_features)
     The data to fit the model to.

<a id="utils.decomposition.SVD.transform"></a>

#### transform

```python
def transform(X)
```

Apply the SVD transformation on X.

Parameters:
- X: numpy array of shape (n_samples, n_features)
     The data to transform.

Returns:
- X_transformed: numpy array of shape (n_samples, n_components)
                 The data transformed into the singular value space.

<a id="utils.decomposition.SVD.fit_transform"></a>

#### fit\_transform

```python
def fit_transform(X)
```

Fit the SVD model to the data X and apply the SVD transformation on X.

Parameters:
- X: numpy array of shape (n_samples, n_features)
     The data to fit the model to and transform.

Returns:
- X_transformed: numpy array of shape (n_samples, n_components)
                 The data transformed into the singular value space.

<a id="utils.decomposition.SVD.get_singular_values"></a>

#### get\_singular\_values

```python
def get_singular_values()
```

<a id="utils.decomposition.SVD.get_singular_vectors"></a>

#### get\_singular\_vectors

```python
def get_singular_vectors()
```

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

Generate a random regression problem.

Parameters
----------
n_samples : int, default=100
    The number of samples.
n_features : int, default=100
    The number of features.
n_informative : int, default=10
    The number of informative features, i.e., the number of features used
    to build the linear model used to generate the output.
n_targets : int, default=1
    The number of regression targets, i.e., the dimension of the y output.
bias : float, default=0.0
    The bias term in the underlying linear model.
effective_rank : int or None, default=None
    If not None, the approximate dimension of the data matrix.
tail_strength : float, default=0.5
    The relative importance of the fat noisy tail of the singular values
    profile if `effective_rank` is not None.
noise : float, default=0.0
    The standard deviation o    f the gaussian noise applied to the output.
shuffle : bool, default=True
    Whether to shuffle the samples and the features.
coef : bool, default=False
    If True, the coefficients of the underlying linear model are returned.
random_state : int or None, default=None
    Random seed.

Returns
-------
X : ndarray of shape (n_samples, n_features)
    The input samples.
y : ndarray of shape (n_samples,) or (n_samples, n_targets)
    The output values.
coef : ndarray of shape (n_features,) or (n_features, n_targets)
    The coefficient of the underlying linear model. Only returned if
    coef=True.

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

Generate a random n-class classification problem.

Parameters
----------
n_samples : int, default=100
    The number of samples.
n_features : int, default=20
    The total number of features.
n_informative : int, default=2
    The number of informative features.
n_redundant : int, default=2
    The number of redundant features.
n_repeated : int, default=0
    The number of duplicated features.
n_classes : int, default=2
    The number of classes (or labels) of the classification problem.
n_clusters_per_class : int, default=2
    The number of clusters per class.
weights : array-like of shape (n_classes,) or None, default=None
    The proportions of samples assigned to each class.
flip_y : float, default=0.01
    The fraction of samples whose class is randomly exchanged.
class_sep : float, default=1.0
    The factor multiplying the hypercube size.
hypercube : bool, default=True
    If True, the clusters are put on the vertices of a hypercube.
shift : float, default=0.0
    Shift features by the specified value.
scale : float, default=1.0
    Multiply features by the specified value.
shuffle : bool, default=True
    Shuffle the samples and the features.
random_state : int or None, default=None
    Random seed.

Returns
-------
X : ndarray of shape (n_samples, n_features)
    The generated samples.
y : ndarray of shape (n_samples,)
    The integer labels for class membership of each sample.

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

Generate isotropic Gaussian blobs for clustering.

Parameters
----------
n_samples : int or array-like, default=100
    If int, it is the total number of samples.
    If array-like, it contains the number of samples per cluster.
n_features : int, default=2
    The number of features.
centers : int or array-like of shape (n_centers, n_features), default=None
    The number of centers to generate, or the fixed center locations.
    If n_samples is an int and centers is None, 3 centers are generated.
    If n_samples is array-like, centers must be
    either None or an array of length equal to the length of n_samples.
cluster_std : float or array-like of shape (n_centers,), default=1.0
    The standard deviation of the clusters.
center_box : tuple of float (min, max), default=(-10.0, 10.0)
    The bounding box for each cluster center when centers are
    generated at random.
shuffle : bool, default=True
    Shuffle the samples.
random_state : int or None, default=None
    Random seed.

Returns
-------
X : ndarray of shape (n_samples, n_features)
    The generated samples.
y : ndarray of shape (n_samples,)
    The integer labels for cluster membership of each sample.
centers : ndarray of shape (n_centers, n_features)
    The centers of each cluster. Only returned if return_centers is True.

<a id="utils.metrics"></a>

# utils.metrics

<a id="utils.metrics.Metrics"></a>

## Metrics Objects

```python
class Metrics(object)
```

<a id="utils.metrics.Metrics.mean_squared_error"></a>

#### mean\_squared\_error

```python
@classmethod
def mean_squared_error(cls, y_true, y_pred)
```

Calculates the mean squared error between the true and predicted values.

Parameters:
- y_true (numpy.ndarray): The true values.
- y_pred (numpy.ndarray): The predicted values.

Returns:
- mse (float): The mean squared error.

<a id="utils.metrics.Metrics.r_squared"></a>

#### r\_squared

```python
@classmethod
def r_squared(cls, y_true, y_pred)
```

Calculates the R-squared score between the true and predicted values.

Parameters:
- y_true (numpy.ndarray): The true values.
- y_pred (numpy.ndarray): The predicted values.

Returns:
- r_squared (float): The R-squared score.

<a id="utils.metrics.Metrics.mean_absolute_error"></a>

#### mean\_absolute\_error

```python
@classmethod
def mean_absolute_error(cls, y_true, y_pred)
```

Calculates the mean absolute error between the true and predicted values.

Parameters:
- y_true (numpy.ndarray): The true values.
- y_pred (numpy.ndarray): The predicted values.

Returns:
- mae (float): The mean absolute error.

<a id="utils.metrics.Metrics.root_mean_squared_error"></a>

#### root\_mean\_squared\_error

```python
@classmethod
def root_mean_squared_error(cls, y_true, y_pred)
```

Calculates the root mean squared error between the true and predicted values.

Parameters:
- y_true (numpy.ndarray): The true values.
- y_pred (numpy.ndarray): The predicted values.

Returns:
- rmse (float): The root mean squared error.

<a id="utils.metrics.Metrics.mean_absolute_percentage_error"></a>

#### mean\_absolute\_percentage\_error

```python
@classmethod
def mean_absolute_percentage_error(cls, y_true, y_pred)
```

Calculates the mean absolute percentage error between the true and predicted values.

Parameters:
- y_true (numpy.ndarray): The true values.
- y_pred (numpy.ndarray): The predicted values.

Returns:
- mape (float): The mean absolute percentage error as a decimal. Returns np.nan if y_true is all zeros.

<a id="utils.metrics.Metrics.mean_percentage_error"></a>

#### mean\_percentage\_error

```python
@classmethod
def mean_percentage_error(cls, y_true, y_pred)
```

Calculates the mean percentage error between the true and predicted values.

Parameters:
- y_true (numpy.ndarray): The true values.
- y_pred (numpy.ndarray): The predicted values.

Returns:
- mpe (float): The mean percentage error.

<a id="utils.metrics.Metrics.accuracy"></a>

#### accuracy

```python
@classmethod
def accuracy(cls, y_true, y_pred)
```

Calculates the accuracy score between the true and predicted values.

Parameters:
- y_true (numpy.ndarray): The true values.
- y_pred (numpy.ndarray): The predicted values.

Returns:
- accuracy (float): The accuracy score.

<a id="utils.metrics.Metrics.precision"></a>

#### precision

```python
@classmethod
def precision(cls, y_true, y_pred)
```

Calculates the precision score between the true and predicted values.

Parameters:
- y_true (numpy.ndarray): The true values.
- y_pred (numpy.ndarray): The predicted values.

Returns:
- precision (float): The precision score.

<a id="utils.metrics.Metrics.recall"></a>

#### recall

```python
@classmethod
def recall(cls, y_true, y_pred)
```

Calculates the recall score between the true and predicted values.

Parameters:
- y_true (numpy.ndarray): The true values.
- y_pred (numpy.ndarray): The predicted values.

Returns:
- recall (float): The recall score.

<a id="utils.metrics.Metrics.f1_score"></a>

#### f1\_score

```python
@classmethod
def f1_score(cls, y_true, y_pred)
```

Calculates the F1 score between the true and predicted values.

Parameters:
- y_true (numpy.ndarray): The true values.
- y_pred (numpy.ndarray): The predicted values.

Returns:
- f1_score (float): The F1 score.

<a id="utils.metrics.Metrics.log_loss"></a>

#### log\_loss

```python
@classmethod
def log_loss(cls, y_true, y_pred)
```

Calculates the log loss between the true and predicted values.

Parameters:
- y_true (numpy.ndarray): The true values.
- y_pred (numpy.ndarray): The predicted probabilities.

Returns:
- log_loss (float): The log loss.

<a id="utils.metrics.Metrics.confusion_matrix"></a>

#### confusion\_matrix

```python
@classmethod
def confusion_matrix(cls, y_true, y_pred)
```

Calculates the confusion matrix between the true and predicted values.

Parameters:
- y_true (numpy.ndarray): The true values.
- y_pred (numpy.ndarray): The predicted values.

Returns:
- cm (numpy.ndarray): The confusion matrix.

<a id="utils.metrics.Metrics.show_confusion_matrix"></a>

#### show\_confusion\_matrix

```python
@classmethod
def show_confusion_matrix(cls, y_true, y_pred)
```

Calculates and displays the confusion matrix between the true and predicted values.

Parameters:
- y_true (numpy.ndarray): The true values.
- y_pred (numpy.ndarray): The predicted values.

Returns:
- cm (numpy.ndarray): The confusion matrix.

<a id="utils.metrics.Metrics.classification_report"></a>

#### classification\_report

```python
@classmethod
def classification_report(cls, y_true, y_pred)
```

Generates a classification report for the true and predicted values.

Parameters:
- y_true (numpy.ndarray): The true values.
- y_pred (numpy.ndarray): The predicted values.

Returns:
- report (dict): The classification report.

<a id="utils.metrics.Metrics.show_classification_report"></a>

#### show\_classification\_report

```python
@classmethod
def show_classification_report(cls, y_true, y_pred)
```

Generates and displays a classification report for the true and predicted values.

Parameters:
- y_true (numpy.ndarray): The true values.
- y_pred (numpy.ndarray): The predicted values.

Returns:
- report (dict): The classification report.

<a id="utils.modelSelection"></a>

# utils.modelSelection

<a id="utils.modelSelection.ModelSelectionUtility"></a>

## ModelSelectionUtility Objects

```python
class ModelSelectionUtility()
```

<a id="utils.modelSelection.ModelSelectionUtility.get_param_combinations"></a>

#### get\_param\_combinations

```python
@staticmethod
def get_param_combinations(param_grid)
```

Generates all possible combinations of hyperparameters.

Returns:
- param_combinations (list): A list of dictionaries containing hyperparameter combinations.

<a id="utils.modelSelection.ModelSelectionUtility.cross_validate"></a>

#### cross\_validate

```python
@staticmethod
def cross_validate(model,
                   X,
                   y,
                   params,
                   cv=5,
                   metric='mse',
                   direction='minimize',
                   verbose=False)
```

Implements a custom cross-validation for hyperparameter tuning.

Parameters:
- model: The model Object to be tuned.
- X (numpy.ndarray): The feature columns.
- y (numpy.ndarray): The label column.
- params (dict): The hyperparameters to be tuned.
- cv (int): The number of folds for cross-validation. Default is 5.
- metric (str): The metric to be used for evaluation. Default is 'mse'.
    - Regression Metrics: 'mse', 'r2', 'mae', 'rmse', 'mape', 'mpe'
    - Classification Metrics: 'accuracy', 'precision', 'recall', 'f1', 'log_loss'
- direction (str): The direction to optimize the metric. Default is 'minimize'.
- verbose (bool): A flag to display the training progress. Default is False.

Returns:
- tuple: A tuple containing the scores (list) and the trained model.

<a id="utils.modelSelection.GridSearchCV"></a>

## GridSearchCV Objects

```python
class GridSearchCV(object)
```

Implements a grid search cross-validation for hyperparameter tuning.

<a id="utils.modelSelection.GridSearchCV.__init__"></a>

#### \_\_init\_\_

```python
def __init__(model, param_grid, cv=5, metric='mse', direction='minimize')
```

Initializes the GridSearchCV object.

Parameters:
- model: The model Object to be tuned.
- param_grid (list): A list of dictionaries containing hyperparameters to be tuned.
- cv (int): The number of folds for cross-validation. Default is 5.
- metric (str): The metric to be used for evaluation. Default is 'mse'.
    - Regression Metrics: 'mse', 'r2', 'mae', 'rmse', 'mape', 'mpe'
    - Classification Metrics: 'accuracy', 'precision', 'recall', 'f1', 'log_loss'
- direction (str): The direction to optimize the metric. Default is 'minimize'.

<a id="utils.modelSelection.GridSearchCV.fit"></a>

#### fit

```python
def fit(X, y, verbose=False)
```

Fits the model to the data for all hyperparameter combinations.

Parameters:
- X (numpy.ndarray): The feature columns.
- y (numpy.ndarray): The label column.
- verbose (bool): A flag to display the training progress. Default is True.

Returns:
- model: The best model with the optimal hyperparameters.

<a id="utils.modelSelection.RandomSearchCV"></a>

## RandomSearchCV Objects

```python
class RandomSearchCV(object)
```

Implements a random search cross-validation for hyperparameter tuning.

<a id="utils.modelSelection.RandomSearchCV.__init__"></a>

#### \_\_init\_\_

```python
def __init__(model,
             param_grid,
             iter=10,
             cv=5,
             metric='mse',
             direction='minimize')
```

Initializes the RandomSearchCV object.

Parameters:
- model: The model Object to be tuned.
- param_grid (list): A list of dictionaries containing hyperparameters to be tuned.
- iter (int): The number of iterations for random search. Default is 10.
- cv (int): The number of folds for cross-validation. Default is 5.
- metric (str): The metric to be used for evaluation. Default is 'mse'.
    - Regression Metrics: 'mse', 'r2', 'mae', 'rmse', 'mape', 'mpe'
    - Classification Metrics: 'accuracy', 'precision', 'recall', 'f1', 'log_loss'
- direction (str): The direction to optimize the metric. Default is 'minimize'.

<a id="utils.modelSelection.RandomSearchCV.fit"></a>

#### fit

```python
def fit(X, y, verbose=False)
```

Fits the model to the data for iter random hyperparameter combinations.

Parameters:
- X (numpy.ndarray): The feature columns.
- y (numpy.ndarray): The label column.
- verbose (bool): A flag to display the training progress. Default is True.

Returns:
- model: The best model with the optimal hyperparameters.

<a id="utils.modelSelection.segaSearchCV"></a>

## segaSearchCV Objects

```python
class segaSearchCV(object)
```

Implements a custom search cross-validation for hyperparameter tuning.

<a id="utils.modelSelection.segaSearchCV.__init__"></a>

#### \_\_init\_\_

```python
def __init__(model,
             param_space,
             iter=10,
             cv=5,
             metric='mse',
             direction='minimize')
```

Initializes the segaSearchCV object.

Parameters:
- model: The model Object to be tuned.
- param_space (list): A list of dictionaries containing hyperparameters to be tuned. 
    Should be in the format: [{'param': [type, min, max]}, ...]
- iter (int): The number of iterations for random search. Default is 10.
- cv (int): The number of folds for cross-validation. Default is 5.
- metric (str): The metric to be used for evaluation. Default is 'mse'.
    - Regression Metrics: 'mse', 'r2', 'mae', 'rmse', 'mape', 'mpe'
    - Classification Metrics: 'accuracy', 'precision', 'recall', 'f1', 'log_loss'
- direction (str): The direction to optimize the metric. Default is 'minimize'.

<a id="utils.modelSelection.segaSearchCV.fit"></a>

#### fit

```python
def fit(X, y, verbose=False)
```

Fits the model to the data for iter random hyperparameter combinations.

Parameters:
- X (numpy.ndarray): The feature columns.
- y (numpy.ndarray): The label column.
- verbose (bool): A flag to display the training progress. Default is True.

<a id="utils.polynomialTransform"></a>

# utils.polynomialTransform

<a id="utils.polynomialTransform.PolynomialTransform"></a>

## PolynomialTransform Objects

```python
class PolynomialTransform(object)
```

This class implements Polynomial Feature Transformation.
Polynomial feature transformation is a technique used to create new features from the existing features by raising them to a power or by creating interaction terms.

Parameters:
- degree: int, default=2
    The degree of the polynomial features.

Attributes:
- n_samples: int
    The number of samples.
- n_features: int
    The number of features.
- n_output_features: int
    The number of output features.
- combinations: list of tuples of shape (n_features,)
    The combinations of features(X) of degree n.
- bias: bool, default=True
    Whether to include a bias term in the output features.

<a id="utils.polynomialTransform.PolynomialTransform.__init__"></a>

#### \_\_init\_\_

```python
def __init__(degree=2)
```

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
class VotingRegressor(object)
```

Implements a voting regressor.
Takes a list of fitted models and their weights and returns a weighted average of the predictions.

<a id="utils.voting.VotingRegressor.__init__"></a>

#### \_\_init\_\_

```python
def __init__(models, model_weights=None)
```

Initialize the VotingRegressor object.

Parameters:
- models: list of models to be stacked
- model_weights: list of weights for each model. Default is None.

<a id="utils.voting.VotingRegressor.predict"></a>

#### predict

```python
def predict(X)
```

Predict the target variable using the fitted models.

Parameters:
- X: input features

Returns:
- y_pred: predicted target variable

<a id="utils.voting.VotingRegressor.get_params"></a>

#### get\_params

```python
def get_params()
```

Get the parameters of the VotingRegressor object.

Returns:
- params: dictionary of parameters

<a id="utils.voting.VotingRegressor.show_models"></a>

#### show\_models

```python
def show_models(formula=False)
```

Print the models and their weights.

