Handson ML with scikit learn notes PART 1 


## Gradient Descent



The general idea of Gradient Descent is to tweak parameters iteratively in order to minimize a cost function.

When using Gradient Descent, you should ensure that all features have a similar scale (e.g., using Scikit-Learn's StandardScaler class), or else it will take much longer to converge.

Batch GD involves calculations over the full training set X, at each Gradient Descent step! This is why the algorithm is called Batch Gradient Descent: it uses the whole batch of training data at every step

Stochastic Gradient Descent just picks a random instance in the training set at every step and computes the gradients based only on that single instance.

Stochastic Gradient Descent has a better chance of finding the global minimum than Batch Gradient Descent does

The function that determines the learning rate at each iteration is called the learning schedule

Mini batch GD computes the gradients on small random sets of instances called minibatches.



## Polynomial regression



We can actually use a linear model to fit nonlinear data. A simple way to do this is to add powers of each feature as new features, then train a linear model on this extended set of features. This technique is called Polynomial Regression.


Learning curves

Learning curves: these are plots of the model's performance on the training set and the validation set as a function of the training set size (or the training iteration).

## The Bias/Variance Tradeoff

Bias : This part of the generalization error is due to wrong assumptions, such as assuming that the data is linear when it is actually quadratic. A high-bias model is most likely to underfit the training data.

Variance : This part is due to the model's excessive sensitivity to small variations in the training data. A model with many degrees of freedom (such as a high-degree polynomial model) is likely to have high variance, and thus to overfit the training data.

Irreducible error : This part is due to the noisiness of the data itself. The only way to reduce this part of the error is to clean up the data (e.g., fix the data sources, such as broken sensors, or detect and remove outliers).

Increasing a model's complexity will typically increase its variance and reduce its bias.

Conversely, reducing a model's complexity increases its bias and reduces its variance. This is why it is called a tradeoff.



## Ridge regression



A regularization term is added to the cost function.

This forces the learning algorithm to not only fit the data but also keep the model

weights as small as possible



## Lasso(Least Absolute Shrinkage and Selection Operator) Regression (L1 norm)


An important characteristic of Lasso Regression is that it tends to completely eliminate the weights of the least important features (i.e., set them to zero).



## Elastic Net



Elastic Net is a middle ground between Ridge Regression and Lasso Regression. The

regularization term is a simple mix of both Ridge and Lasso's regularization terms,

and you can control the mix ratio r. When r = 0, Elastic Net is equivalent to Ridge

Regression, and when r = 1, it is equivalent to Lasso Regression.



It is almost always preferable to have at least a little bit of

regularization, so generally you should avoid plain Linear Regression.

Ridge is a good default but if you suspect that only a few features are actually useful, you should prefer Lasso or Elastic Net since they tend to reduce the useless features eights down to

zero as we have discussed. In general, Elastic Net is preferred over Lasso since Lasso

may behave erratically when the number of features is greater than the number of

training instances or when several features are strongly correlated.




It is preferred to have at least a little bit of regularization, so generally we should avoid plain regression.

Ridge regression is good default.

If we want to use only few features we can use Lasso or Elastic net.

Elastic Net is preferred over lasso because lasso may behave erratically when no.of features > no.of training instances



## Logistic Regression



It computes a weighted sum of the input features (plus a bias term), but instead

of outputting the result directly like the Linear Regression model does, it outputs the

logistic of this result.



## Softmax Regression



The Logistic Regression model can be generalized to support multiple classes directly, without having to train and combine multiple binary classifiers. This is called Softmax Regression, or Multinomial Logistic Regression

Just like the Logistic Regression classifier, the Softmax Regression classifier predicts

the class with the highest estimated probability







#  Chapter 5



## Support vector machine



SVM classification

 You can think of an SVM classifier as fitting the widest possible street (represented by the parallel dashed lines) between the classes. This is called large margin classification.

Adding more training instances “off the street” will not affect the decision boundary at all: it is fully determined (or "supported") by the instances located on the edge of the street. These instances are called the support vectors.

SVMs are sensitive to the feature scales.



Hard margin classification

If we strictly impose that all instances be off the street and on the right side, this is

called hard margin classification.

There are two main issues with hard margin classification. First, it only works if the data is linearly separable, and second it is quite sensitive to outliers.



Soft margin classification

The objective is to find a good balance between keeping the street as large as possible and limiting the margin violations (i.e., instances that end up in the middle of the street or even on the wrong side). This is called soft margin classification.

In Scikit-Learn's SVM classes, you can control this balance using the C hyperparameter: a smaller C value leads to a wider street but more margin violations

If your SVM model is overfitting, you can try regularizing it by reducing C.





Nonlinear SVM classification

One approach to handling nonlinear datasets is to add more features, such as polynomial features (as you did in Chapter 4); in some cases this can result in a linearly separable dataset.



Polynomial kernel

If your model is overfitting, you might want to reduce the polynomial degree. Conversely, if it is underfitting, you can try increasing it. The hyperparameter coef0 controls how much the model is influenced by high degree polynomials versus low-degree polynomials.



Another technique to tackle nonlinear problems is to add features computed using a

similarity function (like Gaussian RBF) that measures how much each instance resembles a particular landmark. But similarity features can be computationally expensive



 Gaussian Radial Basis Function (RBF) Kernel

It is a bell-shaped function varying from 0 (very far away from the landmark) to 1 (at the landmark).

 γ (gamma) acts like a regularization hyperparameter: if your model is overfitting, you should reduce it, and if it is underfitting, you should increase it (similar to the C hyperparameter)

If gamma value is increased, the bell shaped curve becomes narrower. That means the influence of the instance becomes smaller. The decision boundary becomes irregular. For smaller gamma values, the decision boundary becomes smoother.





As a default we could go for the linear svc kernel it works faster for large datasets and if dataset is not too large we can use the RBF kernel.



LinearSVC is based on liblinear library, it takes O(m x n) time complexity. It is very precise but takes long time. It is controlled by tolerance parameter tol.



SVC class supports kernel trick, it has training time complexity between O(m2 x n) and O(m3 x n)



SVM Regression



SVM Regression tries to fit as many instances as possible on the street while limiting margin violations (i.e., instances off the street). The width of the street is controlled by a hyperparameter ϵ.

To tackle nonlinear regression tasks, you can use a kernelized SVM model.

The LinearSVR class scales linearly with the size of the training set (just like the LinearSVC class), while the SVR class gets much too slow when the training set grows large (just like the SVC class).



SVMs can also be used for outlier detection



The decision boundary is the set of points where the decision function is equal to 0

The smaller the weight vector w, the larger the margin.



Hard margin linear SVM classifier objective

If we define t(i) = –1 for negative instances (if y(i) = 0) and t(i) = 1 for positive

instances (if y(i) = 1), then we can express this constraint as t(i)(wT x(i) + b) ≥ 1 for all instances.



we need to:

 minimize(w,b) 1/2(w(T) w)

subject to t(i)(wTx(i)+b)>=1



minimizing that would give us a wider margin





To get the soft margin objective,

we need to introduce a slack variable ζ(i) ≥ 0 for each instance:

ζ(i) measures how much the ith instance is allowed to violate the margin.



To get a trade off between these two objectives, we can use the C hyperparameter.



Softmargin linear SVM objective



minimize 1/2(wTw) + C



minimize   1/2(wT . w) + C∑ζ(i)               i=1 to m

w, b,ζ





## Quadratic programming problems

Convex quadratic optimization problems with linear constraints. Such problems are known as Quadratic Programming (QP) problems.



In these we have to



minimize    1/2(pT Hp) + fTp

     p



subject to Ap<=b





subscripts are used below np means n with subscript p



p is an np dimensional vector (np= number of parameters),

H is an np× np matrix,

f is an np dimensional vector,

A is an nc× np matrix (nc= number of constraints),

b is an nc-dimensional vector



## Kernel



In Machine Learning, a kernel is a function capable of computing the dot product ϕ(a)T ϕ(b) based only on the original vectors a and b, without having to compute (or even to know about) the transformation ϕ.

The function K(a, b) = (aT b)^2 is called a 2nd-degree polynomial kernel.



Different kernels in SVM



Linear : K(a,b) = aTb

Polynomial : K(a,b) = (γaTb+r)^d         d=degree

Gaussian RBF : K(a,b) = exp(−γ || a-b ||^2)

Sigmoid : K(a,b)= tanh(γaTb+r)



Hinge Loss

The function max(0, 1 – t) is called the hinge loss function





# Chapter 6



## Decision Tree



Scikit learn uses CART (Classification And Regression Tree) algorithm which produces trees with parents having only two children ( binary tree) , Other algorithms like ID3 can have more than two children

A Decision Tree can also estimate the probability that an instance belongs to a particular class k: first it traverses the tree to find the leaf node for this instance, and then it returns the ratio of training instances of class k in this node.

CART algorithm : The algorithm first splits the training set in two subsets using a single feature k and a threshold tk.

CART cost function is:

J(k, tk)=m (left) /m . G (left) +m (right)/m . G (right)

 where,

G (left/right) measures the impurity of the left/right subset,

m (left/right) is the number of instances in the left/right subset.



While predicting, Decision Trees are generally approximately balanced, so traversing the Decision Tree requires going through roughly O(log(m)) nodes. So Predictions are very fast even with larger datasets

But for training it compares all features on all samples at each node. This makes the training complexity O(n x mlog(m)).

Gini impurity tends to isolate the most frequent class in its own branch of the tree, while entropy tends to produce slightly more balanced trees. Gini impurity is slightly faster to compute.

Decision Trees can overfit data if unconstrained because the tree structure would adapt itself to fit the training data, fitting it very closely and most likely overfitting it. Such model is called non parametric model.



A non parametric model not because it does not have any parameters (it often has a lot) but because the number of parameters is not determined prior to training, so the model structure is free to stick closely to the data.



A parametric model such as a linear model has a predetermined number of parameters, so its degree of freedom is limited, reducing the risk of overfitting (but increasing the risk of underfitting).



Reducing max\_depth will regularize the model and thus reduce the risk of overfitting.

Increasing min\_\* hyperparameters or reducing max\_\* hyperparameters will regularize the model



Pruning

A node whose children are all leaf nodes is considered unnecessary if the purity improvement it provides is not statistically significant. Standard statistical tests, such as the χ2 test, are used to estimate the probability that the improvement is purely the result of chance (which is called the null hypothesis). If this probability, called the p value, is higher than a given threshold (typically 5%, controlled by a hyperparameter), then the node is considered unnecessary and its children are deleted. The pruning continues until all unnecessary nodes have been pruned.



Decision Tree Regression

This tree would looks similar to the classification tree but here the nodes would contain a value instead of a class.

The algorithm splits each region in a way that makes most training instances as close as possible to that predicted value.

The CART algorithm works mostly the same way as earlier, except that instead of trying to split the training set in a way that minimizes impurity, it now tries to split the training set in a way that minimizes the MSE.

Cost function is:



J(k, tk)=m (left) /m . MSE (left) +m (right)/m . MSE (right)



Decision Trees are sensitive to train set rotation , to overcome that we do PCA

They are sensitive to small variations in data ( high variance)







## Ensemble Learning and Random forests



Ensemble methods : Bagging, Boosting, Stacking

Majority vote classifiers work with great accuracy even if they have weak learners (random guessing classifiers)

Suppose you build an ensemble containing 1,000 classifiers that are individually correct only 51% of the time (barely better than random guessing). If you predict the majority voted class, you can hope for up to 75% accuracy! However, this is only true if all classifiers are perfectly independent, making uncorrelated errors, which is clearly not the case since they are trained on the same data. They are likely to make the same types of errors, so there will be many majority votes for the wrong class, reducing the ensemble's accuracy.

 One way to get diverse classifiers is to train them using very different algorithms



Hard voting is majority voting for a class

Soft voting is majority voting for a class probability ( It usually achieves higher performance)



To select voting preference we can just put voting = "soft" or voting = "hard". But for soft voting, we must use classifiers which can predict class probabilities.



## Bagging and Pasting



The approach is to use the same training algorithm for every predictor, but to train them on different random subsets of the training set. When sampling is performed with replacement, this method is called bagging. If sampling is done without replacement it is called pasting.

Once all predictors are trained, the ensemble can make a prediction for a new instance by simply aggregating the predictions of all predictors. The aggregation function is typically the statistical mode (i.e., the most frequent prediction, just like a hard voting classifier) for classification, or the average for regression.

Bagging scales well and often gives better models.

In code to use bagging we put bootstrap = True, and for pasting we put bootstrap = False. In the BaggingClassifier Hyperparameters.



In bagging some instances may be sampled several times and the others may not be sampled at all. This means the classifiers have not seen those unsampled instances, these are called as Out Of Bag instances. Because of these, we no longer would need to create a separate validation set.

We can just evaluate the ensemble itself by averaging out the Out Of Bag evaluations for each predictor.

In Scikit-Learn we can do this by setting oob\_score = True





## Random Patches and subspaces



Bagging classifier supports sampling features too. It is controlled by two hyperparameters : max\_features and bootstrap\_features

Each predictor would be trained on random subset of input features

Sampling both training instances and features is called the Random Patches method

Keeping all the training instances but sampling features is called Random subspaces method.





## Random forest



Random forest is an ensemble of Decision Trees, generally trained via bagging. (max\_samples = size of train set)

It introduces extra randomness when growing the trees. It searches for the best feature among the random subset of the features.

This results in a greater diversity, which trades higher bias for lower variance. It generally yields an overall better model.

We can make the trees even more random by setting random thresholds for each feature. A forest of such extremely random trees is called Extremely Randomized Trees ensemble.

Training these is faster than regular Random forests.

Scikit-Learn measures a feature's importance by looking at how much the tree nodes that use that feature reduce impurity on average.

Random Forest are very useful in feature selection





## Boosting



In boosting we combine weak learners into a strong learner. The general idea is to train them sequentially, each corrects it's predecessor.

AdaBoost is popular. In this, the predictor corrects the part where the predecessor underfits



Working process of AdaBoost



 A first base classifier (such as a Decision Tree) is trained and used to make predictions on the training set.

The relative weight of misclassified training instances is then increased. (giving more importance for the misclassified instances)

A second classifier is trained using the updated weights

And again it makes predictions on the training set,

Weights are updated, and so on



## AdaBoost Algorithm

Each instance weight w(i) is initially set to 1/m.

Train first predictor and find weighted error rate on training set   r(j) = sum(w (i)) / sum(w (i))   i=1 to m

Find predictor's weight. The more accurate predictor is the higher its weight will be, if it is guessing randomly the value would be close to 0, if it is worse than random guessing the value would be negative.
αj = η log((1-rj)/rj)

The weights are updated
w(i)       if y\_hat(j)(i) = y(i)
w(i) =
w(i)exp(αj)  if y\_hat(j)(i) != y(i)

Finally, a new predictor is trained using the updated weights, and the whole process is repeated

The algorithm stops when the desired number of predictors is reached, or when a perfect predictor is found.





Scikit-Learn uses SAMME, It is like a multi class version of AdaBoost.

When there are just 2 classes SAMME acts as AdaBoost.



## Gradient Boosting

Just like AdaBoost, Gradient Boosting works by sequentially adding predictors to an ensemble, each one correcting its predecessor.

But the only difference is, instead of tweaking the instance weights at every iteration, this method tries to fit new predictor to the residual errors made by the previous predictor

























## Chapter 9



Unsupervised learning techniques



Clustering

Anamoly detection

Density detection



Clustering

The task is to find similar instances and assigning them to clusters.

Applications include : Customer segmentation, Data analysis, Anamoly detection, Semi-supervised learning, search engines, Segmenting an image.

Clusters can be understood by algorithms in different ways : instances around the centroid, Densely packed instances

We use cluster numbers as labels here

Assigning each instance to a single cluster is called hard clustering

Assigning each instance a score per cluster is called soft clustering

Score can be like the distance between instance and centroid, or it can be similarity score.



K means



Working process is :

We start by placing the centroids randomly (e.g., by picking k instances at random and using their locations as centroids). Then label the instances, update the centroids, label the instances, update the centroids, and so on until the centroids stop moving.

K means is generally one of the fastest Clustering algorithms.

Risk in this is that our algorithm may not converge to the right solution, this can be overcome by improving the centroid initialization

One approach is to use n\_init hyperparameter, another approach is to run algorithm many times and keep the best solution.

K means in sklearn uses the performance metric, interia : The mean squared distance between each instance and it's closest centroid. It keeps model with lowest inertia.

K means++ algorithm is proposed, in this one the algorithm finds the centroids farther from each other, so the algorithm is much less likely to converge to a sub optimal solution.

It works by :

* Take one centroid c(1), chosen uniformly at random from the dataset.
* Take a new centroid c(i), choosing an instance x(i) with probability: D(x(i))^2



                       sum(D(x(j))^2         j=1 to m

where D(x(i)) is the distance between the instance x(i) and the closest

centroid that was already chosen.

* This probability distribution ensures that instances further away from already chosen centroids are much more likely be selected as centroids.
* Repeat the previous step until all k centroids have been chosen.
* Sklearn default uses this algorithm
* Another different variants of Kmeans are : Charles Elkan triangle inequality method, MinibatchKmeans
* To find the optimal k number of clusters value, we could use the "silhouette score" : Mean silhouette coefficient over all instances. ( Sihlouette coefficient = (b-a)/max(a,b)) where a is the mean distance to the instances in same cluster and b is the mean distance to the instances of the next closest cluster)
* The silhouette coefficient can be in between -1 and +1

 		- If it is 0, then it means that coefficient is closer to a cluster boundary

 		- If it is near -1, it means that the instance may have been assigned to the wrong  cluster.



* Silhouette diagram helps us in visualizing these more effectively
* Drawbacks for K-means are : It doesn't behave well with datasets of varying sizes, densities and non spherical shapes.
* For elliptical clusters, Gaussian mixture models works great
* Scaling is necessary before running a K means model, or else the clusters may be very stretched and K means would perform poorly.



Image segmentation is the process of partitioning an image into multiple segments.



Semantic segmentation :  In semantic segmentation, all pixels that are part of the same object type get assigned to the same segment.



Instance segmentation : In instance segmentation, all pixels that are part of the same individual object are assigned to the same segment.



Color segmentation : The process of assigning the pixels to the same segment if they have the same color.



Clustering for preprocessing

* It is helpful in dimensionality reduction.
* We can create a pipeline that will first cluster the training set into let's say 50 clusters and replace the images with their distance to these 50 clusters
* The best value of k is simply the one that results in the best classification performance during cross-validation.



Clustering for semi supervised learning



* Used when we have many unlabeled instances and few labeled instances.
* Propagation the labels to all the other instances in the same cluster is called label propagation
* Partial propagation is when we propagate the labels to only the nearest ones in a range lets say 20% of the nearest instances.
* It boosts the accuracy generally



Active learning

* To continue improving the model and our training set, we can do few rounds of active learning.
* One common method is uncertainity sampling :

 		- In this the model is trained on the labeled instances gathered so far, and 				this model is used to make predictions on all the unlabeled instances.

 		- The instances for which the model is most uncertain is manually labelled 				by the expert.

 		- Then we iterate the process again and again, until the performance 			         	improvement stops being worth the labelling effort.





DBSCAN



* This algorithm defines clusters as continuous regions of high density.

Working :

1. For each instance the algorithm counts how many instances are located within a small distance (epsilon) from it. This region is called the instance's ε-neighborhood
2. If an instance has atleast min\_samples instances in its neighborhood then it is considered as core distance. (this means that core instances are the instances which are located in dense regions)
3. All instances in the neighborhood of a core instance belong to the same cluster.
4. Any instance that is not a core instance and doesn't have one in its neighborhood is considered an anamoly



* This algorithm works well if the clusters are dense enough and they are well separated by low-density regions.
* Though DBSCAN can't predict which cluster a new instance belongs to.





Gaussian Mixture models



* SO basically GMMs assumes that the data clusters are generated from the gaussian distribution. 
* The instances which are from same cluster would form an ellipse.
* It also assumes that the parameters are known
* We must know how many clusters are there 
* We try to find the optimal values for mean (mu, the center of our Guassian) , covariance (sigma, controls the spread of the distribution) and pi(probablilty that an instance belongs to a cluster k). 
* We use the expectation maximization method.
* We assign random parameters and repeats the steps until convergence
   1. Assigning instances to the clusters (Expectation step)
   2. Updating the clusters (maximization step)
* It uses soft cluster assignments



Assigning each instance to a cluster = Hard clustering

Assigning the probability that it belongs to a cluster = Soft clustering



* In real world data would be high dimensional, or it could have many clusters, or few instances. Here EM algorithm would struggle to converge, so we need to limit the no.of parameters that the algorithm has to learn
* We can do this by limiting the range of shapes and orientations that clusters can have.(By giving constraints on covariance matrices)
* In code we can just use the "covariance\_type" hyperparameter



* Instances that deviate strongly from the norm are called as outliers, we can detect these by using our GMM
* The outlier is simply the instance located in the low density region
* To select the optimal no.of cluster we use Bayesian Information Creation or Akaike Information creation criterias



BIC = log(m) p - 2 log(L hat)

AIC = 2p - 2 log(L hat)



m = no.of instances, p = no.of parameters, L hat = maximized value of likelihood function of model





* Rather than manually searching for optimal no.of clusters, it is possible to use the "BayesianGaussianMixture" class which can give weights equal to zero ( or close) to unnecessary clusters.
* We just need to set the value of "n\_components" to a value which we believe would be greater than optimal no.of clusters.









