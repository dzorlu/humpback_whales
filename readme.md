
[WIP]
Pre-trained networks
-----------------



Few-shot learning
-----------------
Motivated by the failure of conventional deep learning methods to work well on one or few examples per class, and the
close resemblance to how humans actually learn, there has a resurgence of interest in one/few shot learning.

One way to think about this approach if that the model effectively trains to be a good learning algorithm, where it takes
only few examples and produce a predictor that can be applied to new examples. Due to this analogy, training under the
paradigm is referred to as meta-learning.



For the particular task at hand -whale classification-, we want to learn an embedding function that embeds examples
belonging to the same class close together while keeping embeddings from separate classes far apart (1). The first
implementation is also called a metric-based method. A great detailed post about meta-learning can be found here (3).


To achieve this goal, I use triplet loss detailed in [here](https://arxiv.org/abs/1703.07737). [Triplet generator](https://github.com/dzorlu/humpback_whales/blob/master/data/triplet_generator.py) generates triplets to pass on the
model architecture by passing P classes and K images of each class. Because many classes only contain a single image,
if there aren't enough images avaiable for a class, the generator creates augmented images to produce K images in total.
I chose to exclude `new_whales` class because it is a catch-all class, and the images are not expected to form a
distinctive cluster in the embedding space.

I have found that fine-tuning the MobileNet with 40 frozen layers works the best. I also tried CLR (5) but reducing
the learning rate on a pleatue lishgted edged the performance of the model trained using CLR.

Embedding Space and Neighbors Approach
--------------------------------------
The few-shot learning approach embeds each image on a 128-dimensional vector space. At inference time, the model embeds
the test images but we still need to find the closest cluster/class the image belongs to.



Ensembles
---------

Reptile. Always include `new_whale` class.

