`   `
[WIP]
Pre-trained networks
-----------------



Few-shot learning
-----------------
Motivated by the failure of conventional deep learning methods to work well on one or few examples per class, and the
close resemblance to how humans actually learn, there has a resurgence of interest in one/few shot learning. One way to 
think about this approach is that the model effectively trains to be a good learning algorithm, whereby with only few 
training examples, the predictor can generalize to complete new tasks. 

For the particular task at hand -whale classification-, we want to learn a function that embeds examples
belonging to the same class close together while keeping embeddings from separate classes far apart.

To achieve this goal, I use triplet loss detailed in [here](https://arxiv.org/abs/1703.07737). 
[Triplet generator](https://github.com/dzorlu/humpback_whales/blob/master/data/triplet_generator.py) generates triplets 
to pass on the model architecture by passing P classes and K images of each class. 
For this particular problem, because many classes only contain a single image, the generator creates augmented images 
to produce K images in total if there aren't enough images avaiable for a given class.

I have found that fine-tuning the MobileNet works the best. I also implemented [CLR](https://arxiv.org/abs/1506.01186), 
but the performance gain was minimal. 

Embedding Space and Neighbors Approach
--------------------------------------
The few-shot learning approach mentioned above embeds each image on a 128-dimensional vector space. 
At inference time, the model embeds the test images but we still need to find the closest cluster/class
 the image belongs to. Here, I use a simple k-neigbors classifier. 


Ensembles
---------
On top of the MobileNet / K-neigbors classifier structure, I also trained an optimization-based method, 
[Reptile](https://arxiv.org/abs/1803.02999). The idea here is to find an initialization for the parameters such that
when we optimize these parameters at test time, learning is fast - the model generalizes from a small number of examples
 from the test task. The module is trained with 5-shot 10-class classification tasks at training time. At test time,
 the trained model takes the top 10 predictions of the first module, and re-ranks them at test time through one-shot learning.
 With the ensemble approach, I was able to get a better score. 
 

Last, an excellent tutorial on meta-learning can be found [here](https://lilianweng.github.io/lil-log/2018/11/30/meta-learning.html).
