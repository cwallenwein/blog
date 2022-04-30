# Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes

![](https://i.imgur.com/3F8J1mc.gif)
###### Fig. 1: Reconstruction of a transparent object

# Introduction
Obtaining the 3D shape of an object from a set of images is a well-studied problem. The corresponding research field is called Multi-view 3D-reconstruction. Many proposed techniques achieve impressive results but fail to reconstruct transparent objects. Image-based transparent shape reconstruction is an ill-posed problem. Reflection and refraction lead to complex light paths and small changes in shape might lead to completely different appearance. Different solutions to this problem have been proposed, but the acquisition setup is often tedious and requires a complicated setup. In 2020 a group of researchers from the University of California in San Diego state, they have found a technique that enables the reconstruction of transparent objects using only a few unconstrained images taken with a smartphone. This blog post will provide an in-depth look into the paper “Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes” by Zhengqin Li, Yu-Ying Yeh and Manmohan Chandraker[^1].



# Previous work
Transparent object reconstruction has been studied for more than 30 years [^34]. A full history of related research is out of scope. Approaches were typically based on physics, but recently also deep learning researchers have attempted to find a solution. "Through the looking glass" is able to solve this challenging problem by combining the best of both worlds. The following is a brief overview of important work of the last years. A common characteristic of both approaches is the use of a synthetic dataset.

## Physics-based approaches
The foundation of most research is the vast array of knowledge in the field of optics. The authors use this knowledge to formulate their mathematical models and improve them utilizing one manifestation of optimization algorithms. Recent research [^30],[^31] uses synthetic datasets to overcome the hurdle of acquiring enough training data. This comes at the cost of a mismatch between the performance on real-world data compared to synthetic datasets. When testing on real-world data, these approaches typically require a complicated setup including multiple cameras taking images from various fixed viewpoints.
A paper from 2018 by Wu et al. [^31] captures 22 images of the transparent object in front of predefined background patterns. The camera viewpoints are fixed, but the transparent object rotates on a turntable. 

The underlying optics concepts used in physics-based papers are fundamental to understand "Through the looking glass". The section [Overview of optics fundamentals](#Optics-fundamentals) will introduce these topics.

## Deep Learning-based approaches
The setup of deep learning-based approaches is usually simpler. Using RGB [^32]/RGB-D[^33] images of the transparent object, models learn to predict e.g. the segmentation mask, the depth map and surface normals. These models are typically based on Encoder-Decoder CNNs. Deep learning methods inherently need far more data and therefore also leverage synthetic datasets. 



# Important concepts
Prior to introducing the proposed methods, some basic concepts from the fields of optics and 3D graphics shall be clarified. 
## Normals and color mapping
A normal is a vector that is perpendicular to some object. A surface normal is a vector that is perpendicular to a surface and it can be represented like any other vector $[X,Y,Z]$.

A normal map encodes the surface normal for each point in an image. The color at a certain point indicates the direction of the surface at this particular point. The color at each point is described by the three color channels $[R,G,B]$. A normal map simply maps the $[X,Y,Z]$-direction of the surface normals to the color channels.

## Optics fundamentals
![](https://i.imgur.com/CjBbxEq.png)
###### Fig. 2: Superposition of reflected and refracted environment images on a window pane.

Light propagates on straight paths through vacuum, air and homogeneous transparent materials. At the interface of two optically different materials, the propagation changes: In most configurations, a single path is split into two paths. For large angles, all light reflects back into the object and no light refracts. This is called total internal reflection. Snell's law of refraction and the Fresnel equations allow calculating precise angles of reflection and refraction and the fraction of reflected and refracted light. In an image acquisition situation, beam splitting creates superimposed images. The higher the index of refraction (IOR, denoted $\text{n}_1$ and $\text{n}_2$ in Fig. 2), the slower the light travels in the optically dense matter, the stronger the surface reflection and the higher the angles of refraction and the shift of the refracted image. More about these concepts can be found at [^41], [^40] and [^42]. 



# Proposed method
## Problem setup
![](https://i.imgur.com/j50ia0g.png)
###### Fig. 3: Problem setup

The inputs to the model are
* 5-20 unconstrained images of the transparent object,
* the corresponding silhouette segmentation masks for all images,
* the environment map,
* the index of refraction of the transparent material.


![](https://i.imgur.com/1bG0ran.png)
###### Fig. 4: Light path visualization with local surface normals $N^1$ and $N^2$


The proposed method limits the light path simulation to a maximum of two bounces. The camera looks onto the transparent object. 





## Overview
The authors propose the following contributions:
* A physics-based network that estimates the front and back normals for a single viewpoint. It leverages a fully differentiable, also physics-based, rendering layer.
* Physics-based point cloud reconstruction using the predicted normals. 
* A publicly available synthetic dataset [^3] containing 3600 transparent objects. Each object is captured in 35 images from random viewpoints, resulting in a total of 120,000 high-quality images.

The model starts off by initializing a point cloud of the transparent shape. This point cloud is inaccurate but serves as a good starting point for further optimization. In the next step, the physics-based neural network estimates the normal maps $N^1$ and $N^2$ for each viewpoint. The predicted, viewpoint-specific features, will then be mapped onto the point cloud. Finally, the authors use point cloud reconstruction to recover the full geometry of the transparent shape. The model is trained using the synthetic dataset and can be tested on real-world data. The code is publicly available on Github [^2].

## Space carving
Given a set of segmentation masks and their corresponding viewpoint, the space carving algorithm, first introduced by Kutulakos et al. more than 20 years ago, is able to reconstruct a good estimate of the ground truth shape, called visual hull (paper[^50], intuitive video[^51]). Front and back normal maps can be calculated from the visual hull. They will later be referred to by the notation $\tilde{N^1}$, $\tilde{N^2}$, or by the term visual hull initialized normals. These normals already provide a good estimate for the ground truth normals.

## Normal prediction

### Differentiable rendering layer
The model utilizes differentiable rendering to produce high-quality images of transparent objects from a given viewpoint. The renderer is physics-based and uses the Fresnel equations and Snell's law to calculate complex light paths. Differentiable rendering is an exciting, new field that emerged in the last years as it allows for backpropagation through the rendering layer. This video [^52] provides a good introductory overview of the topic.
To render the image from one viewpoint, the differentiable rendering layer requires the environment map $E$ and the estimated normal maps $N^1$, $N^2$ for this particular viewpoint. It outputs the rendered image, a binary mask indicating points where total internal reflection occurred and the pointwise rendering error with masked out environment. The rendering error map is calculated by comparing the rendered image to the ground truth image, cf. Fig. 5.

![](https://i.imgur.com/1HxaGOX.png)
![](https://i.imgur.com/gFGKWKM.png)
![](https://i.imgur.com/Hb7HPv2.png)
###### Fig. 5: Rendered image, total internal reflection mask and rendering error for a transparent object



### Cost volume
The search space to find the correct normal maps $N^1$, $N^2$ is enormous. Each point in the front and back normal map could have a completely different surface normal. As stated before, the visual hull initialized normal maps are a good estimate for the ground truth normal maps. Therefore, the search space will be restricted to normal maps close to the visual hull initialized normals. To further reduce the search space, $K$ normal maps for the front and back surface are randomly sampled around the visual hull initialized normal maps. $K$ normal maps for both $N^1$ and $N^2$ lead to $K \times K$ combinations. According to the authors, $K=4$ gives good results. Higher values for $K$ only increase the computational complexity without improving the quality of the normal estimations. The entire cost volume consists of the front and back normals, the rendering error and the total internal reflection mask, cf. Fig. 6.

![](https://i.imgur.com/TQfjvxk.png)
###### Fig. 6: The cost volume



### Normal prediction network
To estimate the surface normals an encoder-decoder CNN is used. The cost volume is still too large to be fed into the network. Therefore, the authors first use learnable pooling to perform feature extraction on the cost volume. They concatenate the condensed cost volume together with
* the image of the transparent object
* the image with masked out environment
* the visual hull initialized normals
* the total internal reflection mask
* the rendering error

and feed everything to the encoder-decoder CNN. $L_N=\vert N^1 - \hat{N^1}\vert ^2 + \vert N^2 - \hat{N^2}\vert^2$ is used as the loss function. It is simply the $L^2$ distance between the estimated normals ($N^1, N^2$) and the ground truth normals ($\hat{N^1}, \hat{N^2}$).

## Point cloud reconstruction
The normal prediction network gives important information about the transparent object from different viewing angles. But somehow the features in the different views have to be matched with the points in the point cloud. Subsequently, a modified PointNet++ [^54]  will predict the final point locations and final normal vectors for each point in the point cloud. Finally, the 3D point cloud will be transformed into a mesh by applying Poisson surface reconstruction [^53].


### Feature mapping
The goal of feature mapping is to assign features to each point in the initial point cloud. In particular, these features are the normal at that point, the rendering error and the total internal reflection mask. The features are known for each viewpoint but not for the points in the point cloud. A point of the point cloud can be mapped from 3D-space to the 2D point of each viewpoint to retrieve the necessary information. In some cases, a point might not be visible from a particular angle, if so, it will not be taken into account during feature mapping. For each point, there are usually 10 different views and sets of features. It now has to be decided, which view(s) to take into account when creating the feature vectors. The authors try three different feature mapping approaches, see section 3.2 Feature Mapping [^1] for more details. Selecting the view with the lowest rendering error leads to the best results.

### Modified PointNet++
Given the mapped features and the initial point cloud, PointNet++ predicts the final point cloud and the corresponding normals. The authors were able to improve the predictions by modifying the PointNet++ architecture. In particular, they replaced max-pooling with average pooling, passed the front and back normals to all skip connections and applied feature augmentation. The best results are obtained with a chamfer-distance-based loss. The chamfer distance can be used to measure the distance between 2 point clouds. The authors try two other loss functions. For details see Table 2 of the paper [^1].

 
### Poisson surface reconstruction
Poisson surface reconstruction was first introduced in 2006 [^53] and is still used in recent papers. It takes a point cloud and surface normal estimations for each point of the point cloud and reconstructs the 3D mesh of the object.


# Evaluation
## Qualitative results

![](https://i.imgur.com/ELmYXsF.gif)
###### Fig. 7: Qualitative results of Li et al.
At first sight, the quality of the reconstructions looks really good. The
scenes look reasonable and no big differences between ground truth and reconstructions are visible.

## Testing on real-world data
To test the model on real-world data the following is needed:
* images of the transparent object,
* silhouette segmentation masks of all images,
* environment map.

The authors claim that real-world testing can be done by "light-weight mobile phone-based acquisition" [^70]. The images of the object can indeed be captured with commodity hardware like a smartphone camera. COLMAP[^61] is used to determine the viewpoint of the images. However, the segmentation masks were created by the authors manually, and to capture the environment map, a mirror sphere has to be placed at the position of the transparent object.

## Comparison to latest SOTA
Quick reminder: this is deep learning research. It comes as no big surprise, that there is a newer paper [^60], with a different approach, that works better. This newer paper is by Lyu et al. and it's the successor of [^31]. Figure 5 shows the qualitative results of Li et al.[^1] compared to Lyu et al. The left side presents the results of Lyu et al. (1st column) compared with their ground truth (2nd column). On the right side, the results of Li et al. are displayed (ground truth: 3rd column, reconstruction: 4th column).

![](https://i.imgur.com/FA2KbQJ.png)
![](https://i.imgur.com/K4vB6kA.png)
###### Fig. 8: Qualitative comparison between Li et al.[^1] and Lyu et al. [^60]

It is clearly visible that the results of Li et al.[^1] are oversmoothed. There is no space between the hand and the head of the monkey. Additionally, neither the eyes of the dog nor the monkey are visible in the reconstructions. Lyu et al.[^60] on the other side successfully reconstructs the eyes of both animals and clearly separates the hand from the head of the monkey. One possible reason for this oversmoothing is the average pooling in the modified PointNet++. It has to be taken into account, however, that the underlying ground truth in both papers is slightly different and that Li et al.[^1] optimized for easy acquisition of the shape. Lyu et al.[^60] improved their acquisition ease compare to [^31] but is still more restricted than Li et al.[^1]. A quantitative comparison between both papers can be found in Table 1. It displays the reconstruction error in the form of the average per-vertex distance to the corresponding ground truth. Lyu et al. was able to cut the per-vertex distance approximately in half for all tested shapes.

###### Table 1: Reconstruction error of Li et al.[^1] and Lyu et al.[^60] (based on Table 1 of [^60])
|       | initial | Li et al. 2020[^1] | Lyu et al. 2020[^60] |
|------ | ------- | ------------------ | -------------------- |
| Mouse | 0.007164| 0.005840           | <b>0.003075          |
| Dog   | 0.004481| 0.002778           | <b>0.002065          |
| Monkey| 0.005048| 0.004632           | <b>0.002244          |
| Pig   | 0.004980| 0.004741           | <b>0.002696          |



# Conclusion

This paper proposed a novel approach that combined different paths and provides good results, but the reconstruction is not as effortless as claimed.

For the sake of simplicity, some relevant details were left out. In case of questions, read the paper [^1] and the code [^2] or [send me an email](mailto:christian.wallenwein@tum.de). Thank you to Yu-Ying Yeh for clarifying questions about the paper, Eckhard Siegmann for suggestions and proofreading and Pengyuan for tips and advice.



# References

[comment]: <> (1. The paper)

[^1]: Li, Zhengqin, Yu-Ying Yeh, and Manmohan Chandraker. "Through the looking glass: neural 3D reconstruction of transparent shapes." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.

[^2]: "Transparent Shape Dataset", 2020, uploaded by Zhengqin Li, https://github.com/lzqsd/TransparentShapeReconstruction

[^3]: "Transparent Shape Dataset", 2020, uploaded by Zhengqin Li,
["https://github.com/lzqsd/TransparentShapeDataset]("https://github.com/lzqsd/TransparentShapeDataset)

[comment]: <> (2. Introduction)

[comment]: <> (3. Previous work)

[^30]: Qian, Yiming, Minglun Gong, and Yee-Hong Yang. "Stereo-based 3D reconstruction of dynamic fluid surfaces by global optimization." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

[^31]: Wu, Bojian, et al. "Full 3D reconstruction of transparent objects." arXiv preprint arXiv:1805.03482 (2018).

[^32]: Stets, Jonathan, et al. "Single-shot analysis of refractive shape using convolutional neural networks." 2019 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2019.

[^33]: Sajjan, Shreeyak, et al. "Clear grasp: 3d shape estimation of transparent objects for manipulation." 2020 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2020.

[^34]: Murase, Hiroshi. "Surface shape reconstruction of an undulating transparent object." Proceedings Third International Conference on Computer Vision, 1990

[comment]: <> (4. Important concepts)

[^40]: The University of British Columbia, Department of Mathematics "The Law of Reflection", accessed 16. July 2021, [https://secure.math.ubc.ca/~cass/courses/m309-01a/chu/Fundamentals/reflection.htm](https://secure.math.ubc.ca/~cass/courses/m309-01a/chu/Fundamentals/reflection.htm)

[^41]: The University of British Columbia, Department of Mathematics "The Law of Refraction", accessed 16. July 2021, [https://secure.math.ubc.ca/~cass/courses/m309-01a/chu/Fundamentals/snell.htm](https://secure.math.ubc.ca/~cass/courses/m309-01a/chu/Fundamentals/snell.htm)

[^42]: Wikipedia "Fresnel equations", accessed 16. July 2021, [https://en.wikipedia.org/wiki/Fresnel_equations](https://en.wikipedia.org/wiki/Fresnel_equations)

[comment]: <> (5. Proposed method)

[^50]: Kutulakos, Kiriakos N., and Steven M. Seitz. "A theory of shape by space carving." International journal of computer vision 38.3 (2000): 199-218.

[^51]: Computerphile "Space Carving - Computerphile", 2016, [https://www.youtube.com/watch?v=cGs90KF4oTc&t=73s](https://www.youtube.com/watch?v=cGs90KF4oTc&t=73s)

[^52]: UofT CSC 2547 3D & Geometric Deep Learning "CSC2547 Differentiable Rendering A Survey", 2021, [https://www.youtube.com/watch?v=7LU0KcnSTc4](https://www.youtube.com/watch?v=7LU0KcnSTc4)

[^53]: Kazhdan, Michael, Matthew Bolitho, and Hugues Hoppe. "Poisson surface reconstruction." Proceedings of the fourth Eurographics symposium on Geometry processing. Vol. 7. 2006.

[^54] Qi, Charles R., et al. "Pointnet++: Deep hierarchical feature learning on point sets in a metric space." arXiv preprint arXiv:1706.02413 (2017).

[comment]: <> (6. Evaluation)

[^60]: Lyu, Jiahui, et al. "Differentiable refraction-tracing for mesh reconstruction of transparent objects." ACM Transactions on Graphics (TOG) 39.6 (2020): 1-13.

[^61]: Schonberger, Johannes L., and Jan-Michael Frahm. "Structure-from-motion revisited." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

[comment]: <> (7. Conclusion)

[^70]: ComputerVisionFoundation Videos "Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes", 2020, [https://www.youtube.com/watch?v=zVu1v4rasAE&t=53s](https://www.youtube.com/watch?v=zVu1v4rasAE&t=53s)
