# Classifiers
From this i want you to thuroughly analyse the diffrerent differences and similarities between my current classifier and the other ones in the litterature discussed below. I want you to focus on the differences in data acquisition, preprocessing, model architecture, training methodology, evaluation metrics, and real-world applicability. I want you to also highlight any gaps or limitations in the existing literature that my classifier addresses. Finally, I want you to explain how my classifier builds upon or diverges from previous work in the field. 
## Littersture review
\subsection{Modern deep-learning based approaches}
An alternative approach to the model-based feature extraction described above, is fully data-driven classification which discards the underlying physics and solely looks at the patterns and structures within the data itself. One paper that investigates this approach for mine detection is~\autocite{thomas2024machinelearningclassificationmetallicobjects}. This study uses two coils; one for transmission and one for reception. An induced emf in the reception coil is generated from the eddy currents induced in the target from a Heaviside step-off function in the transmission coil. As a basis for classification is both the scattered emf and the total emf used. The scattered emf is the emf induced from the eddy currents in the target, while the total emf is the scattered emf plus what's induced from the transmission coil. Signal conditioning, amplification and digitization is performed using industrial-grade laboratory equipment. Five different machine learning models are then applied to classify signatures coming from eight different targets. The linear models such as the perceptron and the logistic regression both operate on the 1D time-domain signatures. This also applies for the two non-linear models; a 1D dense Neural Network (NN) and a 1D CNN. A 2D CNN also classifies the the various objects based of the spectrogram of the signatures. The ROC plot shows that the top-performing models are the ones that operate on the 1D signature directly. The dense NN shows better overall performance with a $\Probability{FA}$ approaching 1 at $\Probability{D} \approx 0.1$. The second best performing model is the 1D CNN where the $\Probability{FA}$ approaches 1 around $\Probability{D} \approx 0.2$. However, as highlighted in~\autocite{thomas2024machinelearningclassificationmetallicobjects}, will the better classifier depend on the target. Objects 1 and 2 are classified best by the 1D CNN, and objects 6, 7 and 8 are classified best by the fully connected neural network with one exception for object 6 in the scattered case~\autocite{thomas2024machinelearningclassificationmetallicobjects}. One limitation in the methodology of the study is however that all of the measurements are conducted in a electromagnetically shielded laboratory. Future work in~\autocite{thomas2024machinelearningclassificationmetallicobjects} also highlights that
\begin{quote}
    ''Environmental effects will alter the pulse shape, and the electromagnetic field and matter interaction, introducing additional effects that may reduce classification accuracy, and these too should be investigated.''.
\end{quote}
The same lack of transferability to a real-world scenario is also seen in the closely related, recent study~\autocite{minhas2024deeplearningbasedclassificationantipersonnelmines}. This study aims to evaluate the performance of different machine learning algorithms to accurately detect and classify sub-gram metal pins commonly used in Anti Personnel Mines (APMs). As described in~\autocite{minhas2024deeplearningbasedclassificationantipersonnelmines}, is the entire dataset collected in a laboratory with minimal external noise. Although the study introduces classes for mineralized soil represented by clay-based materials to better simulate the field performance, is it reason to believe that the results will differ in a real-world scenario. 

Although having the addressed limitations as discussed above, is also the fully data-driven approach in~\autocite{minhas2024deeplearningbasedclassificationantipersonnelmines} presenting some key findings which are relevant for the application at hand. Similar to~\autocite{thomas2024machinelearningclassificationmetallicobjects} is the simpler 1D CNN coming out as the top-performing model, achieving a validation accuracy in the classification over 93.5\%. This includes a subpar classification accuracy between air and sand, which is not a critical misclassification compared to e.g. misclassifying a APM as sand. Overall proves the 1D CNN superior to the two other proposed models, being the K-Nearest Neighbor and the SVM. These findings align well with the previously discussed studies~\autocite{thomas2024machinelearningclassificationmetallicobjects} and~\autocite{simic2023landmineidentificationpulseinductionmetal}, where the CNN consistently shows strong classification performance and ranks among the top-performing models. Comparable performance is also seen when used with other data acquisition techniques, due to the inherent feature extraction capabilities of the CNN~\autocite{vakili2020performanceanalysiscomparisonmachinedeep}.

In addition to the outlined gaps and limitations, is also the discussed literature showcasing a more general gap regarding the distance to the target. For example is the study~\autocite{minhas2024deeplearningbasedclassificationantipersonnelmines} positioning the object max 2 cm from the center of the coil. The study~\autocite{thomas2024machinelearningclassificationmetallicobjects} also locates all targets within 1 to 5.5 cm from the center of the coil. Similar classification depths is also seen in other relevant studies~\autocite{simic2023rapidobjectdepthestimationpositionreferenced}~\autocite{safatly2021detectionclassificationlandminesusingmachine}. No studies have been found that directly addresses the binary classification of conductive objects with pulse induction over larger distances. Especially not in a agricultural setting, where research on this topic is scarce. For the applications outlined above in section \ref{sec:background} is classification depths up to 40 cm needed.

Another general gap in the available literature is the properties of the objects used in papers considered with pulse induction metal detection. The paper~\autocite{thomas2024machinelearningclassificationmetallicobjects} presents eight objects, all being bigger than the foreign objects typically found forage grass. Conversely addresses~\autocite{minhas2024deeplearningbasedclassificationantipersonnelmines} conductive landmine components in the order of a couple of grams or less. This is also the case in many of the aforementioned studies, and demonstrates the extensive research on pulse induction for landmine detection purposes. Research on detection and classification of metallic litter in agricultural environments is however limited.

The broader study aims to detect the presence of metallic litter in forage grass. It builds upon existing research, especially~\autocite{thomas2024machinelearningclassificationmetallicobjects} and~\autocite{minhas2024deeplearningbasedclassificationantipersonnelmines}. Neither of these studies uses a physics-based model and both implementations of the 1D CNN demonstrates as previously mentioned strong classification performance across all evaluated metrics. Hence, will a custom 1D CNN also be investigated in this paper. Note however that CNNs often entails extensive computational costs when compared to simpler models such as linear regression, KNNs and SVMs~\autocite{vakili2020performanceanalysiscomparisonmachinedeep}. To compensate for this cap has chipmakers in recent years been scrambling to develop develop custom hardware accelerators, which can provide decades worth of CNN acceleration directly on embedded platforms~\autocite{c2000realtimemicrocontrollersticom}. The use of such specialized hardware, together with rigorous selection of data points as well as careful trade-offs in the model architecture is therefore expected to enable embedded porting of the custom 1D CNN classifier. The proposed approach also brings novelty when it comes to the use case, which primarily addresses metallic litter detection in an agricultural context. Lastly will the feasibility of long-range binary classification with high-power pulse induction address aforementioned gaps in the literature. Verification on actual data from the fields will also be addressed.

## The papers discussed:
### thomas2024machinelearningclassificationmetallicobjects
Machine learning classification of metallic objects using pulse induction electromagnetic data
Ryan Thomas, Brian Salmon, Damien Holloway and Jan Olivier

Published 4 March 2024 • © 2024 The Author(s). Published by IOP Publishing Ltd
Measurement Science and Technology, Volume 35, Number 6
Citation Ryan Thomas et al 2024 Meas. Sci. Technol. 35 066103
DOI 10.1088/1361-6501/ad2cdd

DownloadArticle PDF
Authors
Figures
Tables
References
Download PDF
Article metrics
1106 Total downloads

33 total citations on Dimensions.
Submit
Submit to this Journal
Share this article
Article information
Abstract
This paper presents the classification of metallic objects using total and scattered pulse induction electromagnetic data, with a classification accuracy greater than 90%. Machine learning classification is applied to raw electromagnetic induction (EMI) data without the use of a physics-based model. The EMI method is applied to 8 metallic objects placed at increasing distances from 10–55 mm to the EMI sensing system. The EMI sensing system consists of two RL circuits placed in close proximity. Metallic objects are classified using linear algorithms including a perceptron and multiclass logistic regression, and nonlinear algorithms including a neural network, a 1D and 2D convolutional neural network (CNN). EMI data was collected using an experiment in an electromagnetically shielded laboratory. Feature maps are presented that explain the salient components of the EMI data used by the 1D and 2D CNN.

Export citation and abstractBibTeXRIS

Previous article in issue
Next article in issue

Original content from this work may be used under the terms of the Creative Commons Attribution 4.0 license. Any further distribution of this work must maintain attribution to the author(s) and the title of the work, journal citation and DOI.


1. Introduction
The ability to remotely sense the current state of an environment holds many advantages. One such case is the ability to characterize and classify metallic objects. EMI sensing offers a means for identifying and classifying metallic objects in varying environments. The EMI method has been used in steel characterization [11], buried object detection [39], defect detection [72], corrosion detection [30, 68], plate thickness estimation [19], and classification of metallic objects in walk-through metal detectors [33]. EMI sensing is conducted in both the time and frequency domain.

In the time domain, EMI sensing involves a transmitter coil and a receiver coil in proximity to a metallic object. The transmitter coil has a pulse current that creates a nearly uniform magnetic field close to the coil. When the current is switched off, the magnetic field collapses and induces eddy currents in a nearby metallic object. These eddy currents diffuse from the surface to within the metallic object over the time scale of tens of milliseconds [5]. The decay is a function of size, shape, conductivity and permeability of the metallic object [43]. This decay function, which is an object dependent and unique signature, may be used to classify a metallic object using the magnetic polarizability tensor [6, 41, 42].

For terrestrial EMI sensing, an equivalent induced dipole model has been used to describe time domain EMI scattering of a metallic object. One of the earliest implementations involved modeling a metallic object with equivalent magnetic and electric dipoles and then used the principle of reciprocity to obtain the voltage response of a receiver coil [13]. The induced dipole method was given a thorough theoretical basis and described by the magnetic polarizability tensor in [4]. To calculate the magnetic polarizability tensor, an inversion method is used, which involves fitting EMI data to a forward model of multiple orthogonal dipoles located at the center of the metallic object. A parametric forward model in addition to a nonlinear least squares algorithm was used by [42] to characterize metallic objects. Nonferrous metallic objects of various sizes, shapes, and materials were characterized using an inversion-based magnetic polarizability tensor measurement from time domain EMI data [54]. In [7] it was shown, using synthetic data, that a terrestrial dipole polarizability model could be used to characterize the scattered data from a buried metallic object in the conducting underwater environment.

In the frequency domain, the secondary magnetic field produced by induced eddy currents has a real part called the ‘in-phase’ and an imaginary part called the ‘quadrature’ that are used for classification purposes [52]. This eddy current response of a metallic object is frequency dependent and may be used to fingerprint objects in a process known as electromagnetic induction spectroscopy (EMIS) [64, 66]. Metallic objects were identified using a normalized EMIS spectrum, which is independent [64, 66] of the orientation or depth of the metallic objects, and compared to a library of EMI spectral data [23]. This method assumes that the magnetic field at a metallic object is uniform and therefore the object is small compared with its distance to, and size of, the sensor. The normalized spectrum is therefore range-independent and the target identification is based on only spectral shapes. The frequency domain magnetic polarizability tensor was used to characterize US coinage [14]. An inversion procedure to estimate the location and magnetic polarizability tensor of metal targets from broadband EMI data was presented in [15]. Whether frequency or time domain methods are used, characterization typically depends upon data inversion.

The inversion method has the inherent advantage of providing localization and orientation information as well as the induced dipole properties of the metallic object [44]. Inversion of the magnetic polarizability tensor has the disadvantage of typically using an iterative inversion method for classification [45]. The inherent iteration of the algorithm limits its use in a real-time classification system due to time constraints. One potential alternative that will enable real-time classification is machine learning methods. Machine learning methods may require significant compute time for training of a network, however, once training has completed, classification of new metallic objects is quickly completed. This makes machine learning an alternative to physics-based EMI data inversion for the purpose of metallic object classification in a real-time system.

Machine learning algorithms, with the successful deployment of deep learning approaches to image classification such as AlexNet in 2012 [28], are able to successfully detect objects [21]. Both kernel machines such as support vector machines, and artificial neural networks such as CNNs are capable of classification but CNNs have several advantages. CNNs are capable of automated feature extraction, which is not the case for kernel machines. Automation of the feature extraction process removes the need for expert human operators thereby improving productivity by reducing costs. Additionally, the features do not have to be known in advance. Rather, the convolution process is able to find the most important features used for classification by automatically optimizing filter weights. CNNs have the disadvantage of often requiring millions of weights whereby identifying the significance of weights is challenging. If the underlying data set has significant variation then a very large training set of data will be required for classification, which may be difficult or impossible to obtain. Despite the difficulty in obtaining sufficient quantity of data, good progress has recently been made in using machine learning for metallic object classification.

Low frequency electromagnetic systems, such as induction sensors near electrical conductors, may be numerically modeled using the magneto-quasi-static assumption and the Finite Difference Time Domain (FDTD) method [40]. The FDTD method was used to numerically model the electromagnetic scattering of buried objects in sediment layers under sea water [26]. The electric field values were used as inputs to a neural network and were shown to be capable of estimating the conductivity of buried objects. In other work, a numerical technique combining the multi-domain pseudospectral time-domain method and the Monte Carlo method was used to calculate the scattering of an object buried below a random rough surface separating two half-spaces [31].

Machine learning approaches have been applied to the EMI sensing problem. An EMI system in a laboratory and a 1D CNN was used to classify 7 small hidden metallic objects from time-domain magnetic polarizability tensor features, which included spheres, cuboids, and cylinder shapes [55]. The 1D CNN was trained on simulations and tested on measurements, and achieved 98% accuracy (with zero false negatives) for both multiclass and binary ‘threat or non-threat’ classification problems.

The depth of metallic objects was estimated using a pulse induction metal detector and a 1D CNN [53]. Simulated data of the time domain decay of metallic ellipsoids using an orthogonal dipole model nearby an EMI system were classified using machine learning techniques [61]. Thirty-three classification strategies based on eleven dimensionality reduction methods were investigated, including artificial neural networks, with the best classification achieving 99% accuracy for material-based and shape-based classification.

In other work, probabilistic (logistic regression, multi-layer perceptron, gradient boost) and non-probabilistic (decision trees, random forests, support vector machines) machine learning algorithms were trained to classify metallic objects using a dictionary of computed magnetic polarizability tensor spectral signatures in the frequency domain [63]. This was used to classify coins (number of objects equal to 8) and various threat and non-threat objects (number of objects equal to 15). The best performing probabilistic classifier was the gradient boost algorithm for both the coins and threat and non-threat class classification.

Machine learning methods have been used for object detection using ground penetrating radar (GPR). Traditionally, GPR clutter signal strength was suppressed using methods such as non-negative matrix factorization, singular value decomposition, principal component analysis, or independent component analysis [29]. Following clutter removal, target recognition was undertaken, often by human interpretation but also using genetic programming and neural networks [27]. More recently, region based CNN algorithms have been used to detect buried objects without removing clutter signals [46]. In [2] a simulated training data set was created using the gprMax electromagnetic simulation software [62] and augmented using a generative adversarial network. Classification was then undertaken on real data. The use of augmented generative adversarial network data, which increased the total records in the training set, resulted in increased target detection accuracy. While metallic objects have been detected without clutter removal, some EMI researchers have use statistical methods to remove clutter. In [49] a frequency-domain EMI sensor, the GEM-3 [65], was used to detect closely spaced metallic objects with overlapping EMI signatures. Independent components analysis and blind signal separation algorithms were used to extract source signatures in multiple target environments, with object signatures closely matching a library of EMI signatures. In other work, a wheeled robot with a GPR antenna was used on both synthetic and field data to detect reinforcing steel bars in a concrete slab [18]. A deep neural network (UNet) segmentation model was used to remove background noise from raw GPR B-scan images, and the dielectric information of the subsurface targets was determined using a ResNet34 encoder and residual neural network encoder (RNN).

Research has been done towards detection and classification without a physics-based model for use in real-time EMI systems. In [51], EMI time signatures of metallic spheroids were used with the joint diagonalization method to estimate the number of targets in a field survey. The joint diagonalization method was used to classify targets in real-time and without the need for a forward model. Joint diagonalization de-noises EMI data without distorting the target signal. In [60], a partially supervised approach was used to detect buried radioactive targets without the use of a physics-based model. EMI data in the frequency domain was collected using a GEM-3 [65] sensor. Potential targets were positively identified through feature extraction from raw EMI quadrature data. A decision tree identified targets of interest if the quadrature peak was larger than a threshold value, and the quadrature data had a negative second derivative (concavity test). Metallic object classification was then done using a one-class support vector machine.

Although machine learning methods have been used to classify metallic objects using EMI data, there is still the inherent difficulty of explaining the inner workings of neural network algorithms. Neural networks have been described as a black-box due to the difficultly in understanding the salient features used by the convolution and pooling layers of the network. The first attempts at understanding the important features and weights of convolutional layers were undertaken using a 2D CNN. In [71], a 2D CNN model with alternating layers of convolutional sparse coding and max pooling were used to capture image information including low-level edges, mid-level edge junctions, high-level object parts and complete objects. This work was further expanded in [70] by mapping network activities back to the input pixel space using a deconvolutional network, which showed the input pattern that originally caused a given activation in the feature maps. Early feature map visualization was conducted on the ImageNet [16] data set, but recent research has focused on the signal processing domain. In [35], the learned features of a 2D CNN were visualized for 2D B-scans of GPR data. Feature extraction and visualization has also been applied to 1D CNNs on time domain data, including analog circuit fault diagnosis [69], spectroscopy analysis [50], and fault detection and diagnosis of industrial processes [12].

Previous research implementing machine learning algorithms to classify metallic objects has focused on the use of the magnetic polarizability tensor, which is a physics-based approach that requires inversion of data to obtain the features used for classification. The approach taken in this paper is to apply machine learning methods to EMI data, without the use of physics-based models. This has the benefit of fast, real-time classification compared to inversion modelling, which requires greater classification computation time. This paper also differs from previous research by using the entire time domain data signal, which includes the early time information. Typically, inversion of time domain EMI data is done on data that begins at 0.1 ms after a current is switched off [7]. As mentioned previously, when the metallic object is small relative to the sensor, and the distance to the sensor, then a single dipole model is sufficient for inversion modeling. This paper differs from previous research by placing a large object very close to the sensor system. This means that a multipole model [34, 38, 57], rather than a dipole model, would be required to model the time domain decay curve and hence placing the object at different distances from the sensor will result in different decay rates of higher order multipoles [59] (the method presented in [64, 66] whereby data can be normalized and is depth invariant is not valid in this work).

In this paper, a time domain EMI signal is classified using a 1D CNN. Additionally, the time-domain EMI signal is converted into a spectrogram and then classified using a 2D CNN. The merit of using a time-frequency plot and a 2D CNN is to investigate the feature map, which provides insight into the part of the time-frequency plot that the 2D CNN assigned large weights. This provides explainability of the inner workings of the 2D CNN and the important features of the EMI data.

The paper is organized as follows: section 2 introduces the theory of neural networks, while section 3 discusses the research method including the experiment, data set, and the structure of the neural networks. Section 4 contains the experimental results and discussion and section 5 concludes the work.

2. Theory
The machine learning algorithms used in this work include both non-probabilistic (perceptron) and probabilistic (multiclass logistic regression, neural network, 1D CNN, 2D CNN) varieties. A probabilistic machine learning algorithm ensures that all outputs sum to 1 [63] and may be analyzed using a receiver operating characteristic (ROC) curve. Additionally, the machine learning algorithms used in this work include both linear (perceptron, multiclass logistic regression) and nonlinear (neural network, 1D CNN, 2D CNN) varieties.

2.1. Perceptron
A perceptron is a deterministic binary classifier of the form [37]


where x n is input vector number n of data set size N, is the transpose of the weight vector, bt is a bias term, are all the parameters, and is the indicator function where if x is true, else , i.e. a Heaviside step function.

Use of an error function minimization algorithm to determine the weight vector of the perceptron algorithm is challenging. If the error function is taken to be the total number of misclassified objects then this results in a piecewise constant function with discontinuities wherever a change in occurs. As the gradient is zero almost everywhere the gradient of the error function cannot be used for optimization purposes [8]. Additionally, the perceptron will only converge for linearly separable data [37].

2.2. Multiclass logistic regression
A binary logistic regression classifier is obtained by replacing the Heaviside step function of the perceptron with a sigmoid function. Unlike the perceptron, the logistic regression may be solved using gradient-based optimization methods and the algorithm, like the perceptron, is linear. The binary logistic regression may be expanded to the multiclass case by replacing the sigmoid with a softmax function and expanding the weight matrix.

A multiclass logistic regression is capable of classifying multiple objects, and has the form [37]


where is the input vector number, is the class label, is the softmax function, W is a C × D weight matrix, b is a C-dimensional bias vector, are all the parameters, and Cat is categorical distribution that is a discrete probability distribution with one parameter per class. A nonlinear classifier may be constructed by combining perceptrons in multiple layers, called a multilayer perceptron or an artificial neural network, with a nonlinear activation function applied to the weights and bias.

2.3. Neural network
The object classification problem in this paper is a multiclass classification problem, which may be solved using a neural network supervised learning algorithm. The mapping from observed data to classification in a neural network is

where Θ is observed data collected by sensors, is a binary output either 0 or 1, Ω is a neural network that maps the observed data to the output .

A neuron is a singular building block of a neural network. A neuron yj is typically a nonlinear function of a variable aj that is in turn a linear combination of input weights and a bias where the linear combinations are adaptive parameters [8]. The function describing an artificial neuron is

where i is the number of the input from a neuron, j refers to the neuron number, wji are the weights, are the biases, xi is the input and is the activation function. Artificial neurons can be linked to other artificial neurons to form deep neural networks [21].

Nonlinear data is more accurately modeled using a nonlinear, rather than a linear, activation function. A commonly used activation function is the Rectified Linear Unit (ReLU) [1] given by

where a is the input to the function. ReLU has been shown to outperform other activation functions such as the logistic sigmoid and the hyperbolic tangent for some classification tasks [20, 22]. The ReLU activation function has advantages including sparsity, excellent convergence performance and low computational cost [67]. However, the ReLU activation function has the disadvantage of the ‘dying ReLU problem’ whereby a neuron may not be activated during model training [32]. Use of a leaky-ReLU [17], which has a small, positive gradient for values less than or equal to zero, largely mitigates the ‘dying ReLU problem’. A leaky-ReLU was used as the activation function in a CNN based inversion of EMI data to estimate subsurface electrical conductivity [36, 47]. In this paper the ReLU was used as the activation function in the neural network and 1D and 2D CNN.

For multiclass problems the softmax function,

which is a generalization of the logistic function for multiple dimensions j, is often used in the final output layer of the neural network [21]. The softmax converts a vector of K real numbers into a probability distribution of K possible outcomes scaled from 0 to 1, which makes it a probabilistic classification algorithm.

The softmax function is a continuously differentiable function, where variations in classification confidence can be assessed and assigned, e.g. output closer to zero or one is more confidently predicted. The output of the network is compared to the desired user specified output and the significance of the differences are quantified using a loss function.

Consider the standard multiclass classification problem. The categorical cross-entropy loss function Γ, also known as the log loss function, may be used to quantify the loss of a multiclass neural network given by [8]

where k is the class, pk is the probability of the true label given by either 0 or 1, and distribution qk is the predicted value between 0 and 1 given by the softmax function. For N total samples the loss function should be averaged over all samples


The derivative of the cross-entropy function, which is necessary for optimization including gradient descent optimization, is


The log loss function is then used to optimize weights and bias in the neural network.

The backpropagation algorithm works by computing the gradient of the loss function with respect to each weight by the chain rule, computing the gradient one layer at a time, and iterating backward from the last layer to the first layer.

Using the gradient information to iteratively update the weights by following the negative gradient results in the expression [8]

where w is the weight, s is the step, η is the learning rate, and is the partial derivative of the loss function with respect to the weight. The gradient descent is prone to finding local minima, however good practice of multiple restarts with non-zero Gaussian distributed initial weights have provided sufficiently good minimums in many applications [8]. The optimized model’s weights and bias are then used to classify unseen data not used in the training of this model. In this work a neural network is used to classify metallic objects using time domain data. An Adam optimizer [24], which is a stochastic gradient descent method based on adaptive estimation of first-order and second-order moments, was used to train the neural network. The neural networks used in this work are large (the 2D CNN has over 15 million parameters) and the Adam optimizer was chosen as it is computationally efficient and has small memory requirements, which are attributes well suited to large parameter models.

2.4. CNN
A 1D CNN is an extension of a neural network that includes automated feature extraction [25]. The automated feature extraction consists of one input layer, one or more convolutional and pooling layers, and a flattening of the convolutional data into a 1D array. The automated feature extraction part of the 1D CNN is then connected to a neural network. Feature extraction occurs in the convolutional layers through operations on data with adaptable, learnable kernels. A kernel, otherwise known as a filter, is applied to the image data (input data) using a dot product operator. Pooling layers, such as maximum pooling, reduces the feature space (spatial size) by returning only the largest value in a kernel. This ensures the complexity of the CNN is restricted to a feasible size.

A 2D CNN, like the 1D CNN, is an extension of a neural network, however the input and kernel is 2D rather than 1D. The kernel is iterated over the entire 2D input. In this work the 2D input into the 2D CNN is a spectrogram created from 1D time domain data.

3. Method
3.1. Experiment
This section describes the EMI method and experiment used to generate the time signatures of metallic objects. An EMI sensing setup generally consists of a transmitter coil, receiver coil, control system, and a metallic object placed near the sensor system [58]. The current in a transmitter coil is switched off (Heaviside step-off function), which creates a changing magnetic field. Eddy currents are induced in the metallic object and radiate electromagnetic fields, which are detected by the receiver coil.

The experiment used in this work consisted of an EMI setup as shown in figure 1. The transmitter coil and receiver coil are both RL circuits. The transmitter and receiver coils were created using enameled copper wire wound 1000 times around a plastic spool. The transmitter coil was controlled using a National Instruments cDAQ with a NI 9264 digital-to-analog converter. The signal from the receiver coil was amplified using a Stanford Research Systems model SR560 low-noise preamplifier and recorded using the cDAQ with a NI 9775 analog-to-digital converter. The receiver coil circuit has Rr   =  44 Ω, Lr   =  6.2 mH, a NI 9775 sample rate of 200 kS s−1, and SR560 enabled amplification of 500 times the voltage across a 10 Ω resistor with a band pass filter between 0.1 Hz and 10 kHz. The transmitter coil circuit has Rt   =  298 Ω, Lt   =  150 mH, and a step-off voltage from 2–0 V, with an output rate of 12 kS s−1. The time constant [3] (τ  =  ) of the receiver coil is τr   =  141 , and the transmitter coil is τt   =  503 . The transmitter coil input voltage was a rectangular waveform with pulse width of 100 ms.

Zoom InZoom OutReset image size
Figure 1. EMI method experiment setup. The cDAQ generates a step-off voltage in the transmitter coil, which creates a magnetic flux density that penetrates the metallic object and induces eddy currents. The eddy currents generate a magnetic flux density transient that is detected by the receiver coil. The transient passes through the preamplifier and into a cDAQ analog-to-digital converter. The data is then recorded on the laptop.

Download figure:

Standard imageHigh-resolution image
3.2. Data set
The EMI responses for 8 metallic objects were collected. The objects included a plate, truncated cone, a pot and 5 boxes. The 5 boxes were chosen to determine if the machine learning algorithm could distinguish between objects of similar shape. Each object was placed near the EMI sensor system with separation of between 10 and 55 mm in 5 mm increments. The objects were recorded at varying distances to test the ability of the machine learning algorithms to cut through higher order multipole effects that arise when a large object is close to the sensor system and of a similar size as the sensor. Additionally, the objects were shifted laterally randomly within the range 0–10 mm in the 2D plane parallel to the sensor system. At each vertical step the experiment was replicated 100 times. As there are 10 intervals, each object has 1000 data records with a total of 8000 records for all objects. The EMI data is in a 1D time domain format and is 20 ms duration.

The 1D data was converted into a spectrogram to enable the use of the 2D CNN, which is a widely used computer vision algorithm. The Fourier transform window was set to 1.5 ms duration with 291 overlaps. The frequency points are calculated at 10 Hz intervals between 0 Hz and 2.55 kHz. The spectrogram has dimensions 256 × 256. Classification was applied to both total data (total voltage response of the receiver coil, which is source plus scattered) and scattered data (source voltage minus total voltage of the receiver coil where source voltage is recorded with no metallic object). The results presented in this paper show varying classification accuracy between total and scattered EMI data.

A 10-fold cross-validation [48] was used for all algorithms to understand the statistics, such as the standard deviation, of the classification accuracy. All 8000 records were randomly shuffled and arranged so that an equal number from each metallic object were present in each 10-fold interval (800 records). This means that there were 100 records of each object per 800 records. The first test set consisted of records 1–800, and the train set was 801–8000. The next test set was from 801–1600, and the train set was the combined 1–800 and 1601–8000 records, and so on according to the rules of 10-fold cross-validation.

3.3. Classification algorithms
A perceptron function was used to test the linear separability of the EMI data sets. The perceptron from the Python scikit-learn library was used in the default setting. The perceptron uses a stochastic gradient descent algorithm [9] as a solver with an L2 regularization. Other inputs were α  =  0.0001, maximum epochs  =  1000, and stopping criterion tolerance of 0.001.

The logistic regression function from the Python scikit-learn library was used for multiclass logistic regression classification. The L-BFGS [10], a limited-memory quasi-Newton code for bound-constrained optimization, was used as the solver with L2 regularization with a bias added to the decision function. Other inputs were maximum epochs  =  100, and stopping criterion tolerance of 0.0001.

A conventional neural network, using the TensorFlow library, was applied to the data set. It consisted of three hidden layers (4096, 512, 8), a learning rate of 5 × 10−5, ReLU activation function on the hidden layers, a softmax function on the output layer, a sparse categorical cross-entropy loss function, Adam optimizer, batch size of 32, and was trained for 50 epochs.

A 1D CNN, using the TensorFlow library, was used on the time signature data. Figure 2 shows an illustration of the 1D CNN. It consisted of, in order, a 1D convolution layer of 16 filters of kernel size 128 with ReLU activation, max pooling of size 4, another 1D convolution layer of 16 filters of kernel size 64 with ReLU activation, a densely connected flattened layer, and finally a softmax layer on 8 outputs. The learning rate was set to 1 × 10−4, batch size of 32, Adam optimizer, and the model was trained over 50 epochs.

Zoom InZoom OutReset image size
Figure 2. 1D CNN classifier. An input time signature is convolved with 16 filters of kernel size 128, followed by max pooling with a spatial window size of 4, followed by convolution with 16 filters of kernel size 64. The data is flattened into a densely connected array, then connected to 8 neurons, and finally has a softmax function applied to the classification output.

Download figure:

Standard imageHigh-resolution image
Figure 3 shows the 2D CNN used to classify the metallic objects, which was created using TensorFlow. Batch size of 32 was used. Adam optimizer was used with a learning rate of 2 × 10−4. An initial learning rate of 0.1 was used to accelerate network adaptation and systematically reduced until the model converged without significant variance on a validation set, which is described below. The exponential decay rate for the 1st moment (0.9) and 2nd moment (0.999) were set as the default TensorFlow values for the Adam optimizer. The 2D CNN reached a maximum classification accuracy on the train set typically within the 10–12 epoch range for all 10-fold cross-validation iterations, and therefore the epoch was set to 15 to account for any variance in training.

Zoom InZoom OutReset image size
Figure 3. 2D CNN classifier. An input spectrogram has a 3 × 3 filter convolved over all pixels, and then max pooling is applied. Two more convolutions are applied followed by a max pooling and then the data is flattened into an array. The flattened array is connected to a neural network with a softmax function applied to the classification output.

Download figure:

Standard imageHigh-resolution image
A total of 32, 3 × 3 filters were used in the convolutional layer. Filter size of 3 × 3 was chosen as it is a small size that has small computational load while providing horizontal and vertical edge detection, and smoothing. It also includes a center pixel with all surrounding neighbor pixels allowing for relationships between pixels to be captured in the 2D CNN. The Visual Geometry Group Network (VGGNet) [56], a deep learning algorithm that achieved high classification accuracy on the ImageNet [16] data set, also uses 3 × 3 filters with 16–19 layers to reduce model size and therefore computation time. A stack of 2 convolutional layers with 3 × 3 filters (without spatial pooling in between) is parameterized by 2  weights (where C is the number of channels and in this work C = 1) and a single 5 × 5 convolutional layer is parameterized by 1  weights, resulting in 28% less weights for the 3 × 3 filters case. In addition to requiring less weights and therefore less computational load, there is a ReLU nonlinear activation function between convolutional layers, which makes the decision function more discriminative [56]. No pooling layer was implemented between convolutional layers 2 and 3 so that the output size would remain constant and the feature maps in these layers could be overlaid to reveal salient feature selection by the 2D CNN, as shown in figure 10.

A ReLU activation function was applied to the output of each layer excluding the final layer. A maximum pooling operation was applied after the first convolutional layer and the third convolutional layer to reduce the size of the model. The model has three layers of convolution and two layers of max pooling. The images were flattened into a single column array. This completes the feature extraction process. The flattened array was then densely joined to a single layer neural network of 128 nodes. A softmax function was applied to the last layer of the neural network that contains 8 classification outputs. Each classification outputs a single number where the classification chosen has the largest value.

The hyperparameters of the machine learning models were optimized using a validation set. The validation set was created by randomly shuffling the EMI data. The first 10% of the data was used for validation, and the remaining 90% was used for training. The order of the randomly shuffled data set remained the same for tuning hyperparameters across all machine learning models. Once hyperparameters were chosen, the EMI data set was reshuffled again and 10-fold cross-validation was used to create the training and test sets.

4. Results
4.1. EMI data
Each metallic object was found to have a unique EMI time signature response to the EMI method. Photos of the metallic objects, an example of the spectrogram of the receiver coil voltage response of each metallic object placed near the sensor system, and a plot of the EMI time signature responses, are given in figure 4 for the total EMI data. Figure 5 shows the scattered EMI data equivalent.

Zoom InZoom OutReset image size
Figure 4. Object images, spectrograms, and total EMI time signature data. Spectrograms are expressed as power per frequency on a scale of −20 dB Hz−1 to −130 dB Hz−1. Objects are 20 mm from the sensor system.

Download figure:

Standard imageHigh-resolution image
Zoom InZoom OutReset image size
Figure 5. Object images, spectrograms, and scattered EMI time signature data. Spectrograms are expressed as power per frequency on a scale of −40 dB Hz−1 to −120 dB Hz−1. Objects are 20 mm from the sensor system and time signatures are voltage source minus total.

Download figure:

Standard imageHigh-resolution image
In figure 4 the time signatures of the metallic objects vary in maximum voltage, time of maximum voltage point, decay rate, and shape of the voltage response. From the time domain plot, each metallic object has a unique signature. The spectrogram of each object is a logarithmic plot. Some objects have a receiver coil voltage response that decays below the noise floor before other objects. Object 2 and 3, for example, have a receiver coil voltage reduced to the noise floor by 8 ms, whereas object 8 has receiver coil voltage above the noise floor beyond 10 ms. This is likely attributable to a greater permeability of object 8 compared to objects 2 and 3.

Figure 5 shows the time signatures and spectrograms of the scattered EMI data set. The maximum voltage point between objects is more easily observed compared to the total EMI data set. For example, the peak voltage of object 8 occurs approximately 0.3 ms later than object 3. There is also an additional point of information compared to the total EMI data set called the zero–crossing point. Object 8 crosses the zero voltage point over 1 ms after object 3. The decay rates also differ between objects with object 8 sustaining receiver coil voltage above the noise floor for greater duration than other objects. The spectrograms also show a zero point on most plots, reflecting the zero–crossing point of the data. As in the total case, object 8 spectrogram in the scattered EMI data shows a sustained receiver coil voltage above the noise floor for a longer duration than object 7.

4.2. Classification accuracy
Table 1 shows the mean classification accuracy and standard deviation using 10-fold cross-validation on both the total and scattered EMI data. Mean classification accuracy was improved for all algorithms in the scattered versus total EMI data set. Additionally, the standard deviation was reduced in all algorithms. In the scattered case, the 1D CNN was the best performing algorithm. In the total EMI data set, the neural network was the best performing algorithm. There was a large improvement of 13.6% in the mean classification accuracy of the 2D CNN in the scattered versus total EMI data set. The perceptron and multiclass linear regression (MLR) under-performed the machine learning algorithms, suggesting that the EMI data set is not linearly separable, which was expected as the higher order multipoles are nonlinear and are required to adequately describe close range EMI scattering phenomena.

Table 1. Mean classification accuracy (%) and standard deviation using 10-fold cross-validation.

Algorithm	Total	Scattered
Perceptron	51.5±8.7	65.6±5.5
MLR	77.4±2.1	85.4±1.0
Neural network	95.6±2.5	98.0±1.4
1D CNN	94.0±3.3	98.1±1.2
2D CNN	82.0±3.5	95.6±1.1
Figure 6 shows a box and whisker plot and confusion matrices for the total EMI data. Each colored data point in the box and whisker plot represents the classification accuracy of a single outcome of the 10-fold cross-validation. The perceptron has been omitted from the box and whisker plot to enable clearer plotting of the remaining algorithms. All 10-fold cross-validation data points of the neural network are greater than the 2D CNN and multiclass logistic regression. The 1D CNN has only a single data point overlapping the data in the 2D CNN. The distributions in the neural network and the 1D CNN are very similar and mostly overlap. From the box and whisker plot it is clear that for the EMI data in this work the neural network and the 1D CNN, which are both 1D and nonlinear, outperform the other algorithms.

Zoom InZoom OutReset image size
Figure 6. Total EMI data (a) box and whisker plot of classification accuracy where each dot represents a single data point in a 10-fold cross-validation and (b)–(e) confusion matrices summed over all 10-fold cross-validation data points.

Download figure:

Standard imageHigh-resolution image
The confusion matrices in figure 6 show the total summed over all 10-fold cross-validation data sets, where perfect classification would result in a value of 1000 on the diagonal elements. The perceptron confuses objects 1 (246 correct) and 6 (242 correct) the most, and object 5 (719 correct) and object 8 (891 correct) the least. The linear perceptron manages to classify just over half (51.5±8.7%) of objects correctly. The nonlinear neural network is more accurate than the linear perceptron. The neural network confuses objects 3 (915 correct) and 6 (888 correct) the most, and object 5 (997 correct) and object 8 (994 correct) the least. Although the neural network (95.6±2.5%) outperforms the 1D CNN (94.0±3.3%), there are some objects that are more accurately classified by the 1D CNN. For example, object 1 (988 versus 969) and object 2 (969 versus 945) are more accurately classified by the 1D CNN, suggesting that some algorithms may be more suited to distinguishing between some objects although total classification accuracy is less than alternative algorithms.

Figure 7 shows a box and whisker plot for the scattered EMI data. All 10-fold cross-validation data points of the nonlinear algorithms (neural network, 1D CNN, 2D CNN) are greater than the linear multiclass logistic regression. The distributions in the neural network and the 1D CNN are very similar and mostly overlap. Although the 2D CNN has improved using the scattered EMI data, it still underperforms the 1D CNN and neural network with most data points being less than the data points in these algorithms.

Zoom InZoom OutReset image size
Figure 7. Scattered EMI data (a) box and whisker plot of classification accuracy where each dot represents a single data point in a 10-fold cross-validation and (b)–(e) confusion matrices summed over all 10-fold cross-validation data points.

Download figure:

Standard imageHigh-resolution image
The confusion matrices for the scattered EMI data are shown in figure 7. The perceptron confuses objects 5 (229 correct) and 6 (324 correct) the most, and object 1 (971 correct) and object 8 (904 correct) the least. The linear perceptron classifies 65.6±5.5% of objects correctly. The nonlinear algorithms are more accurate than the linear perceptron. The 1D CNN performs best (98.1±1.2%) and confuses object 4 (950 correct) and 6 (945 correct) the most, and object 1, 2, and 3 (1000 correct) the least. Objects 1, 2, and 3 are all boxes of similar shape, and both the 1D and 2D CNN were able to perfectly classify these objects. Although the 1D CNN outperforms the neural network, there are some objects that are more accurately classified by the neural network. For example, object 7 (991 versus 979) is more accurately classified by the neural network, suggesting that, like the total EMI data case, some algorithms may be more suited to distinguishing between some objects although total classification accuracy is less than alternative algorithms.

Comparing the confusion matrices of the total and scattered EMI data yields interesting insights. The total EMI data set does not perfectly classify any one object as shown in figure 6. Conversely, the scattered EMI data set shown in figure 7 perfectly classifies objects 1, 2, and 3 using the 1D CNN and 2D CNN (1000 correct). The multiclass logistic regression on scattered EMI data also classifies object 1 with 100% accuracy across 10-folds despite being a linear algorithm. In the total EMI data set, object 8 is consistently classified most accurately with object 1–3 being less accurate. Conversely, in the scattered EMI data set objects 1–3 are most accurately classified with object 8 being confused more often. One possible explanation is provided through the interpretation of the second convolutional layer feature map in figure 9. Object 1 has a large weight assigned in the second convolutional layer approximately at the zero–crossing point and a few milliseconds afterwards. This is not evident in the second convolutional layers of the other objects. Comparatively, the total EMI data set of figure 8 does not have a large weight at the zero–crossing point of the second convolutional layer of object 1. Rather, all objects have a somewhat similar weight applied to the second convolutional layer. It could be that the zero–crossing point in the scattered EMI data provides additional information that is not present in the total EMI data, leading to greater classification accuracy and distinction between objects.

Zoom InZoom OutReset image size
Figure 8. Feature maps of the 1D CNN trained on the total EMI time signatures. The input has 4000 data points, the first convolutional layer outputs 3873 data points across 16 filters and these are stacked on each other in the surface plot of ‘Convolution 1’. The second convolutional layer outputs 905 data points across 16 filters after a max pooling with spatial window size of 4. Convolutional plots have been scaled to the same size of the input for illustration purposes. Red signifies large weight, and blue signifies low weight, assigned by the 1D CNN. The 1D CNN has used early time features in the input signal by assigning large weight to the signal in the few milliseconds after the step-off pulse begins. The time signatures have been normalized. Numbers in the top right hand corner of the plots signify the object number.

Download figure:

Standard imageHigh-resolution image
Zoom InZoom OutReset image size
Figure 9. Feature maps of the 1D CNN trained on the scattered EMI time signatures. The input has 4000 data points, the first convolutional layer outputs 3873 data points across 16 filters and these are stacked on each other in the surface plot of ‘Convolution 1’. The second convolutional layer outputs 905 data points across 16 filters after a max pooling with spatial window size of 4. Convolutional plots have been scaled to the same size of the input for illustration purposes. Red signifies large weight, and blue signifies low weight, assigned by the 1D CNN. The 1D CNN has used early time features in the input signal by assigning large weight to the signal in the few milliseconds after the step-off pulse begins. The time signatures have been normalized. Numbers in the top right hand corner of the plots signify the object number.

Download figure:

Standard imageHigh-resolution image
4.3. Feature maps
The inner workings of the hidden layers of a neural network may be interpreted through the use of feature maps. Visualization of the neuron activation levels at the end of the convolution and max pooling layers provides insight into the abstracted features that were used by the CNN for classification.

Feature maps of the 1D CNN trained on the total EMI time signatures are shown in figure 8. The input has 4000 data points, the first convolutional layer outputs 3873 data points across 16 filters and these are stacked on each other in the surface plot of ‘Convolution 1’. The second convolutional layer outputs 905 data points across 16 filters after a max pooling with spatial window size of 4. Convolutional plots have been scaled to the same size of the input for illustration purposes. Red signifies large weight, and blue signifies low weight, assigned by the 1D CNN. The 1D CNN has used early time features in the input signal by assigning large weight to the signal in the few milliseconds after the step-off pulse begins.

Feature maps of the 1D CNN trained on the scattered EMI time signatures are shown in figure 9. The structure of the 1D CNN is the same for the scattered and total EMI data sets. The 1D CNN on scattered time signature data has used early time features in the input signal by assigning large weight to the signal in the few milliseconds after the step-off pulse begins, as was done in the total EMI data case.

Feature maps for the 2D CNN of total EMI data are shown in figure 10. Numbers in the top right hand corner of the plots signify the object number. Feature maps of the third convolutional layer (scaled between 0.5 and 1 and in yellow to red colors) are overlaid on the second convolutional layer (scaled between 0 and 0.5 and in blue to green colors) for objects 10 mm from the EMI sensor system. The 2D CNN algorithm has assigned large weight to the outer envelope of the spectrograms corresponding to importance of high frequency parts of the signal a few milliseconds after the Heaviside step-off pulse begins (current switched off) and low frequency parts of the signal at late time.

Zoom InZoom OutReset image size
Figure 10. Feature maps of the third convolutional layer (scaled between 0.5 and 1 and in yellow to red colors) are overlaid on the second convolutional layer (scaled between 0 and 0.5 and in blue to green colors) for objects 10 mm from the EMI sensor system. Total data was used. The 2D CNN algorithm has assigned large weight to the outer envelope of the spectrograms. Numbers in the top right hand corner of the plots signify the object number.

Download figure:

Standard imageHigh-resolution image
All objects have very small weight assigned in the third convolutional layer in the first 1 ms after the Heaviside step-off pulse has begun, suggesting that in the total EMI data case the 2D CNN is biased towards late time data for the classification of the metallic objects. For example, object 7 has the 2D CNN feature map weight decrease much sooner than object 8, corresponding to the algorithm assigning large weight to the sustained voltage response of object 8 versus object 7. One possible explanation is the voltage response of the source and metallic objects, as given in figure 4, are very similar in the first 0.5 ms after the voltage step-off. The source decays rapidly 1 ms after the voltage step-off with the voltage response of the metallic objects being sustained for a longer period of time. Effectively the 2D CNN on the total EMI data set may be removing the source voltage by assigning very low weight to the first 1 ms of the time signature.

Feature maps for the 2D CNN of scattered EMI data are shown in figure 11, which includes feature maps of the third convolutional layer for objects 10 mm from the EMI sensor system. Numbers in the top right hand corner of the plots signify the object number. The 2D CNN algorithm has assigned large weight to the early time part of the spectrograms, in the first few milliseconds after the beginning of the Heaviside step-off pulse. There is also a difference between objects. Object 1 has greater weight assigned to late time effects compared to other objects (for example object 5).

Zoom InZoom OutReset image size
Figure 11. Feature maps of the third convolutional layer for objects 10 mm from the EMI sensor system where scattered data was used. The 2D CNN algorithm has assigned large weight to the early time part of the spectrograms. There is also a difference between objects with object 1 having greater weight assigned to late time effects compared to other objects. Numbers in the top right hand corner of the plots signify the object number.

Download figure:

Standard imageHigh-resolution image
One possible explanation for differences in large weight assigned to early times in the 2D CNN trained on scattered EMI data relates to the shape of the metallic objects. Early time in the receiver coil voltage response corresponds to high frequencies where object penetration is shallow and correlates with large objects with large surface area [51]. This suggests that the 2D CNN on scattered EMI data is primarily using large frequencies for classification. At later times eddy currents penetrate deeply into metallic objects and low frequency harmonics dominate the voltage response, which corresponds to metal content and volume of the object. For example, object 7 is a large, thin plate and has a strong early voltage response and then rapidly decays as shown in the scattered EMI time signatures plot in figure 5. In figure 11(7) (corresponding to object 7) the 2D CNN has assigned large weight to the early time part (associated with a large surface area) of the spectrogram and very small weight to the late time part (associated with a small volume). Object 8 has different weighting to object 7. Object 8 is a pot with a large surface area (large early time voltage response) and is very thick (large late time voltage response). In figure 11(8) (corresponding to object 8) the 2D CNN has assigned large weight to the early and late time part of the spectrogram.

There is a difference between the part of the spectrogram given large weight in the total and scattered EMI data sets. In the total case, the late time, relatively higher frequency is given large weight and the early time, low frequency part of the spectrogram has small weight. Conversely, the scattered EMI data set has large weight assigned to the early time and low frequency part of the spectrogram. In summary, the 2D CNN uses different parts of the time-frequency plot dependent on whether or not the EMI data set is total or scattered.

4.4. Receiver operating characteristic curve
Figure 12 shows the ROC curve with the total and scattered EMI data. The 2D CNN (Total), and multiclass logistic regression function perform poorly compared with the other algorithms. The area under the curve (AUC) in the total case (AUC = 0.9997) exceeds the scattered case (AUC = 0.9988) for the neural network for this particular k-fold. There is clearly superior performance of the nonlinear algorithms (NN, 1D CNN, 2D CNN) versus the linear algorithm (MLR).

Zoom InZoom OutReset image size
Figure 12. ROC curve for total and scattered EMI data.

Download figure:

Standard imageHigh-resolution image
5. Conclusions
In this paper metallic objects were classified using EMI data and machine learning methods. The time domain data collected in an electromagnetically shielded laboratory was classified using a perceptron, multiclass logistic regression, neural network, and 1D CNN. Additionally, time domain data was converted into a spectrogram and a 2D CNN was used to classify the metallic objects. Mean classification accuracy of a 10-fold cross-validation was greater for all algorithms in the scattered versus total EMI data set. In the scattered EMI data set, the 1D CNN was the most accurate classifier with 98.1±1.2%.

Feature maps were plotted and provided explainability of the 1D and 2D CNN algorithms. In the case of the 1D CNN, early time data was given large weight for both the total and scattered EMI data sets. In the case of the 2D CNN trained on the total EMI data set, the late time, higher frequency part of the spectrogram was given large weight and the early time information was not important. Conversely, the scattered EMI data set had large weight given to the early time, low frequency part of the spectrogram. Choice of total versus scattered data set evidently impacts the classification accuracy of machine learning algorithms, as well as the part of a time-frequency plot given large weights by the 2D CNN.

The approach taken in this paper was to apply machine learning methods directly to EMI data, without the use of physics-based models such as inversion of the magnetic polarizability tensor. It was shown that it is possible to classify metallic objects with high accuracy without the use of physics-based models by using machine learning algorithms on raw EMI data. Nonlinear algorithms were required to obtain high classification accuracy of the metallic objects using raw EMI data.

The machine learning approach presented in this paper has the benefit of fast, real-time classification compared to nonlinear inversion modeling, which requires greater classification computation time.

Previous research typically excluded the early time information and focused on the signal in the 0.1–10 ms range. In this paper, the entire time signature was used as input and the 1D CNN and 2D CNN assigned weights to part of the time signature that were used for classification, which is known as automated feature extraction. The results in this paper suggest that EMI signal data in the 0–0.1 ms range could hold useful classification information.

Metallic objects in this work were large relative to the EMI sensor system and placed close by the sensor system, which meant that a multipole model would be required to model the time domain decay curve. The nonlinear machine learning algorithms were able to cut through these multipole effects and classify the metallic objects with high accuracy, and linear algorithms were inferior to nonlinear algorithms.

While results are promising, further work on alternative technologies, including varying pulse shapes with varying bandwidth, may provide further insight and increase detectability of metallic objects. Square-integrable pulses have controllable frequency spectra and defined energy, unlike the Heaviside step-off pulse used in this work, and should be investigated. Classification of metallic objects submerged in dielectrics such as water, and weak conductors such as sand and seawater, replicate the conditions in which metallic objects are found. Environmental effects will alter the pulse shape, and the electromagnetic field and matter interaction, introducing additional effects that may reduce classification accuracy, and these too should be investigated.

Data availability statement
The data cannot be made publicly available upon publication because they are owned by a third party and the terms of use prevent public distribution. The data that support the findings of this study are available upon reasonable request from the authors.

Conflict of interest
The authors report there are no competing interests to declare.

### minhas2024deeplearningbasedclassificationantipersonnelmines
Deep learning-based classification of anti-personnel mines and sub-gram metal content in mineralized soil (DL-MMD)
Shahab Faiz Minhas, Maqsood Hussain Shah & Talal Khaliq 
Scientific Reports volume 14, Article number: 10830 (2024) Cite this article

3036 Accesses

3 Citations

Metricsdetails

Abstract
De-mining operations are of critical importance for humanitarian efforts and safety in conflict-affected regions. In this paper, we address the challenge of enhancing the accuracy and efficiency of mine detection systems. We present an innovative Deep Learning architecture tailored for pulse induction-based Metallic Mine Detectors (MMD), so called DL-MMD. Our methodology leverages deep neural networks to distinguish amongst nine distinct materials with an exceptional validation accuracy of 93.5%. This high level of precision enables us not only to differentiate between anti-personnel mines, without metal plates but also to detect minuscule 0.2-g vertical paper pins in both mineralized soil and non-mineralized environments. Moreover, through comparative analysis, we demonstrate a substantial 3% and 7% improvement (approx.) in accuracy performance compared to the traditional K-Nearest Neighbors and Support Vector Machine classifiers, respectively. The fusion of deep neural networks with the pulse induction-based MMD not only presents a cost-effective solution but also significantly expedites decision-making processes in de-mining operations, ultimately contributing to improved safety and effectiveness in these critical endeavors.

Similar content being viewed by others

Mineral prospectivity prediction based on convolutional neural network and ensemble learning
Article Open access
30 September 2024

Adaptive signal recognition in mines based on deep learning
Article Open access
25 March 2025

Enabling deeper learning on big data for materials informatics applications
Article Open access
19 February 2021
Introduction
Metal-based mine detectors have been extensively employed in humanitarian and military de-mining operations for the past seventy years. However, as technology advances, new challenges and difficulties arise. Notably, the metal content in mines, particularly anti-personnel mines and integrated explosive devices (IEDs), is decreasing, while the quantity of trash and miscellaneous metal per square meter is increasing due to human activities and conflicts. According to the UN de-mining operation report1, for every mine found, there are about a thousand small pieces of scrap metal detected.

In the past decade, the notable advancement in demining endeavor has been the introduction of Ground Penetrating Radar (GPR). However, for practical mine detection purposes, GPR still requires the support of a metal detector. This combination of GPR and a metal detector, known as a hybrid technology2, is currently utilized by advanced military forces worldwide. In this case, the decision based on the detection can be independent of each other using separate signal processing techniques or can be a combination of (both) using data fusion algorithm as shown in Ref3. Some widely used GPR-based mine detectors include the Vallon Minehound VMR3 and CEIA ALIS-RT etc.4,5. Nonetheless, this technology is extremely expensive when compared to traditional metal detectors, as it costs eight to ten times more approximately (Ref4, quote can be obtained via contact) and still relies on an integrated metal detector sensor.

Over the years, there have been significant advancements in the search head assembly of metallic mine detectors, including the use of multi-strand coils6 to enhance sensitivity and the implementation of various factors to reduce noise. However, the true potential of metallic mine detectors (MMDs) remains largely untapped, leaving ample room for improvement, particularly in the field of signal processing. Given the progress in machine learning and robust artificial intelligence techniques, it is a logical progression to explore the application of these techniques in MMDs. Recent mine detectors have already incorporated machine learning techniques with a limited focus, such as automatic soil compensation and the differentiation of nonferrous and ferrous materials7. A similar concept from GPR (2D image) is utilized using MMDs only8, with a machine learning based classification of mines & metals. It is limited in terms of classification accuracy in soil and is also limited in terms of practical utility (explained later).

In the study discussed in Ref9, the focus is on binary classification—determining whether an object is a mine or not with using spatial measurement diversity. The results indicate best performance of a 50% probability of detection (& identification) of mines with minimal false alarm rate (of less than 5%). However, this rate increase substantially (above 30%) with the increase in detection probability especially for mines with low metal content (like APM), which render it impractical. Another approach, detailed in Ref10, involves a custom architecture with two receiver coils and one transmitter coil, which utilizes broadband frequencies (ranging from 60 Hz to 15.8 kHz) and employs an inversion procedure to estimate the magnetic properties of metal targets. Unlike typical MMDs, this method is distinct in its complexity, as instead of moving the electromagnetic interference (EMI) sensor over the target, the target itself is moved over the sensor. It is neither feasible to practically move such large rectangular EMI sensor by a user in a field environment over a target nor authors have shown/discussed its efficacy against any buried target.

In order to circumvent the limitations such as limited accuracy, high complexity, limited practical utility, in this research, we utilize a pulsed induction (PI) metal detector. This type of technique has advantages in various scenarios where other non-PI based detectors (e.g. CW) would face challenges11, particularly in environments with highly conductive materials in the soil or surroundings. Additionally, PI-based systems have the ability to detect metal at greater depths compared to other systems. In the following section, we will provide a brief explanation of the operation of a PI detector. However, it is important to note that in mine detection algorithms, a significant portion of the received signal curve information (sample by sample) is not extracted, as the focus is primarily on integration (of the curve samples)12 and filtering to ensure reliability and user safety. Our research, on the other hand, centers on this received signal curve and will be presented in the subsequent sections.

The main contributions of this paper are highlighted in the following:

1.
Design of data acquisition front end to ensure amplified signal with low noise and enhanced detection capability.

2.
Development of a post-detection classification algorithm based on a novel deep neural networks (DNN) architecture.

3.
Creation of a comprehensive dataset through practical scenarios in a laboratory environment to facilitate algorithm development and evaluation.

4.
Application of the proposed algorithm to accurately classify detected targets as either anti-personnel mines (APMs) or non-APMs in both mineralized soil and non-mineralized environments (air or sand).

5.
Achievement of a high validation accuracy of 93.5% for the proposed novel algorithm on the dataset, showcasing its effectiveness in mine classification.

Rest of the paper is organized such that, the current systems’ limitations and problems are provided in Sect. “Pulse induction MMD”, followed by the discussion of data acquisition and the motivation of proposed algorithm in Sect. “Data acquisition”. The proposed AI-based classification algorithm is presented in Sect. “Proposed AI model”. Section “Simulations, results & discussions” will cover simulations and results, and finally, Sect. “Conclusion” will present the conclusion. To the best of our knowledge the work proposed in this paper has not been tackled in this manner in literature. We provide the open-source code and datasets (MinhasSF/MMD_AI (github.com)) for further improvements in a collaborative manner.

Pulse induction MMD
The PI detector used in this research has a single sensor coil, comprising of multi-strands wire. The advantage of using single coil for transmitter & receiver (Tx/Rx) will ensure a same channel response for transmitted and received signals. The PI metal detector sends powerful, short bursts (pulses) of current through a coil of (multi-strands) wire which will magnetize the sensor coil at the rate of 1150 μs (approx.) for positive polarity pulses and similar is the case for negative polarity pulses. The reason for using the bipolar pulses is to reduce the risk of triggering magnetically-activated mines and booby-traps13. The pulse time of single transmission is about 45–65 μs to charge (magnetize) sensor coil and hence produce magnetic flux to charge magnetic material (targets) around it. When the pulse ends, charged coil demagnetizes and magnetic flux around it collapses producing flyback voltage of few hundred volts (i.e. sharp electrical spike) at sensor coil. Magnetic flux collapse induces eddy current in target material which oppose demagnetization of sensor coil. This opposed demagnetizing effect by a certain decay rate is a characteristic of metal content in the target7.

In case of mineralized soil, the field has magnetization properties similar to metal target. This decay rate is deterministic and can be referred to as its signature. The strength of this magnetized soil on the sensor coil can be larger than coming from a target with small metal content especially in an APM which can be less than 0.3 g or even 0.2 g. As per Geneva convention of amended protocol II14 on certain conventional weapons, an APM must have 8 g of detection metal plate but is not considered in this research. To detect the target APM buried in mineralized soil, machine learning tools are used by different mine detectors for example Vallon VMH3, VMH44 and others that generally learn this predictable signal and then removes it from the received signal and thus the remaining signal is due to the target. The machine learning employed is on spot learning that does the mineralized soil compensation (automatically) only. The machine learning employed is limited in scope and is typically without any hidden layers, this compensation will be referred to as machine learning based compensation.

To ensure reliability, the compensated signal (less mineralized signal) is integrated and after passing through post-processing algorithms, it is finally fed to a comparative threshold-based alarm generator. It is indeed a very reliable system (with negligible false negative), as the target metal content shape, size, type, depth and orientation that are fundamental to classification are put aside. Any information present within the compensated signal is not considered, only strength of integrated signal will define the target through sound (louder the sound the bigger is the target and vice versa). With this, the decision making of calling it a target of concern comes down to the experience and instincts of a trained user. Prior to integration, it is still crucial to extract additional information for classification to aid in the decision-making process.

At this point, it is relevant to discuss a case that uses post-integrated dataset for machine learning based classification in Ref8. For this a grid of [11 × 10] data points is created and a robotic arm is used to sweep the area of size 60 cm × 50 cm. The authors use a deep convolutional neural network composed of several non-linear transformations. Instead of extracting information from the received signal at a particular location within the grid, the whole received signal is integrated into one data point or one sample per location which leads to a limited performance. In this research the focus is on the received signal curve and the information present in it, will be discussed in the following.

Data acquisition
In order to understand the received signal, it is essential to first briefly describe the working of transmitter and receiver of PI MMD used in this research (block diagram shown in Fig. 1). Transmitter (Tx) works on pulse induction principle with high voltage charge pump and switched coil damping. Usually metal detectors Tx described as total loss system flyback voltage (energy produced due to magnetic field collapse) is damped through ohmic resistance causing loss of energy (see pp 143–145 of Ref15). In current system (Earle Model is followed16) Tx flyback energy is stored in capacitor(s). Hence, energy loss from flyback energy is conserved for coil charge in subsequent cycle where energy is transferred from charged capacitor(s) to coil in capacitor/inductor resonance with frequency of ½ 
π
 and then continues with normal low voltage charging to sustain the current (and magnetic field) across the coil. This method generates approximately constant current and constant magnetic field across coil during charging as opposed to negative exponential current in total loss system. (for details see Figs. 3 and 4 in Ref16).

Figure 1
figure 1
The block diagram of pulse induction MMD showing transmitter (Tx), receiver (Rx) and the sensor coil.

Full size image
At the end of charging cycle, coil current (magnetic flux) collapses creating voltage flyback which charges capacitor(s) for 2us to repeat the cycle (self). After 2us, rest of the energy is damped through ohmic resistance in switched damping circuit, to get shortest possible time constant that a target can have to be detected. Receiver blocking circuit is composed of bidirectional analog switch which isolates receiver (Rx) circuit with Tx circuit and also provides some resistance to high voltage signal.

The receiver (Rx) circuit (for detailed working see pp 75–77 Ref15), is composed of voltage clipping, conditioning and amplification. Voltage needs to be clipped to work in connivance with sensitive electronics along with signal conditioning which involves low pass filters. The received signal undergoes clamping and reversal as a result of the internal circuitry of the receiver switch and the positioning of diodes. For completeness, the received signal chain, depicted in Fig. 2, will be briefly discussed here.

Figure 2
figure 2
Signal chain of data acquisition system.

Full size image
The signal chain of the data acquisition system is depicted in Fig. 2, and a pictorial view of the signal at each stage is shown in Fig. 3. The system is divided into three parts: (a) preamplifier, (b) differential line driver, and (c) ADC driver. Before preamplifier receiver blocking will be briefly discussed, as the name implies, safeguards sensitive components in data acquisition systems from flyback voltage. It accomplishes this by establishing a resistive pathway alongside shunt clamping diodes, ensuring that the signal remains within the safe operating voltage range. In preamplifier stage, the incoming signal undergoes signal conditioning, which includes passing through a two-pole 25 MHz low-pass filter. After this initial step, it proceeds through a non-inverting amplifier with 22 times amplification, followed by another low-pass filter with a cut-off frequency of 720 kHz.

Figure 3
figure 3figure 3
Stage wise pictorial view of received signal of negative pulse at different stages of signal chain of data acquisition.

Full size image
The incoming signal is converted to a differential signal using a Fully Differential Amplifier (FDA) with a gain of − 1 while rejecting common-mode noise. This FDA functions as a Differential Signalling Line Driver similar to Low Voltage Differential Signalling (LVDS), ensuring that the signal is transmitted over a spiral cable wire with matched impedance accepted by termination resistor at the end spiral cable.

Hereafter, Fully Differential Amplifier (FDA) accepts a differential input and produces a differential output. It serves as the driving stage for a 16-bit Successive Approximation Register Analog-to-Digital Converter (SAR ADC) working at 2 Mega samples/s. Additionally, the signal path includes a single-pole RC filter of bandwidth of 380 kHz for charge transient created by ADC during sampling process. The resultant signal can be sampled using an ADC once it has settled to the ground with small “kink”.

The received signal shown in Fig. 4 is ideal as it does not include any noise and interference and also the shape of curve may vary as it is based on architecture of search-head of MMD, in our case coil and Tx/Rx assembly. The sensitive area of the curve starts (just after the “kink”) from left to right and is being marked in sections (i.e. from a1 to a2, b1 to b2, c1 to c2 & d1 to d2), where very low metal contents are mostly present in the initial sections. The decay rate is being calculated using signal processing techniques on different sections. The sections width, the start, the end, overlapping and redundancy etc., are few factors that determine the robustness of compensation. It can be done manually or through machine learning (mostly on spot learning); however, the scope of this paper is not to discuss the soil compensation or to remove mineralization effect.

Figure 4
figure 4
Received signal picked up by the sensor coil and has passed through pre-processing.

Full size image
The purpose of this paper is to go beyond compensation and to do mine/metal classification by fully utilizing the received signal curve in different environments i.e. mineralized and non-mineralized environments (air & sand). The reason for applying AI technique will also become evident later on in next section.

To ensure completeness, we will now address the critical aspect of achieving pinpoint accuracy. Typically, a metallic mine detector operates in two modes: normal and mineralized (on-spot learning). Once an object buried underground is detected and identified as the target of interest by a trained operator using these modes, the subsequent task is to precisely locate the target's position. This precision is achieved by sweeping the search-head, oriented parallel to the ground, of the MMD multiple times over the target from various angles. The strongest signal is typically detected when the target is positioned at the center of the coil or search-head, ensuring an accurate pinpoint location. Subsequently, a classification algorithm (which will be discussed in the following section) becomes essential to classify the detected target, in our case, an APM mine. For pinpointing, normal mode is recommended in this research i.e. without machine learning based soil compensation since it is replaced by a novel DL-MMD classifier.

Proposed AI model
Prior to delving into the algorithm architecture, we will first focus on the number of classes that are to be classified and the dataset. Typically, in soil compensation we have two classes one is air and the other is mineralized soil. Air means that either the sensor coil has no mineralized soil in its proximity (just air) or the soil in proximity is not mineralized. Mostly the first option is considered otherwise there can be a slight bias in decay rate due minute mineral content within non-mineralized soil. However, in this research both are considered and for the later one sand is used. The default shape of the curve in which nothing is in proximity of the sensor coil, it will be called air (class A). The other eight classes are when sensor coil is exposed to or in proximity of mineralized soil (class B), sand (class C), APM (class D), vertical paper pins (class E), APM in presence of mineralized soil (class F), APM in presence of sand (class G), vertical paper pins in presence of mineralized soil (class H) and vertical paper pins in presence of sand (class I). The class A dataset is shown by a matrix 
 as below:

(1)
where 
 is a vector containing the concatenated received signal coming from the positive and negative transmitted pulses. The total number of pulses per class are N and the total number of samples per pulse (inclusive of both positive & negative) are given by M. Figure 5 shows the 3D image of the digitally synced received signal that is used to populate the dataset 
, where x-axis shows the number of samples in a pulse, y-axis shows the amplitude in volts and z-axis shows the number of pulses. It can be observed from the figure that there is a slight variation from pulse to pulse that can be due to either thermal noise or due to any external weak interfering signal. The number of synced pulses shown are 665 of the negative pulses only (for simplicity) with time duration of 75 μs per pulse at a sampling rate of 2 MHz. However, the dataset 
 contains the data from both positive and negative pulses, obtained just after the kink i.e. 122 samples (61 μs) per pulse. Similar matrices 
 ,
 , 
, 
, 
,
, 
 and 
 are for other eight classes. For each class, there will be a one-hot encoded vector representing the label, which is represented by a matrix with dimensions [N × 9].

Figure 5
figure 5
The 3D synced received signal of air, the number of pulses is equal to 665 and number of samples per pulse is equal to 150. It is important to point out that only a limited number of samples are shown of a negative pulse at sampling rate of 2 MHz. Similar is the case for positive pulse.

Full size image
In the last section, we have discussed the soil compensation robustness briefly in which two areas of the received signal curve processing needs to be pointed out here i.e. overlapping of sections and redundancy of channels. The overlapping and redundancy are to ensure that the signature of mineralized soil does not get mixed with the signature of other materials that might have close decay rate. So, a multi-channel approach is already present in soil compensation mode. In addition to this, the presence of positive and negative pulses of PI system, increase the numbers of channels.

Exploiting the same concept of filtering, decay rate calculation, signal strength indication through integration, curve analysis, anomaly segregation (not discussed here) and statistical calculation over multiple pulses in time. In summary, the process entails capturing and extracting the local, temporal, and spatial patterns/features collectively. For this AI based algorithms are the most suitable to extract the full potential of information in the received signal. The primary advantage of using this model is transfer learning, which can be integrated with on-spot learning. The latter requires less than a minute to perform soil compensation in a field environment. The intended purpose is to utilize it in future scenarios for real-time learning of unseen classes. Here we will be applying a novel AI based model architecture (DL-MMD) comprising of one-dimensional CNN, as the signal is also one dimensional. The details of the layers within the model can be seen in Table 1 and network structure is illustrated in Fig. 6. The batch size is eight and it refers to the number of samples that are processed together before the model’s parameters are updated. Readers are encouraged to refer17 for detailed mathematical insight of CNN for 1-D data.

Table 1 Configuration parameters of DL-MMD model.
Full size table
Figure 6
figure 6
Network structure of DL-MMD.

Full size image
The functionality of each layer has been extensively documented and researched. However, we provide a brief overview of each layer to discuss their intended purpose in the given context. The first layer represents the input to the model which defines the shape and type of the data that will be fed into the model. In our case dataset from nine classes as represented in Eq. (1) for one of the classes is fed into the input layer. It is followed by 1D CNN layer that applies a set of filters to input data, capturing local patterns and extracting relevant features. The kernel size is (k1) 5 with stride (s) of 1, activation function used is Rectified Linear Unit (ReLU) and the number of channels is 36. The selection of the number of channels has been thoroughly optimized. Any increase in the number of channels has led to overfitting. It is essential to point out that information in each sample within the received signal curve is not statistically independent from its neighbouring samples as there is a certain amount of correlation due to the process of demagnetizing of sensor coil. For this reason, fully connected layer is not considered at the start as it will lead to overfitting.

AveragePooling1D layer (pool/window size = 5) will perform averaging on the local regions of the output of the previous convolutional layer, allowing the network to capture the average activation within each region. Setting the stride of one will not reduce the dimensionality of data, so maximum information moves to next layer i.e., batch normalization. For the extraction of higher-level features and patterns, again Conv1D layer is used (with reduced 18 channels) and is followed by MaxPooling1D (pool size = 5) with stride of one that selects the maximum value within each pooling window. A third Conv1D layer is added with 18 channels but a different kernel size (k2) 9 (actually double of k1) with similar stride (s) of 1 is used. It is followed by MaxPooling1D layer with the same pool size and the stride length i.e. 9.

Finally, the flatten layer converts the multi-dimensional representation into a one-dimensional vector, preparing it to be fed into a fully connected (FC) layer i.e. Dense layer. This fully connected layer contains 32 neurons and connects each neuron in previous layer to every neuron within this layer and uses ReLU as activation function. It is then connected to the fully connected (FC) layer at the end, which employs the Softmax activation function and is designed for the classification of inputs into one of nine classes in a supervised manner. By computing probabilities using the provided hot encoded vector labels, it enables the determination of the most likely class for each input using Adamax optimizer. The next section will highlight the result from this learning model architecture.

Simulations, results & discussions
The experimental arrangement in MMD is a prime factor that defines the integrity of the dataset. The dataset is obtained in lab environment with a PI sensitive coil made up of muti-stranded wire with coil diameter of 170 mm. It is mounted on a transparent acrylic sheet with a miniaturized Tx/Rx (also mounted) at a distance of 100 mm. The electromagnetic field (EMF) simulation of search-head in close proximity of mine is shown in Fig. 7. The received signal is digitized, and synchronized data is obtained for both the transmitted positive and negative pulses. The dataset is then populated with this synchronized pulse data. The pulse repetition frequency, including both pulses, is 880 Hz.The number of pulses M (refer to Eq. (1)) obtained per class is 1330, representing concatenated positive and negative pulses. It is done to simplify the model, with a total number of concatenated samples being N = 244, consisting of 122 samples from each received pulse, respectively. It is approximately 3 s of pulsed data per class.

Figure 7
figure 7
Shows Electromagnetic field simulation of search head in (a) and search head in proximity of mine in (b).

Full size image
The samples/targets used to represent the nine classes (previously discussed) include minrl/brick (mineralized soil), sand (non-mineralized soil), APM (standard 0.2 gm) and vertical paper pins (0.2 gm). Mineralization is an indication of magnetic permeability (or susceptibility) of the surface soils that have been exposed to high temperatures and heavy rainfall or water for extended periods of time, often exhibit high mineralization due to the presence of residual iron components. For an in-depth exploration of the magnetic susceptibility across a wide range of soil types, you can find comprehensive information in reference18. The choice of using brick, a clay-based material, as a representative sample for mineralized soil is grounded in its unique composition. It contains minerals like iron oxide, such as magnetite or hematite, and exhibits relatively low electrical conductivity19. These distinctive characteristics significantly enhance its detectable response when subjected to a MMD. In fact, this response is typically more robust than that of conventional mineralized soil (from which it originates) or even APM. For the sake of simplicity and consistency, we will refer to this material as "minrl" throughout this paper.

All of the targets mentioned pose their own challenges, but they are placed in close proximity to the MMD, within a distance of no more than 20 mm parallel to the surface of the coil. The targets are positioned at the center of the coil. The received signals from different target samples of a positive and a negative transmitted pulses can be observed in Figs. 8 and 9 respectively. The figures display a magnified section of the received signal, focusing on the initial samples that are more strongly influenced by the secondary magnetic field compared to later samples. It can also be seen that signals vary in opposite directions as per polarity of the transmitted pulses.

Figure 8
figure 8
Received signals of a positive transmitted pulse picked up at the sensor coil from the secondary magnetic field produced by the eddy currents induced within the targets. The x-axis shows few numbers of samples (initial part of the signal) per pulse and y-axis shows amplitude of the signal in volts. Signals from nine targets air, APM, pins, minrl, minrl + APM, minrl + pins, sand, sand + APM and sand + pins have been shown.

Full size image
Figure 9
figure 9
Received signals of a negative transmitted pulse picked up at the sensor coil from the secondary magnetic field produced by the eddy currents induced within the targets. The x-axis shows few numbers of samples (initial part of the signal) per pulse and y-axis shows amplitude of the signal in volts. Signals from nine targets air, APM, pins, minrl, minrl + APM, minrl + pins, sand, sand + APM and sand + pins have been shown.

Full size image
The overall dataset comprises a total of 11,970 pulses, representing nine different classes. The dataset is sufficiently diverse, as illustrated in Fig. 10 by examining inter-class distances. For this analysis, two distances are employed: Euclidean distance, which measures point-to-point distance, and Bhattacharyya distance, a metric indicating dissimilarity between two probability distributions. Two cases will be briefly discussed here: one involving the Euclidean distance between air and pins, where the maximum distance is observed as depicted in Fig. 10, which is also evident in the received signal shown in Figs. 8 and 9. The second case pertains to the Bhattacharyya distance between air and sand, illustrating minimal dissimilarity. The impact of this dissimilarity will become evident in the overall results. To prepare this dataset for modelling, these pulses are randomly shuffled and subsequently split into two separate sets: a training dataset containing 10,773 pulses and a validation dataset comprising 1197 pulses.

Figure 10
figure 10
Shows inter-class similarity through Euclidean and Bhattacharyya distances.

Full size image
During the model training phase, input data is structured as a matrix with dimensions [10,773 × 244], and the output, following a supervised learning approach, is provided as a one-hot encoded labeled matrix with dimensions [10,773 × 9]. The accuracy of the trained model on the provided data is tracked across multiple epochs, including both training and validation accuracy. In the context of this training process, one “epoch” signifies a complete iteration over the entire training dataset of size [10,773 × 244], with all training samples processed by the model. Figure 11 depicts the trend, showing that as the training process repeats over multiple epochs, the model steadily enhances its performance and optimizes its parameters. After 4000 epochs, the trained accuracy reaches approximately 98%, while the validation accuracy hovers above 93%. It also shows that the DL-MMD model has more or less converged at 4000 epochs, by achieving the optimum training performance. Likewise, it’s evident that the model’s error loss diminishes with the progression of epochs, as illustrated in Fig. 12.

Figure 11
figure 11
Shows the accuracy and validation accuracy of novel DL-MMD model versus epochs. For comparison, the validation accuracy of KNN and SVM classifier are also shown for k = 8 and C = 100 respectively.

Full size image
Figure 12
figure 12
Shows the loss and validation loss of novel DL-MMD model versus epochs.

Full size image
Figure 11, also shows that the presented model performs substantially better compared to support vector machine (SVM) and K-Nearest Neighbors (KNN) classifiers. The main working principle of SVM is to separate several classes in the training set with a surface that maximizes the margin (decision boundary) between them. It uses Structural Risk Minimization principle (SRM) that allows the minimization of a bound on the generalization error20. SVM model used in this research achieved a training accuracy of 93.6% and a validation accuracy of 86.5%, which is far lower than the performance achieved by the presented model. The parameter for kernel function used is the most popular i.e. radial basis function (RBF) and the value of regularization parameter c optimally selected is 100. The regularization parameter controls the trade-off between classifying the training data correctly and the smoothness of the decision boundary. Figure 13 shows the influence of the regularization parameter c, on the performance of the classifier. The gamma is automatically calculated based on the inverse of the number of features, which ensures that each feature contributes equally to the decision boundary. The hyperparameter optimization is achieved through a manual grid search method. The code iterates through a predefined list of C values [0.1, 1, 10, 100, 1000, 10000], and for each value of C, it trains a Support Vector Machine (SVM) classifier with a radial basis function (RBF) kernel and evaluates its performance on the training and test sets. The accuracy and C values are then plotted to visually check the best performance. It can be seen that the generalization error increases when the value of C is greater than 100, the SVM starts to overfit the training data and thus resulting in decrease in validation accuracy.

Figure 13
figure 13
Shows the accuracy of SVM classifier versus regularization parameter C.

Full size image
While K-Nearest Neighbors (KNN) model with 8 neighbors (k) achieved a training accuracy of 92.6% and a validation accuracy of 90.7% (see Fig. 11), which is lower than the performance achieved by the presented model. To enable comparative analysis, it is essential to showcase the performance of this non-parametric machine learning algorithm. In this context, the algorithm predicts the value of a new data point by considering the majority vote or average of its k nearest neighbors within the feature space21. Figure 14 illustrates the influence of the hyperparameter k, the number of neighbors, on the performance of the algorithm. The graph demonstrates that the validation accuracy reaches a maximum of 90.7% when 8 neighbors are considered.

Figure 14
figure 14
Shows the accuracy of KNN classifier versus number of neighbors k.

Full size image
To further analyze the DL-MMD model versus the experimental data, one more graph has been plotted shown in Fig. 15. This graph illustrates the comparative performance of the presented model using a different data split ratio (70–30), with 70% for training and 30% for validation. The graph shows a slightly degraded performance when compared to the split ratio (90–10) of 90% for training and 10% for validation. However, it still shows validation accuracy of above 88% at 4000 epochs. This degradation is attributed to epistemic uncertainty (model uncertainty) due to slightly less effective learning on a reduced training data and as the training data increases, this uncertainty also reduces.

Figure 15
figure 15
Shows the accuracy and validation accuracy of novel DL-MMD model versus epochs at two different data split ratios i.e. of 90–10 and 70–30.

Full size image
The performance of the model can also be inferred from the confusion matrix shown in Fig. 16. It provides a tabular representation of the predicted and actual class labels, giving a very important analysis of the models in terms of true positives, true negatives, false positives, and false negatives. For an application perspective of an MMD, safety of the user is of utmost importance for which false negative matters a lot since mine as target must not be missed.. The overall prediction accuracy is above 93.5%, however, for cases of air and sand it is approximately 85 and 86.5% respectively, inferred from the confusion matrix. These two classification cases of relatively less prediction accuracy can be neglected since sand being wrongly classified as air only and vice-versa. These two classes (air & sand) do not trigger any detection alarm by an MMD, thus misclassification of them will not impact efficiency of DL-MMD classifier. It also highlights the fact that sand (of river) has minimal mineralized content and is generally designated as non-mineralised soil. It is therefore difficult to separate the boundary between these two classes in presence of noise and interference.

Figure 16
figure 16
Confusion matrix of the proposed DL-MMD classification on 9 classes.

Full size image
In addition to this, two further cases need to be examined: one involves mineralized soil (minrl) being wrongly classified as APM, and the other involves APM in sand (sand + APM) being wrongly classified as minrl. The first case is of false positive, it will generate a false alarm and will waste time of the user by requiring unnecessary further investigation. The second case is of more importance i.e. of false negative where an APM is detected but wrongly classified by a DL-MMD and will be discussed in next section. Apart from them, there are minor cases e.g. an APM misclassified as APM in sand (sand + APM), it will not have any impact since target of concern (APM) will remain the same but now being shown buried in sand. The occurrence of all these misclassification cases (apart from the air/sand case & vice-versa) is less than 5% approximately.

These results have been obtained by a substantial dataset based on actual data acquired in two sets of 665 (pulses per class) each obtained at two different times through the experimental setup explained previously and then combined together. Comprehensive simulations have been carried out in the Tensor Flow environment for evaluation of the proposed method. In addition to this, the algorithm has been extensively tested with an increased number of layers and channels, resulting in overfitting. Furthermore, the proposed model has been tested with different optimizers, such as Adagrad, Adamax, and Adam. The comparative analysis of Adam and Adamax can be seen in Fig. 17. Both show equivalent performance after 2000 epochs.

Figure 17
figure 17
Shows the accuracy and validation accuracy of novel DL-MMD model versus epochs using two different optimizers Adamax and Adam.

Full size image
In addition to the aforementioned analysis, the dataset underwent evaluation using other prevalent classification algorithms22, which utilize the principle of ensemble learning. However, upon comparison, the proposed deep learning architecture exhibited superior performance, achieving an accuracy exceeding 90%. The confusion matrices of these classification algorithms, AdaBoost and Bagged tree, are depicted in Figs. 18, 19, and 20, with the dataset partitioned into an 80/20 ratio, resulting in accuracies of 75.4%, 80%, and 83.3%, respectively. AdaBoost was employed without PCA, utilizing the maximum number of splits and learners set to 30, with a learning rate of 0.1. For Bagged tree, only Model 2 underwent preprocessing with PCA with a variance of 95%. They both utilized the same number of learners as AdaBoost and a maximum split of 11,969.

Figure 18
figure 18
Confusion matrix model 1 AdaBoost.

Full size image
Figure 19
figure 19
Confusion matrix model 2 Bagged Tree.

Full size image
Figure 20
figure 20
Confusion matrix model 3 Bagged Tree.

Full size image
It is pertinent to mention that there is always redundant information within the received signal that creates background bias, especially in sensitive areas with low metal content. Information regarding the detection of APM mines buried at different depths is available (in the parameter decay rate), but it is not utilized. Therefore, for an APM buried at a different depth (relative to the search head) to the one it is trained on, there is a chance that it can be misclassified. The information exists, but it needs to be pre-processed before feeding the signal to the model. One approach could be to use focused AI models, similar to those shown in Ref23, that inject synthetic bias into the signal to generalize the model in our case at different depths. Another approach can be to localize the area with different decay rates, similar to the one shown in Ref24 for 2D image application. One of the future work will be to utilize this information and integrate it into the DL_MMD architecture.

Conclusion
In this paper, we present a new approach using deep learning for distinguishing between anti-personnel mines (APM) and small vertical paper pins (0.2 g) in both mineralized soil and non-mineralized environments like air or sand. Our method is particularly important for practical demining operations, aiming to minimize both false positives and false negatives. This ensures more accurate target classification, ultimately expanding the search area.

Our innovative learning framework combines the pulse induction method with Convolutional Neural Networks (CNN) to act as a classifier for post-detection targets, providing a unique advantage. It has shown success in classifying mineralized soil and has the potential for extension to other soil types. The system demonstrates an impressive validation accuracy of 93.5% across nine distinct classes.

However, a challenge remains in accurately classifying APM, especially in scenarios like an APM buried in sand (sand + APM), where there is a 5.2% false negative rate, often misclassified as mineralized soil (minrl). To address this, we suggest that the operator’s decision should not solely rely on the deep learning (DL) model’s classification, given its post-detection nature. Instead, the operator’s experience using the Mine Metal Detector (MMD) should be considered. Despite this challenge, the DL model boasts a high confidence rate of 94.8% in correctly classifying the target (sand + APM), providing valuable information to enhance operational efficiency. For fully automated mine detection, a multisensory approach can ensure complete reliability in classifying mines based on metal content and material density.

To further enhance our model’s accuracy and expand its classification capabilities, future work will involve incorporating a broader range of mine types and increasing the magnetic footprint through spatial measurement diversity by introducing another target position, such as the side of the coil. This expansion will double the dataset with minimal modification to the DL-MMD algorithm. An advantage of using DL-MMD is that standard mine characteristics, such as metal content, size, shape, and orientation, remain consistent regardless of burial location. This consistency significantly boosts the algorithm’s accuracy and reliability in real-world scenarios.