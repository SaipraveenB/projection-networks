# projection-networks

This repository documents the ACPNN (Action-Conditional Projection Neural Network) architecture, which is designed specifically to
simulate the natural mage formation system. Instead of using standard Convolutional networks, which generally work best for 
image formation/classification, this project tests a new architecture that has specific units structured to simulating the way a 3D scene 
is projected onto a 2D image.

Our goal for the network is to be able to predict the next frame, given an action (W,S,A,D). Our only inputs are the previous actions and the 'rendered' image.

The following two images describe the image projection operation.

<img src="https://github.com/SaipraveenB/projection-networks/blob/master/perspective-diagram.png" width="512">
<img src="https://github.com/SaipraveenB/projection-networks/blob/master/perspective.png" width="512">

The typical convolutional architecture used to deal with this type of reconstruction 

<img src="https://github.com/SaipraveenB/projection-networks/blob/master/conv-deconv.PNG" width="512">

Our architecture to deal with this problem: 

<img src="https://github.com/SaipraveenB/projection-networks/blob/master/acpnn-relu-dqn-2.png" width="512">

The ACPNN operates on UV texture coordinates rather than using convolutional layers are inputs. While CNNs are good at extracting features for classification, they are not ideal for reconstruciton operations. For this reason, we attempt to use the UV indices of eqach pixel as an input.

An example synthetic 2D world is shown below (The rendered 1D image is shown below). The black dot and the line represetnt the position and direction of the agent.

<img src="https://github.com/SaipraveenB/projection-networks/blob/master/doubleline-wp.png" width="512">

The steady improvement of the validation and testing error (The tests are based on mean squared error of randomly generated paths)

<img src="https://github.com/SaipraveenB/projection-networks/blob/master/inpacpnn_wo_conv.jpg" width="512">
