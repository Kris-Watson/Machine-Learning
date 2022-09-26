Trains a resnet18 neural network on stop sign data stored in "Data".
Build a conda environment using spec-file.txt to run the project.

A resnet neural network takes the output from a node and passes it not only to the current layer + 1, but also current layer + 2.
This diminishes the effect of the vanishing gradient problem and allows neural networks to be trained significantly faster.

You can use your own image and check if it has a stop sign on it:
1) Simply add your file to the main directory
2) Navigate to the bottom of the file and uncomment it.
3) Change myPath = <filename> (Where <filename> is the location of your file from the current working directory).
4) Run all the cells below the comment block!.
