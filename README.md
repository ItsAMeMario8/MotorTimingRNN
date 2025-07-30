My lovely instructions that you can totally follow:
#everything that is a comment will be.. commented like so : #

git clone git@github.com:ItsAMeMario8/superllama.git

#First you should pull using:

git pull

#After you pull origin, you can use my one and only python file to run an RNN (Recurrent Neural Network) simulation. Before you can run this you will need to open a terminal and either through conda or pip. In this README I will go through pip because that's how I did it. First you need a virtual environment for pip:

python3 -m venv venv

source .venv/bin/activate

pip -m ensurepip --upgrade

#Once you have pip you can then install the packages needed for the RNN:

pip install numpy

pip install pandas

pip install matplotlib

pip install torch torchvision torchaudio

pip install neurogym 

#From there I would run the file you cloned using python3 in the virtual environment. Just a breif overveiw of the RNN. In this case I am running an LSTM () on a motor timing dataset. The RNN will train on this data and can then change it's weights to produce an output as close to the data as possible. The first graph given is it's loss over the training time. The second is the activity of hidden layers that is used to store previous input data to help make predictions later. The last graph shown will be the weights that the RNN has reshaped as it trained using the eigenvalues of the output tensor. This current model has 81 hidden layers and it's hidden layer dimentions are set to 15. Upping the dimention will make the training long but it will give better results. Changing the number of hidden layers is much easier, the only thing is that it must produce a positive integer when square rooted.

pdflatex MotorTiming.tex

#With this you can build and read the paper as a pdf
