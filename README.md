My lovely instructions that you can totally follow:
#everything that is a comment will be.. commented like so : #

#Just in case you need to do the whole ssh thing beforehand...

$ sudo apt install openssh-server

#this is a comment, um, the line below is to check if you have an ssh setup already, if it comes up with a file please continue unitl you reach "Assuming you have ssh installed"

$ ls -last ~/.ssh/

$ ssh-keygen

#make a password of a four words strung together thats easy for you to remember!

$ ssh localhost

#logout using

$ exit

#This next part is so that you can essentially bypass your password

$ chmod 600 ~/.ssh/authorized_keys

$ ssh localhost

#^ that should ask you for that password, once done you can exit using

$ exit

#If you want, you can set up so you can later do this on another device using

$ ~/.ssh/config

#Then type the below into the script

ForwardX11 yes
ForwardAgent yes

#save that and then in the terminal type

$ chmod 600 ~/.ssh/config

# once thats done you can go through github ot however you store repos and under settings you should find SSH/GPG keys.
# Click 'Add key' 
# To get your key, type

$ cat .ssh/id_ed25519.pub

#After that you should get your public key that you can now paste into github or whatever platform you are using. Then just hit the add key and your done!

#Assuming you have ssh installed:

$ git clone git@github.com:ItsAMeMario8/superllama.git

#First you pull using:

$ git pull

#After that you can use my one and only python file to run an RNN (Recurrent Neural Network) simulation. This script is to train the RNN on data to later predict future data, either synthisized or real.
My Lovely Project
