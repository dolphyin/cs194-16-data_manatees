#!/bin/bash

# Change the IP address with each restart.
# When you have your own username, i.e. after you have created individual user accounts on the EC2 instance,
# make a copy of this script and change the "NAME" field to your own account name.
# Keep the original of this script, since "ec2-user" will still be your group's master account, and has sudo access. 

IPADDR=54.149.49.34
NAME=ec2-user
LHOST=localhost
SSHKEY=mykey.pem          # change if your key is called something else

for i in `seq 8888 8900`; do
	    FORWARDS[$((2*i))]="-L"
	        FORWARDS[$((2*i+1))]="$i:${LHOST}:$i"
	done

	ssh -i mykey.pem -X ${FORWARDS[@]} -l ${NAME} ${IPADDR}
